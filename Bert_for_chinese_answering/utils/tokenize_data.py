
import logging
from tqdm import tqdm


class QuestionAnswer(object):
    def __init__(self,
                 input_id,
                 attention_mask,
                 token_type_id,
                 answer_bond,
                 answerable,
                 answer,
                 qa_id,
                 context_end_id
                 ):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.token_type_id = token_type_id
        self.answer_bond = answer_bond
        self.answerable = answerable
        self.answer = answer
        self.qa_id = qa_id
        self.context_end_id = context_end_id


class TestQuestion(object):
    def __init__(self,
                 input_id,
                 attention_mask,
                 token_type_id,
                 qa_id,
                 context_end_id
                 ):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.token_type_id = token_type_id
        self.qa_id = qa_id
        self.context_end_id = context_end_id


def prepare_train(train_json, tokenizer, max_context_len, max_query_len, question_start, context_start, bert_len=512):
    examples = []
    assert max_context_len+max_query_len+3 == bert_len
    for id, i in enumerate(tqdm(train_json)):
        for id0, paragraph in enumerate(i['paragraphs']):
            context_id = paragraph['context_id']
            for id1, qa in enumerate(paragraph['qas']):
                answer = qa['answers'][0]
                qa_id = qa['id']
                answerable = qa['answerable']
                qa_context_id = context_id

                if(len(context_id) > max_context_len):
                    # get $context_start of previous words of answer bond
                    if(qa['answer_bond'][0]-context_start > 0):
                        qa_context_id = context_id[qa['answer_bond']
                                                   [0]-context_start:]
                        # cls answer bond +1
                        answer_bond = [qa['answer_bond'][0]-(qa['answer_bond'][0]-context_start)+1, qa['answer_bond'][1]-(
                            qa['answer_bond'][0]-context_start)+1]
                    else:
                        answer_bond = [qa['answer_bond']
                                       [0]+1, qa['answer_bond'][1]+1]
                else:
                    answer_bond = [qa['answer_bond']
                                   [0]+1, qa['answer_bond'][1]+1]

                context_encode = tokenizer.encode_plus(
                    qa_context_id, max_length=max_context_len, pad_to_max_length=True, add_special_tokens=False)

                qa_question_id = qa['question_id']

                question_encode = tokenizer.encode_plus(
                    qa_question_id, max_length=max_query_len+2, pad_to_max_length=True, add_special_tokens=True)
                # type 1 input
                # attention_mask=[1]+context_encode['attention_mask']+[1]+question_encode['attention_mask']+[1]
                # input_id=tokenizer.encode_plus(context_encode['input_ids'],question_encode['input_ids'],add_special_tokens=True,return_attention_mask=False)
                # token_type_id=input_id['token_type_ids']
                # input_id=input_id['input_ids']

                # type 2 input
                # input_id [cls,context_seq...,pad...,sep,query_seq....,pad....,sep]
                # mask     [1(cls),1(context)....,1(pad)..,1(sep),1(query)....1(sep),0(pad)...]
                # token_type [0(cls)..........................0(sep),1........1(sep),0(pad)...]
                attention_mask = [1]*(max_context_len+2) + \
                    question_encode['attention_mask'][1:]
                input_id = [101]+context_encode['input_ids'] + \
                    [102]+question_encode['input_ids'][1:]
                token_type_id = [0]*(max_context_len+2) + \
                    question_encode['attention_mask'][1:]

                train_ans = ''.join(tokenizer.convert_ids_to_tokens(
                    input_id[answer_bond[0]:answer_bond[1]]))

                # _check_bound_special_char(train_ans, qa, id, id0, id1)
                assert len(input_id) == bert_len == len(
                    token_type_id) == len(attention_mask)

                example = QuestionAnswer(
                    input_id,
                    attention_mask,
                    token_type_id,
                    answer_bond,
                    answerable,
                    answer,
                    qa_id
                )
                examples.append(example)
    return examples


def prepare_trainV2(train_json, tokenizer, max_length=512, training=True):
    examples = []
    del_samples = 0
    for id, i in enumerate(tqdm(train_json)):
        for id0, paragraph in enumerate(i['paragraphs']):
            context_token_ids = paragraph['context_id']
            for id1, qa in enumerate(paragraph['qas']):
                question_token_ids = qa['question_id']
                out = tokenizer.prepare_for_model(
                    context_token_ids, question_token_ids, max_length=max_length,
                    truncation_strategy='only_first', pad_to_max_length=True)

                input_ids = out['input_ids']
                token_type_ids = out['token_type_ids']
                attention_mask = out['attention_mask']

                context_end_id = 0
                for idx, token_type_id in enumerate(token_type_ids):
                    if(token_type_id == 1):
                        context_end_id = idx-2
                        break
                qa_id = qa['id']

                assert input_ids[context_end_id+1] == 102
                if training:
                    if(qa['answer_bond'][1] < context_end_id):
                        answerable = qa['answerable']

                        if(answerable == True):
                            answer_bond = [qa['answer_bond']
                                           [0]+1, qa['answer_bond'][1]+1]
                        else:
                            answer_bond = [-1, -1]
                        answer = qa['answers'][0]

                        train_ans = ''.join(tokenizer.convert_ids_to_tokens(
                            input_ids[answer_bond[0]:answer_bond[1]]))
                        # _check_bound_special_char(train_ans, qa, id, id0, id1)

                        example = QuestionAnswer(
                            input_ids,
                            attention_mask,
                            token_type_ids,
                            answer_bond,
                            answerable,
                            answer,
                            qa_id,
                            context_end_id
                        )
                        examples.append(example)
                    else:
                        del_samples += 1

                else:
                    example = TestQuestion(
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        qa_id,
                        context_end_id
                    )
                    examples.append(example)

    if(training):
        print('del %d entry' % (del_samples))
    return examples


def _tokenize_convert2id(sentence, tokenizer):
    logging.getLogger(
        "transformers.tokenization_utils").setLevel(logging.ERROR)
    # token=tokenizer.tokenize(sentence)
    # tokenid=tokenizer.convert_tokens_to_ids(token)
    tokenid = tokenizer.encode(sentence, add_special_tokens=False)
    return tokenid


def convert_string2id(indata, tokenizer):
    data = indata
    data = _sort_ans_start_id(data['data'])

    for idx, i in enumerate(tqdm(data)):
        for idx0, paragraph in enumerate(i['paragraphs']):
            cur_id = 0
            context_id = _tokenize_convert2id(paragraph['context'], tokenizer)
            token_id = []
            for idx1, qa in enumerate(paragraph['qas']):
                qa['question_id'] = _tokenize_convert2id(
                    qa['question'], tokenizer)
                if(qa['answerable']):
                    ans_len = len(qa['answers'][0]['text'])
                    ans_start = int(qa['answers'][0]['answer_start'])

                    token_id += _tokenize_convert2id(
                        paragraph['context'][cur_id:ans_start], tokenizer)
                    token_ans_id = _tokenize_convert2id(
                        paragraph['context'][ans_start:ans_start+ans_len], tokenizer)
                    token_id = token_id+token_ans_id
                    ans_token_len = len(token_ans_id)
                    qa['answer_bond'] = [
                        len(token_id)-ans_token_len, len(token_id)]
                    cur_id = ans_start+ans_len
                else:
                    qa['answer_bond'] = [0, 0]

                if(idx1 == len(paragraph['qas'])-1):
                    token_id += _tokenize_convert2id(
                        paragraph['context'][cur_id:], tokenizer)

            if(len(token_id) != len(context_id)):
                # print(idx,idx0,idx1,len(context_id),len(token_id))
                pass
            paragraph['context_id'] = token_id

    return data


def check_ans_bound(data, tokenizer):
    for id, i in enumerate(data):
        for id0, paragraph in enumerate(i['paragraphs']):
            context_id = paragraph['context_id']
            for id1, qa in enumerate(paragraph['qas']):
                ans = ''.join(tokenizer.convert_ids_to_tokens(
                    context_id[qa['answer_bond'][0]:qa['answer_bond'][1]]))
                _check_bound_special_char(ans, qa, id, id0, id1)


def _check_bound_special_char(ans, qa, id, id0, id1):
    if(ans != qa['answers'][0]['text']):
        ans_re = ans.replace('#', '')
        if(ans_re.lower() != qa['answers'][0]['text'].lower()):
            if '[UNK]' in ans_re:
                pass
            else:
                print(id, id0, id1, ans, qa['answers'][0]['text'])


def _sort_ans_start_id(data):
    for i in data:
        for j in i['paragraphs']:
            j['qas'] = sorted(j['qas'], key=lambda k: (
                k['answers'][0]['answer_start'], len(k['answers'][0]['text'])))
    return data



class TestGenerate(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def _tokenize_convert2id(self,sentence):
        token=self.tokenizer.tokenize(sentence)
        tokenid=self.tokenizer.convert_tokens_to_ids(token)
        return tokenid

    def convert_teststring2id(self,indata):
        data=indata
        for idx,i in enumerate(tqdm(data)):
            for idx0,paragraph in enumerate(i['paragraphs']):      
                context_id=self._tokenize_convert2id(paragraph['context'])
                for idx1,qa in enumerate(paragraph['qas']):
                    qa['question_id']=self._tokenize_convert2id(qa['question'])
                paragraph['context_id']=context_id
        
        return data

