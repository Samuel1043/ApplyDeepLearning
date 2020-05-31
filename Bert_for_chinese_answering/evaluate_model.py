from torch.utils.data import DataLoader,TensorDataset
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import json
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils.tokenize_data import TestGenerate,prepare_trainV2
from utils.model import BertForQAS
import argparse


def features2dataset(features):
    all_input_ids = torch.tensor(
        [f.input_id for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_id for f in features], dtype=torch.long)
    all_context_mask = torch.tensor([[-np.inf]+[0]*f.context_end_id+[-np.inf]*(
        512-f.context_end_id-1) for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_context_mask
    )

    return dataset


def post_process(input_id, start_prop, end_prop, strategy,start_end=40,topk=30):
    sentence = ""
    if(strategy == 'probs max'):
        start_pos = start_prop.argmax()
        end_pos = end_prop.argmax()

        sentence = ''.join(tokenizer.convert_ids_to_tokens(
            input_id[start_pos:end_pos]))
        sentence=sentence.replace('##','')
        return sentence
        
    elif(strategy == 'start end threshold'):
        set_top = topk
        int_topk=False
        start_top = start_prop.topk(set_top)
        end_top = end_prop.topk(set_top)
        best_probs = (-1, [-1, -1])
        for start_prob, start_pos in zip(start_top[0], start_top[1]):
            for end_prob, end_pos in zip(end_top[0], end_top[1]):

                if(0 < end_pos-start_pos <= start_end):
                    if(best_probs[0] < start_prob+end_prob):
                        best_probs = (start_prob.item()+end_prob.item(),
                                      [start_pos.item(), end_pos.item()])

        if(best_probs[0] != -1):
            sentence = ''.join(tokenizer.convert_ids_to_tokens(
                input_id[best_probs[1][0]:best_probs[1][1]]))
            sentence=sentence.replace('##','')
            in_topk=True
        return sentence,in_topk
    else:
        print('err no_strategy used')
    



def load_json(path):
    with open(path, 'r') as r:
        line = r.readline()
        data = json.loads(line)
    return data


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=True,
        help="The output directory where predictions will be written.",
    )

    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. (in json format) ",
    )
    parser.add_argument(
        "--max_answer_length",
        default=40,
        type=int,
        help="The maximum length of an answer that can be generated.",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        default=None,
        type=str,
        help="The pretrain model checkpoint file to evaluate",
    )
    parser.add_argument(
        "--answerable_threshold",
        default=0.5,
        type=float,
        help="The answerable threshold for BCE",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="batch size for evaluating",
    )
    return parser.parse_args()


def evaluate(model,test,batch_size):

    

    test_features,test_dataset=test

    test_dataloader=DataLoader(test_dataset,batch_size=batch_size)
    test_epoch_iterator = tqdm(
            test_dataloader, desc="test Iteration", position=0, leave=True)

    model.eval()
    predictions_post = {}
    with torch.no_grad():
        for num, batch in enumerate(test_epoch_iterator):

            qa_ids = [
                i.qa_id for i in test_features[batch_size*num:(1+num)*batch_size]]
            batch = tuple(t.to('cuda') for t in batch)

            #output =( (batch,2) , (batch,max_context) , (batch,max_context) )
            outputs = model(batch[0], batch[1], batch[2])

            answerables = torch.sigmoid(outputs[0])
            start_props = F.softmax(outputs[1]+batch[3], 1)
            end_props = F.softmax(outputs[2]+batch[3], 1)
            
            count_topk=0
            count_unans=0
            for input_id, answerable, start_prop, end_prop, qa_id in zip(batch[0], answerables, start_props, end_props, qa_ids):
                if(answerable < args.answerable_threshold):
                    predictions_post[qa_id] = ''
                else:
                    predictions_post[qa_id],in_topk = post_process(
                        input_id, start_prop, end_prop, 'start end threshold',start_end=args.max_answer_length,topk=30)
                    count_unans+=1
                    if(in_topk==False):
                        count_topk+=1
    print('answerable not found in topk %.3f'%(count_topk/count_unans))

    with open(args.output_file,'w') as w:
        w.write(json.dumps(predictions_post,ensure_ascii=False))        


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    args=arg_parse()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)

    test=load_json(args.predict_file)
    testgen=TestGenerate(tokenizer)

    test=test['data']

    logging.info('tokenizing test data')
    test=testgen.convert_teststring2id(test)

    logging.info('prepare test features')
    test_features=prepare_trainV2(test,tokenizer,max_length=512,training=False)
    test_dataset=features2dataset(test_features)
    

    config = BertConfig()
    model = BertForQAS(config).cuda()

    model.load_state_dict(torch.load(args.pretrain_checkpoint))

    evaluate(model,(test_features,test_dataset),batch_size=args.batch_size)


