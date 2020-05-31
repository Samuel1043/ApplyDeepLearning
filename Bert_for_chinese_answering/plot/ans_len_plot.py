import matplotlib.pyplot as plt
from transformers import BertTokenizer
import sys
import collections
import json
def convert_string2id(indata, tokenizer):
    data = indata
    ans_token_len_arr = []
    for idx, i in enumerate(data):
        for idx0, paragraph in enumerate(i['paragraphs']):
            cur_id = 0
            
            for idx1, qa in enumerate(paragraph['qas']):
                if(qa['answerable']):
                    ans_len = len(qa['answers'][0]['text'])
                    ans_start = int(qa['answers'][0]['answer_start'])

                    token_ans_id = tokenize_convert2id(
                        paragraph['context'][ans_start:ans_start+ans_len], tokenizer)
                    
                    ans_token_len = len(token_ans_id)
                    ans_token_len_arr.append(ans_token_len)

    return ans_token_len_arr

def tokenize_convert2id(sentence,tokenizer):
  token=tokenizer.tokenize(sentence)
  tokenid=tokenizer.convert_tokens_to_ids(token)
#   tokenid=tokenizer.encode(sentence,add_special_tokens=False)
  return tokenid

def load_json(path):
    with open(path, 'r') as r:
        line = r.readline()
        data = json.loads(line)
    return data

if __name__=='__main__':
    train=load_json(sys.argv[1])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
    ans_token_len=convert_string2id(train['data'],tokenizer)
    cnt=collections.Counter(ans_token_len)
    ans_len_cumalative=cnt.most_common()
    ans_len_cumalative.sort(key=lambda x:x[0])
    x=[]
    y=[]
    for i,j in ans_len_cumalative:
        prop=j/len(ans_token_len)
        x.append(i)
        y.append(prop)

    plt.figure(figsize=(5,5))
    plt.xlabel('answer token length')
    plt.ylabel('Count(%)')
    plt.title('cumalative answer length')
    plt.bar(x,y)
    plt.savefig(sys.argv[2])