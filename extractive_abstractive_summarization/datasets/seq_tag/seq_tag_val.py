#reproduce validation result

from tensorflow.keras.utils import to_categorical
from rouge_score.rouge_scorer import RougeScorer
from multiprocessing import Pool,cpu_count
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.nn as nn
import torch
import sys
import json

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib




# val_remove=[1147, 7085, 8520, 10125]


class TrainData(Dataset):
    def __init__(self,trainData,remove_arr,max_len=300):
        trainX=[]
        trainY=[]
        bound=[]
        padding_idx=trainData.padding
        for idx,i in enumerate(trainData):
            if idx in remove_arr:
                continue
            text=i['text']
            label=i['label']
            
            if(len(text)<max_len):
                text=text+[padding_idx]*(max_len-len(text))
                label=label+[padding_idx]*(max_len-len(label))
            trainX.append(text)
            label=to_categorical(label,num_classes=2)
            trainY.append(label)
            bound.append(i['sent_range'])
        self.trainX=trainX
        self.trainY=trainY
        self.bound=bound
    def __len__(self):
        return len(self.trainX)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        return np.array(self.trainX[idx],dtype=np.long),np.array(self.trainY[idx],dtype=np.float)
    
    def get_bound(self,idx,batch_size):
        return self.bound[idx*batch_size:(idx+1)*batch_size]
    
def predict2sentence_extract_two(data,pred_arr):
    target=[]
    prediction=[]
    assert len(pred_arr)==len(data)
    for p,j in zip(pred_arr,data):
        target.append(j['summary'])
        sent_bounds = {i: bound for i, bound in enumerate(j['sent_bounds'])}
        predict_sent=''
        p['predict_sentence_index']=p['predict_sentence_index'][0:2]
        for sent_idx in p['predict_sentence_index']:
            start, end = sent_bounds.get(sent_idx, (0, 0))
            predict_sent += j['text'][start:end]
        prediction.append(predict_sent)
    return target,prediction




def calculate_rouge_score(prediction,target):
    ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
    USE_STEMMER = False


    rouge_scorer = RougeScorer(ROUGE_TYPES, use_stemmer=USE_STEMMER)
    with Pool(cpu_count()) as pool:
        scores = pool.starmap(rouge_scorer.score,
                            [(t, p) for t, p in zip(target, prediction)])

    r1s = np.array([s['rouge1'].fmeasure for s in scores])
    r2s = np.array([s['rouge2'].fmeasure for s in scores])
    rls = np.array([s['rougeL'].fmeasure for s in scores])
    scores = {
        'mean': {
            'rouge-1': r1s.mean(),
            'rouge-2': r2s.mean(),
            'rouge-l': rls.mean()
        },
        'std': {
            'rouge-1': r1s.std(),
            'rouge-2': r2s.std(),
            'rouge-l': rls.std()
        },
    }
    return scores

import json
def load_json_data(file_name):
    data=[]
    with open(file_name,'r') as r:
        line=r.readline()
        while line:
            data.append(json.loads(line))
            line=r.readline()
    return data

class ExtractiveSeqTagging(nn.Module):
    def __init__(self,hidden_size,pretrain_embedding,bidirectional=True):
        super(ExtractiveSeqTagging,self).__init__()
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding.from_pretrained(pretrain_embedding)
        self.gru=nn.LSTM(hidden_size,hidden_size,bidirectional=bidirectional)
        if bidirectional==True:
            self.out=nn.Linear(hidden_size*2,2)
    
    def forward(self,x,hidden):
        embedded=self.embedding(x).transpose(0,1)
        out,_=self.gru(embedded,hidden)
        out=self.out(out)
        return out
#     h_0 of shape (num_layers * num_directions, batch, hidden_size): 
    def initHidden(self,batch):
        shape=(1,batch,self.hidden_size)
        if self.bidirectional==True:
            shape=(2,batch,self.hidden_size)
            
        return torch.zeros(shape,device='cuda')
    def predict(self,x,hidden):
        embedded=self.embedding(x).transpose(0,1)
        
        
        out,_=self.gru(embedded,hidden)
        out=self.out(out)
        out=nn.functional.softmax(out,2)
        return out


def relative_plot(valData,pred_arr):
    assert len(revise_valData)==len(pred_arr)
    relative_dict=[]

    for i,j in zip(revise_valData,pred_arr):
        sentence_cnt=len(i['sent_bounds'])
        relative_dict+=(list(np.array(j['predict_sentence_index'][:2])/sentence_cnt))
    cnt=Counter(relative_dict)
    x=[]
    y=[]
    for i in cnt.items():
        x.append(i[0])
        if(i[0]<0):
            print(i)
        y.append(i[1])
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 20}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    plt.bar(x,height=y,width=0.03)
    ax.set_xlim(0,1)
    ax.set_xlabel('relative location')
    ax.set_ylabel("density")

valData=np.load('./datasets/seq_tag/valid.pkl',allow_pickle=True)
embedding_pkl=np.load('./data/embedding.pkl',allow_pickle=True)
embedding=embedding_pkl.vectors
revise_valData=load_json_data(sys.argv[1])

batch_size=64
extract_val=TrainData(valData,[])
valLoader=DataLoader(extract_val,batch_size=batch_size,num_workers=1)



hidden_size=embedding.shape[1]
model=ExtractiveSeqTagging(hidden_size,embedding).cuda()
model.load_state_dict(torch.load('./datasets/seq_tag/state_dict/model30.pth'))

model.eval()

pred=[]

for data in valLoader:
    hidden=model.initHidden(data[0].shape[0])
    pred.append(model.predict(data[0].cuda(),(hidden,hidden)).cpu().detach().numpy())


pred_arr=[]
cnt_val=0
for idx,i in enumerate(pred):
    result=np.argmax(i,2).T
    batch_bounds=extract_val.get_bound(idx,batch_size)
    for num,bounds in enumerate(batch_bounds):            
        prediction={}
        candidate=[]
        values=[]
        for num2,j in enumerate(bounds):
            sentence_cnt=np.sum(result[num,:][j[0]:j[1]])
            if(sentence_cnt>0):
                candidate.append(num2)
                values.append(sentence_cnt)
        values=np.array(values)
        candidate=np.array(candidate)
        prediction['id']=valData[cnt_val]['id']

        #post processing 設定 threshold
        prediction['predict_sentence_index']=[candidate[i] for i in np.argsort(-values)]

        cnt_val+=1
        pred_arr.append(prediction)

relative_plot(valData,pred_arr)

target_word,prediction_word=predict2sentence_extract_two(revise_valData,pred_arr)
scores_train=calculate_rouge_score(prediction_word,target_word)
print(scores_train)