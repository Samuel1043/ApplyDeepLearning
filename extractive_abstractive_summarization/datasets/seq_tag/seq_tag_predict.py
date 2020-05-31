from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.nn as nn
import torch
import sys
import json


class TestData(Dataset):
    def __init__(self,testData,remove_arr,max_len=300):
        testX=[]
        bound=[]
        padding_idx=testData.padding
        for idx,i in enumerate(testData):
            if idx in remove_arr:
                continue
            text=i['text']
            if(len(text)<max_len):
                text=text+[padding_idx]*(max_len-len(text))
            testX.append(text)
            bound.append(i['sent_range'])
        self.testX=testX
        self.bound=bound
    def __len__(self):
        return len(self.testX)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        return np.array(self.testX[idx],dtype=np.long)
    
    def get_bound(self,idx,batch_size):
        return self.bound[idx*batch_size:(idx+1)*batch_size]
    
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


def load_json_data(file_name):
    data=[]
    with open(file_name,'r') as r:
        line=r.readline()
        while line:
            data.append(json.loads(line))
            line=r.readline()
    return data
    
def predict2sentence_extract_two_predict(data,pred_arr):
    prediction=[]
    assert len(pred_arr)==len(data)
    for p,j in zip(pred_arr,data):
        sent_bounds = {i: bound for i, bound in enumerate(j['sent_bounds'])}
        predict_sent=''
        p['predict_sentence_index']=p['predict_sentence_index'][0:2]
        for sent_idx in p['predict_sentence_index']:
            start, end = sent_bounds.get(sent_idx, (0, 0))
            predict_sent += j['text'][start:end]
        prediction.append(predict_sent)
    return prediction

test_path=sys.argv[1]

testData=load_json_data(test_path)
testdata=np.load('./datasets/seq_tag/test.pkl',allow_pickle=True)
embedding_pkl=np.load('./data/embedding.pkl',allow_pickle=True)
embedding=embedding_pkl.vectors


# test_remove=[7421, 10381, 10606, 11367, 13617, 14054, 15613, 15755, 17445]

batch_size=64
extract_test=TestData(testdata,[])
testLoader=DataLoader(extract_test,batch_size=batch_size,num_workers=1)


hidden_size=embedding.shape[1]

# load model and predict
model=ExtractiveSeqTagging(hidden_size,embedding).cuda()
model.load_state_dict(torch.load('./datasets/seq_tag/state_dict/model30.pth'))
model.eval()

pred=[]
for data in testLoader:
    hidden=model.initHidden(data.shape[0])
    pred.append(model.predict(data.cuda(),(hidden,hidden)).cpu().detach().numpy())

# postprocessing
pred_arr=[]
cnt_val=0
for idx,i in enumerate(pred):
    result=np.argmax(i,2).T
    batch_bounds=extract_test.get_bound(idx,batch_size)
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
        prediction['id']=testData[cnt_val]['id']

        #post processing 設定 
        prediction['predict_sentence_index']=[int(candidate[i]) for i in np.argsort(-values)][0:2]

        cnt_val+=1
        pred_arr.append(prediction)

#write file
with open(sys.argv[2],'w') as w:
    for idx,i in enumerate(pred_arr):
        if(idx==len(pred_arr)):
            w.write(json.dumps(i))
        else:
            w.write(json.dumps(i)+'\n')