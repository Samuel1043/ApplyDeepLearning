import torch.nn as nn
import torch
import numpy as np
import json
from torch.utils.data import DataLoader,Dataset
import sys


import torch.nn as nn
class Seq2SeqAbstractiveEncoder(nn.Module):
    def __init__(self,hidden_size,pretrain_embedding,bidirectional=True):
        super(Seq2SeqAbstractiveEncoder,self).__init__()
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding.from_pretrained(pretrain_embedding)
        self.lstm=nn.LSTM(hidden_size,hidden_size,bidirectional=bidirectional)
        if bidirectional==True:
            self.out=nn.Linear(hidden_size*2,hidden_size)
        
    def forward(self,x,hidden):
        embedded=self.embedding(x).transpose(0,1)
        out,hidden=self.lstm(embedded,hidden)
        out_hidden=hidden[0][0]
#         concat=torch.cat((hidden[0][0],hidden[0][1]),1)
#         out_hidden=self.out(concat)
#         out_hidden=nn.functional.relu(out_hidden)
        
        return out,out_hidden
#     h_0 of shape (num_layers * num_directions, batch, hidden_size): 
    def initHidden(self,batch):
        shape=(1,batch,self.hidden_size)
        if self.bidirectional==True:
            shape=(2,batch,self.hidden_size)
        
        return torch.zeros(shape,device='cuda')
    def predict(self,x,hidden):
        embedded=self.embedding(x).transpose(0,1)
        
        out=self.lstm(embedded,hidden)
        
        out=self.out(out)
        out=nn.functional.softmax(out)
        return out
    
import torch.nn as nn
class Seq2SeqAbstractiveDecoder(nn.Module):
    def __init__(self,hidden_size,output_size,pretrain_embedding,bidirectional=False):
        super(Seq2SeqAbstractiveDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding.from_pretrained(pretrain_embedding)
        self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,bidirectional=bidirectional)
        self.out=nn.Linear(hidden_size,output_size)
    #key (encoder_output)
    def forward(self,x,prev_hidden,key):
        embedded=self.embedding(x).transpose(0,1)
        out,hidden=self.gru(embedded,prev_hidden)
        key=self.attn(key).permute(1,0,2)
        value=key
        query=out.permute(1,2,0)
        attn_weight=nn.functional.softmax(torch.bmm(key,query),1)
#         out=torch.sum(torch.mul(attn_weight,key),1).unsqueeze(1)
        out=torch.bmm(attn_weight.transpose(1,2),value)
        out=self.out(out)
        
        out = nn.functional.log_softmax(out,2)
        return out,hidden,attn_weight
#     h_0 of shape (num_layers * num_directions, batch, hidden_size): 
    def initHidden(self,batch):
        shape=(1,batch,self.hidden_size)
        if self.bidirectional==True:
            shape=(2,batch,self.hidden_size)
        
        return torch.zeros(shape,device='cuda')
    def predict(self,x,hidden):
        embedded=self.embedding(x).transpose(0,1)
        
        out=self.lstm(embedded,hidden)
        
        out=self.out(out)
        out=nn.functional.softmax(out)
        return out


class TestData(Dataset):
    def __init__(self,testData,remove_arr,max_len=300):
        testX=[]
        padding_idx=testData.padding
        for idx,i in enumerate(testData.data):
            if idx in remove_arr:
                continue
            text=i['text'][0:300]
            if(len(text)<max_len):
                text=text+[padding_idx]*(max_len-len(text))
            testX.append(text)
        self.testX=testX
    def __len__(self):
        return len(self.testX)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        return np.array(self.testX[idx],dtype=np.long)
    
def predict2sentence_abstractive_predict(data,pred_arr,embedding_vocab):
    cnt=0
    prediction=[]
    np_vocab=np.array(embedding_pkl.vocab)
    for p in pred_arr:
        
        for pred in p:
            predin={}
            eos=len(pred)
            predin['id']=data[cnt]['id']
            for idx,i in enumerate(pred):
                if(i==2):
                    eos=idx
                    break
            pred=pred[1:eos]
            pred=list(filter(lambda a: a != 3, pred))

            predin['predict']=' '.join(np_vocab[pred])
            cnt+=1
            prediction.append(predin)
    return prediction
    

def load_json_data(file_name):
    data=[]
    with open(file_name,'r') as r:
        line=r.readline()
        while line:
            data.append(json.loads(line))
            line=r.readline()
    return data


test_path=sys.argv[1]

testData=load_json_data(test_path)
testdata=np.load('./datasets/seq2seq/test.pkl',allow_pickle=True)
embedding_pkl=np.load('./data/embedding_seq2seq.pkl',allow_pickle=True)
embedding=embedding_pkl.vectors

batch_size=100
extract_test=TestData(testdata,[])
testLoader=DataLoader(extract_test,batch_size=batch_size,num_workers=1)


#reproduce result
hidden_size=embedding.shape[1]
output_size=embedding.shape[0]
encoder=Seq2SeqAbstractiveEncoder(hidden_size,embedding).cuda()
decoder=Seq2SeqAbstractiveDecoder(hidden_size,output_size,embedding).cuda()
encoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict_attn/model_encoder13.pth'))
decoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict_attn/model_decoder13.pth'))
encoder.eval()
decoder.eval()

pred_arr=[]
for i,e in enumerate(testLoader):
    hidden=encoder.initHidden(e.shape[0])
    encoder_outputs,encoder_hidden=encoder(e.cuda(),(hidden,hidden))

    decoder_key=encoder_outputs
    decoder_hidden=encoder_hidden.unsqueeze(0)
    
    #80
    target_len=80

    #sos
    decoder_input=torch.LongTensor(e.shape[0],1).fill_(2).cuda()

    use_teacher_forcing=True
    pred_word=[]

    if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
        for di in range(target_len):
            decoder_output, decoder_hidden,attn_weight = decoder(decoder_input, decoder_hidden,encoder_outputs)
            pred_word.append(decoder_output.transpose(0,1).topk(1)[1].detach().cpu().numpy())
            decoder_input = decoder_output.transpose(0,1).topk(1)[1].squeeze(0)  # Teacher 
    pred_arr.append(np.array(pred_word).squeeze().T)

pred_test=predict2sentence_abstractive_predict(testData,pred_arr,embedding_pkl)


with open(sys.argv[2],'w') as w:
    for idx,i in enumerate(pred_test):
        if(idx==len(pred_test)):
            w.write(json.dumps(i))
        else:
            w.write(json.dumps(i)+'\n')