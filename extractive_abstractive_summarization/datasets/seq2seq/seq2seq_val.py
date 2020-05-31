#reproduce valid result  $valid.json 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np 
import json
import sys
from rouge_score.rouge_scorer import RougeScorer
from multiprocessing import Pool,cpu_count
class TrainData(Dataset):
    def __init__(self,trainData,remove_arr,max_len_text=300,max_len_sum=80):
        trainX=[]
        trainY=[]
        attention=[]
        padding_idx=0
        for idx,i in enumerate(trainData):
            if idx in remove_arr:
                continue
            text=i['text'][:max_len_text]
            label=i['summary'][:max_len_sum]
#             attention.append(i['attention_mask'])
            
            if(len(text)<max_len_text):
                text=text+[padding_idx]*(max_len_text-len(text))
            if(len(label)<max_len_sum):
                label=label+[padding_idx]*(max_len_sum-len(label))
            trainX.append(text)
            trainY.append(label)
        self.trainX=trainX
        self.trainY=trainY
        self.attention=attention
    def __len__(self):
        return len(self.trainX)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        return np.array(self.trainX[idx],dtype=np.long),np.array(self.trainY[idx],dtype=np.long)
    
    def get_attention(self,idx,batch_size):
        return self.attention[idx*batch_size:(idx+1)*batch_size]


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
        concat=torch.cat((hidden[0][0],hidden[0][1]),1)
        out=self.out(concat)
        out=nn.functional.relu(out)
        
        return out
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
        self.gru=nn.GRU(hidden_size,hidden_size,bidirectional=bidirectional)
        self.out=nn.Linear(hidden_size,output_size)
    
    def forward(self,x,prev_hidden):
        embedded=self.embedding(x).transpose(0,1)
        out,hidden=self.gru(embedded,prev_hidden)
        out=self.out(out)
        out = nn.functional.log_softmax(out,2)
        return out,hidden
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

def predict2sentence_abstractive_valid(data,pred_arr,embedding_vocab):
    target=[]
    cnt=0
    prediction=[]
    np_vocab=np.array(embedding_pkl.vocab)
    for p in pred_arr:
        
        for pred in p:
            eos=len(pred)
            target.append(data[cnt]['summary'])
            for idx,i in enumerate(pred):
                if(i==2):
                    eos=idx
            pred=pred[1:eos]
            prediction.append(' '.join(np_vocab[pred]))
            cnt+=1
    return target,prediction

def load_json_data(file_name):
    data=[]
    with open(file_name,'r') as r:
        line=r.readline()
        while line:
            data.append(json.loads(line))
            line=r.readline()
    return data


val_path=sys.argv[1]

valData=load_json_data(val_path)
valdata=np.load('./datasets/seq2seq/valid.pkl',allow_pickle=True)
embedding_pkl=np.load('./data/embedding_seq2seq.pkl',allow_pickle=True)
embedding=embedding_pkl.vectors


#reproduce result
hidden_size=embedding.shape[1]
output_size=embedding.shape[0]
encoder=Seq2SeqAbstractiveEncoder(hidden_size,embedding).cuda()
decoder=Seq2SeqAbstractiveDecoder(hidden_size,output_size,embedding).cuda()
encoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict/model_encoder7.pth'))
decoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict/model_decoder7.pth'))
encoder.eval()
decoder.eval()

batch_size=100

extract_val=TrainData(valdata,[])
valLoader=DataLoader(extract_val,batch_size=batch_size,num_workers=1)



#reproduce valid result
hidden_size=embedding.shape[1]
output_size=embedding.shape[0]
encoder=Seq2SeqAbstractiveEncoder(hidden_size,embedding).cuda()
decoder=Seq2SeqAbstractiveDecoder(hidden_size,output_size,embedding).cuda()
encoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict/model_encoder7.pth'))
decoder.load_state_dict(torch.load('./datasets/seq2seq/state_dict/model_decoder7.pth'))
encoder.eval()
decoder.eval()

pred_arr=[]
for i,e in enumerate(valLoader):
    hidden=encoder.initHidden(e[0].shape[0])
    encoder_outputs=encoder(e[0].cuda(),(hidden,hidden))
    decoder_hidden=encoder_outputs.unsqueeze(0)

    #80
    target_len=80

    #sos
    decoder_input=torch.LongTensor(e[0].shape[0],1).fill_(2).cuda()

    use_teacher_forcing=True
    pred_word=[]

    if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
        for di in range(target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            pred_word.append(decoder_output.topk(1)[1].detach().cpu().numpy())
            decoder_input = decoder_output.topk(1)[1].squeeze(0)  # Teacher 
    pred_arr.append(np.array(pred_word).squeeze().T)
    
target,prediction=predict2sentence_abstractive_valid(valData,pred_arr,embedding_pkl.vocab)
scores_val=calculate_rouge_score(prediction,target)
print(scores_val)