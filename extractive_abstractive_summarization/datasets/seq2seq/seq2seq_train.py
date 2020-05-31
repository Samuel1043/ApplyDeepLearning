from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import json
from tensorflow.keras.utils import to_categorical
from rouge_score.rouge_scorer import RougeScorer
from multiprocessing import Pool,cpu_count
import torch.nn as nn
import torch.optim as optim
import time


trainData=np.load('./train.pkl',allow_pickle=True)
valData=np.load('./valid.pkl',allow_pickle=True)
testData=np.load('./test.pkl',allow_pickle=True)
embedding_pkl=np.load('./embedding.pkl',allow_pickle=True)
embedding=embedding.vectors

test_remove=[7421, 10381, 10606, 11367, 13617, 14054, 15613, 15755, 17445]
train_remove=[10318, 13123, 15367, 19945, 22458, 22993, 28541, 32094, 35147, 36770, 43614, 46068, 49182, 53332, 54932, 65569, 68450, 68494, 69867, 70103, 70858]
val_remove=[1147, 7085, 8520, 10125]



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
def remove_dead_entry(data):
    tmp=[]
    for index,i in enumerate(data):
        if(i['text']=='\n' or i['text']==''):
            tmp.append(index)
            
    for index in sorted(tmp, reverse=True):
        del data[index]
    print('remove entry:',tmp)
    return data
def write_revise_data(data,file_name):
    with open(file_name,'w') as w:
        for idx,i in enumerate(data):
            if(idx==len(data)-1):
                word=json.dumps(i)
            else:
                word=json.dumps(i)+'\n'
            w.write(word)



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

def predict2sentence_abstractive_valif(data,pred_arr,embedding_vocab):
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
            pred=pred[:eos+1]
            prediction.append(' '.join(np_vocab[pred]))
            cnt+=1
    return target,prediction



revise_testData=load_json_data('../../data/test_revise.jsonl')
revise_trainData=load_json_data('../../data/train_revise.jsonl')
revise_valData=load_json_data('../../data/val_revise.jsonl')


batch_size=64
extract_train=TrainData(trainData.data,train_remove)
extract_val=TrainData(valData.data,val_remove)

val_batch_size=64
trainLoader=DataLoader(extract_train,batch_size=batch_size,num_workers=1)
valLoader=DataLoader(extract_val,batch_size=val_batch_size,num_workers=1)



data_dim=embedding.shape[1]
hidden_size=data_dim
output_size=embedding.shape[0]

num_epoch=10
lr=0.001
pretrain=torch.Tensor(embedding)


encoder=Seq2SeqAbstractiveEncoder(hidden_size,pretrain).cuda()
decoder=Seq2SeqAbstractiveDecoder(hidden_size,output_size,embedding).cuda()
criterion=nn.NLLLoss()
encoder_optimizer=optim.Adam(params=encoder.parameters(),lr=lr)
decoder_optimizer=optim.Adam(params=decoder.parameters(),lr=lr)

loss=0
best_acc=0


fo = open("loss_train_adl_64_LSTMHID_1_seq2seqattn.txt", "w")

for epoch in range(1,num_epoch+1):
    epoch_start_time = time.time()
    train_loss=0
    val_loss=0
    pred_arr=[]
    
    encoder.train()
    decoder.train()
    
    for i,e in enumerate(trainLoader):
        loss=0
        hidden=encoder.initHidden(e[0].shape[0])
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs=encoder(e[0].cuda(),(hidden,hidden))
        
        decoder_hidden=encoder_outputs.unsqueeze(0)
        #80
        target_len=e[1].shape[1]
        
        #sos
        decoder_input=torch.LongTensor(e[0].shape[0],1).fill_(2).cuda()
        
        use_teacher_forcing=True
        pred_word=[]
        
        if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
            for di in range(target_len):
                print(decoder_input.shape,decoder_hidden.shape)
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                pred_word.append(decoder_output.topk(1)[1].detach().cpu().numpy())
                loss += criterion(decoder_output.squeeze(0), e[1][:,di].cuda())
                decoder_input = decoder_output.topk(1)[1].squeeze(0).cuda()  # Teacher forcing
        else:
            for di in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(0).detach()  # detach from history as input
                loss += criterion(decoder_output.squeeze(0), e[1][:,di].cuda())
                
                if decoder_input.item() == EOS_token:
                    break
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        train_loss+=loss.item()
        
        
        
        pred_arr.append(np.array(pred_word).squeeze().T)
        
        
        progress = ('#' * int(float(i)/len(trainLoader)*40)).ljust(40)
        print('[%03d|%03d] %2.2f sec(s) | %s |' %(epoch,num_epoch,time.time()-epoch_start_time,progress),end='\r',flush=True)
    target,prediction=predict2sentence_abstractive_valid(revise_trainData,pred_arr,embedding_pkl.vocab)
    prediction
    scores_train=calculate_rouge_score(prediction,target)
    
    
    pred_arr=[]
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for i,e in enumerate(valLoader):
            loss=0
            hidden=encoder.initHidden(e[0].shape[0])
            encoder_outputs=encoder(e[0].cuda(),(hidden,hidden))
            decoder_hidden=encoder_outputs.unsqueeze(0)
            
            #80
            target_len=e[1].shape[1]

            #sos
            decoder_input=torch.LongTensor(e[0].shape[0],1).fill_(2).cuda()

            use_teacher_forcing=True
            pred_word=[]

            if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
                for di in range(target_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    pred_word.append(decoder_output.topk(1)[1].detach().cpu().numpy())
                    
                    loss += criterion(decoder_output.squeeze(0), e[1][:,di].cuda())
                    decoder_input = e[1][:,di].unsqueeze(1).cuda()  # Teacher forcing
            else:
                for di in range(target_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(0).detach()  # detach from history as input
                    loss += criterion(decoder_output.squeeze(0), e[1][:,di].cuda())

                    if decoder_input.item() == EOS_token:
                        break
            pred_arr.append(np.array(pred_word).squeeze().T)
            val_loss+=loss.item()
            
            progress = ('#' * int(float(i)/len(valLoader)*40)).ljust(40)
            print('[%03d|%03d] %2.2f sec(s) | %s |' %(epoch,num_epoch,time.time()-epoch_start_time,progress),end='\r',flush=True)
    
    target_word,prediction_word=predict2sentence_abstractive_valid(revise_valData,pred_arr,embedding_pkl.vocab)
    scores_val=calculate_rouge_score(prediction_word,target_word)
    
    fo.write('[%03d|%03d] %2.2f sec(s) | train loss: %2.5f | rouge-1: %2.4f | rouge-2: %2.4f |rouge-l: %2.4f |val loss: %2.5f | rouge-1: %2.4f | rouge-2: %2.4f |rouge-l: %2.4f \n' \
          %(epoch,num_epoch,time.time()-epoch_start_time,train_loss/len(trainLoader.sampler),scores_train['mean']['rouge-1']\
            ,scores_train['mean']['rouge-2'],scores_train['mean']['rouge-l'],val_loss/len(valLoader.sampler),\
            scores_val['mean']['rouge-1'],scores_val['mean']['rouge-2'],scores_val['mean']['rouge-l'] \
            ))
    
    print('[%03d|%03d] %2.2f sec(s) | train loss: %2.5f | rouge-1: %2.4f | rouge-2: %2.4f |rouge-l: %2.4f |val loss: %2.5f | rouge-1: %2.4f | rouge-2: %2.4f |rouge-l: %2.4f' \
          %(epoch,num_epoch,time.time()-epoch_start_time,train_loss/len(trainLoader.sampler),scores_train['mean']['rouge-1']\
            ,scores_train['mean']['rouge-2'],scores_train['mean']['rouge-l'],val_loss/len(valLoader.sampler),\
            scores_val['mean']['rouge-1'],scores_val['mean']['rouge-2'],scores_val['mean']['rouge-l'] \
            ))
    if best_acc<scores_val['mean']['rouge-1']:
        best_acc=scores_val['mean']['rouge-1']
        torch.save(decoder.state_dict(), './state_dict/model_decoder'+str(epoch)+'.pth')
        torch.save(encoder.state_dict(), './state_dict/model_encoder'+str(epoch)+'.pth')
        
fo.close()
    
    
