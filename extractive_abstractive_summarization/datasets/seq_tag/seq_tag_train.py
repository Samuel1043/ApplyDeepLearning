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
embedding=np.load('./embedding.pkl',allow_pickle=True)
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


revise_testData=load_json_data('../../data/test_revise.jsonl')
revise_trainData=load_json_data('../../data/train_revise.jsonl')
revise_valData=load_json_data('../../data/val_revise.jsonl')


batch_size=64
extract_train=TrainData(trainData,train_remove)
extract_val=TrainData(valData,val_remove)
trainLoader=DataLoader(extract_train,batch_size=batch_size,num_workers=1)
valLoader=DataLoader(extract_val,batch_size=batch_size,num_workers=1)




data_dim=embedding.shape[1]
hidden_size=data_dim
out_feat=2

num_epoch=50
lr=0.001
pretrain=torch.Tensor(embedding)


model=ExtractiveSeqTagging(hidden_size,pretrain).cuda()
criterion=nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([11.63]).cuda())
optimizer=optim.Adam(params=model.parameters(),lr=lr)
# model=ExtractiveSeqTagging(30rameters(),lr=lr)
loss=0
best_acc=0


fo = open("loss_train_adl_64_LSTMHID_1.txt", "w")

for epoch in range(1,num_epoch+1):
    epoch_start_time = time.time()
    train_loss=0
    val_loss=0
    pred_arr=[]
    cnt_train=0
    cnt_val=0
    model.train()
    
    for i,e in enumerate(trainLoader):
        hidden=model.initHidden(e[0].shape[0])
        optimizer.zero_grad()
        out=model(e[0].cuda(),(hidden,hidden))
        loss=criterion(out.transpose(0,1),e[1].cuda())
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        
        result=np.argmax(out.detach().cpu(),2).T.numpy()
        batch_bounds=extract_train.get_bound(i,batch_size)
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
            prediction['id']=trainData[cnt_train]['id']
            
            #post processing 設定 threshold
            prediction['predict_sentence_index']=[candidate[i] for i in np.argsort(-values)]
            
            cnt_train+=1
            pred_arr.append(prediction)
        progress = ('#' * int(float(i)/len(trainLoader)*40)).ljust(40)
        print('[%03d|%03d] %2.2f sec(s) | %s |' %(epoch,num_epoch,time.time()-epoch_start_time,progress),end='\r',flush=True)
    target_word,prediction_word=predict2sentence_extract_two(revise_trainData,pred_arr)
    scores_train=calculate_rouge_score(prediction_word,target_word)
    
    pred_arr=[]
    model.eval()
    with torch.no_grad():
        for i,e in enumerate(valLoader):

            hidden=model.initHidden(e[0].shape[0])
            out=model(e[0].cuda(),(hidden,hidden))
            loss=criterion(out.transpose(0,1),e[1].cuda())
            val_loss+=loss.item()
            
            result=np.argmax(out.detach().cpu(),2).T.numpy()
        
            batch_bounds=extract_val.get_bound(i,batch_size)
            
            for num,bounds in enumerate(batch_bounds):            
                prediction={}
                candidate=[]
                values=[]
                for num2,j in enumerate(bounds):
                    sentence_cnt=np.sum(result[num,:][j[0]:j[1]])
                    if(sentence_cnt>0):
                        candidate.append(num2)
                        values.append(sentence_cnt)
                prediction['id']=valData[cnt_val]['id']
                
                values=np.array(values)
                candidate=np.array(candidate)
                
                prediction['predict_sentence_index']=[candidate[i] for i in np.argsort(-values)]
                cnt_val+=1
                pred_arr.append(prediction)
                
            progress = ('#' * int(float(i)/len(valLoader)*40)).ljust(40)
            print('[%03d|%03d] %2.2f sec(s) | %s |' %(epoch,num_epoch,time.time()-epoch_start_time,progress),end='\r',flush=True)
    
    target_word,prediction_word=predict2sentence_extract_two(revise_valData,pred_arr)
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
        torch.save(model.state_dict(), './state_dict/model'+str(epoch)+'.pth')
        
fo.close()
    
    
