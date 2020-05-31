from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn




class BertForQAS(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.bert = BertModel.from_pretrained('bert-base-chinese')
    self.answerable=nn.Sequential(nn.Linear(config.hidden_size,1))
    self.answer_start=nn.Sequential(nn.Linear(config.hidden_size,1))
    self.answer_end=nn.Sequential(nn.Linear(config.hidden_size,1))
  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None):
    inputs = {
          "input_ids":input_ids,
          "attention_mask": attention_mask,
          "token_type_ids": token_type_ids,
          
      }
    #out=(batch_size, sequence_length, hidden_size)
    out,_=self.bert(**inputs)
    
    #(batch,1,2)
    answerable=self.answerable(out[:,0,:])
    #(batch,seq_len,1)
    answer_end=self.answer_end(out)
    answer_start=self.answer_start(out)

    #(batch,2),  (batch,seq_len),(batch,seq_len)  
    return answerable.squeeze(1),answer_start.squeeze(2),answer_end.squeeze(2)

