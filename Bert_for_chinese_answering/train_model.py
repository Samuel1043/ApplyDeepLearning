import tensorflow as tf
import random
from transformers import BertTokenizer, BertConfig
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch.optim as optim
import argparse
from utils.tokenize_data import convert_string2id, check_ans_bound, prepare_train, prepare_trainV2
from utils.model import BertForQAS
from utils.evaluate import Tokenizer, compute_metrics, collect_answers_fromfeat, collect_answers
import os

# def set_seed(num):
#     random.seed(num)
#     np.random.seed(num)
#     torch.manual_seed(num)
#     torch.cuda.manual_seed_all(num)


def features2dataset(features, training=True):
    all_input_ids = torch.tensor(
        [f.input_id for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_id for f in features], dtype=torch.long)
    all_context_mask = torch.tensor([[-np.inf]+[0]*f.context_end_id+[-np.inf]*(
        512-f.context_end_id-1) for f in features], dtype=torch.float)

    if training:
        all_answer_start = torch.tensor(
            [f.answer_bond[0] for f in features], dtype=torch.long)
        all_answer_end = torch.tensor(
            [f.answer_bond[1] for f in features], dtype=torch.long)
        all_answerable = torch.tensor(
            [1 if(f.answerable == True) else 0 for f in features], dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_answer_start,
            all_answer_end,
            all_answerable,
            all_context_mask
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_context_mask
        )

    return dataset


def post_process(input_id, start_prop, end_prop, strategy,topk=30,start_end=30):
    sentence = ""
    if(strategy == 'probs max'):
        start_pos = start_prop.argmax()
        end_pos = end_prop.argmax()

        sentence = ''.join(tokenizer.convert_ids_to_tokens(
            input_id[start_pos:end_pos]))
    elif(strategy == 'start end 30 top 30'):
        set_top = topk
        start_end = start_end

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
    else:
        print('err no_strategy used')

    sentence=sentence.replace('##','')
    return sentence


def train_model(model, train, dev, train_check, tokenizer, tokenizer_script, num_train_epochs=5, batch_size=4,lr=2e-5):

    train_dataset, train_ans = train
    dev_dataset, dev_ans = dev
    train_check_dataset, train_check_ans = train_check

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    train_check_dataloader = DataLoader(
        train_check_dataset, batch_size=batch_size)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    criterion_answerable = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.4])).cuda()
    criterion_answerstart = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    criterion_answerend = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    train_iterator = trange(0, num_train_epochs, desc="Epoch")
    

    # best_answerableacc = 0
    for epoch in train_iterator:
        train_epoch_iterator = tqdm(
            train_dataloader, desc="Train Iteration", position=0, leave=True)
        all_losses = []
        model.train()
        for batch in train_epoch_iterator:
            all_loss = 0

            model.zero_grad()

            batch = tuple(t.to('cuda') for t in batch)

            outputs = model(batch[0], batch[1], batch[2])

            # context_mask
            ans_start = outputs[1]+batch[6]
            ans_end = outputs[2]+batch[6]
            answerable_loss = criterion_answerable(outputs[0], batch[5])
            answerstart_loss = criterion_answerstart(ans_start, batch[3])
            answerend_loss = criterion_answerend(ans_end, batch[4])

            all_loss = answerable_loss+answerstart_loss+answerend_loss

            train_epoch_iterator.set_description("total: %.5f| answerable: %.3f| answerstart: %.3f| answer end: %.3f" \
                                           % (all_loss,answerable_loss,answerstart_loss,answerend_loss))

            all_losses.append((all_loss).item())
            all_loss.backward()
            optimizer.step()
        print('epoch:  %d  loss:%.5f '%(epoch+1,np.mean(all_losses)))
        
        train_check_epoch_iterator = tqdm(
            train_check_dataloader, desc="train_check Iteration", position=0, leave=True)
        model.eval()
        predictions = {}
        predictions_post = {}
        with torch.no_grad():
            for num, batch in enumerate(train_check_epoch_iterator):

                qa_ids = [
                    i.qa_id for i in train_features[batch_size*num:(1+num)*batch_size]]
                batch = tuple(t.to('cuda') for t in batch)

                #output =( (batch,2) , (batch,max_context) , (batch,max_context) )
                outputs = model(batch[0], batch[1], batch[2])

                answerables = outputs[0]
                start_props = F.softmax(outputs[1]+batch[3], 1)
                end_props = F.softmax(outputs[2]+batch[3], 1)

                for input_id, answerable, start_prop, end_prop, qa_id in zip(batch[0], answerables, start_props, end_props, qa_ids):
                    if(answerable < 0):
                        predictions[qa_id] = ''
                        predictions_post[qa_id] = ''
                    else:
                        predictions[qa_id] = post_process(
                            input_id, start_prop, end_prop, 'probs max')
                        predictions_post[qa_id] = post_process(
                            input_id, start_prop, end_prop, 'start end 40 top 50',start_end=args.max_answer_length)
            
            result_train_check = compute_metrics(
                train_check_ans, predictions, tokenizer_script)
            print(result_train_check)
            
            result_train_check_post = compute_metrics(
                train_check_ans, predictions_post, tokenizer_script)
            print(result_train_check_post)

        dev_epoch_iterator = tqdm(
            dev_dataloader, desc="dev Iteration", position=0, leave=True)
        model.eval()
        predictions = {}
        predictions_post = {}
        with torch.no_grad():
            for num, batch in enumerate(dev_epoch_iterator):

                qa_ids = [
                    i.qa_id for i in dev_features[batch_size*num:(1+num)*batch_size]]
                batch = tuple(t.to('cuda') for t in batch)

                #output =( (batch,2) , (batch,max_context) , (batch,max_context) )
                outputs = model(batch[0], batch[1], batch[2])

                answerables = outputs[0]
                start_props = F.softmax(outputs[1]+batch[3], 1)
                end_props = F.softmax(outputs[2]+batch[3], 1)

                for input_id, answerable, start_prop, end_prop, qa_id in zip(batch[0], answerables, start_props, end_props, qa_ids):
                    if(answerable < 0):
                        predictions[qa_id] = ''
                        predictions_post[qa_id] = ''
                    else:
                        predictions[qa_id] = post_process(
                            input_id, start_prop, end_prop, 'probs max')
                        predictions_post[qa_id] = post_process(
                            input_id, start_prop, end_prop, 'start end 30 top 30')
            
            result_dev = compute_metrics(
                dev_ans, predictions, tokenizer_script)
            print(result_dev)
            
            result_dev_post = compute_metrics(
                dev_ans, predictions_post, tokenizer_script)

            print(result_dev_post)
        torch.save(optimizer.state_dict(),'./save/model_optim_epoch%s.pth'%(str(epoch+1)))
        torch.save(model.state_dict(
        ), './save/model_epoch%s.pth' % (str(epoch+1)))

        with open('./save/performance_epoch%s.json' % (str(epoch+1)), 'w') as w:
            w.write('train losses: %.5f'%(np.mean(all_losses))+'\n\n\n')
            w.write(json.dumps(result_train_check)+'\n\n\n')
            w.write(json.dumps(result_train_check_post)+'\n\n\n')
            w.write(json.dumps(result_dev)+'\n\n\n')
            w.write(json.dumps(result_dev_post))
            


def load_json(path):
    with open(path, 'r') as r:
        line = r.readline()
        data = json.loads(line)
    return data

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default=None,
        type=str,
        required=True,
        help="The training data to train the model",
    )

    parser.add_argument(
        "--dev_data",
        default=None,
        type=str,
        help="The dev data to check the performance of model",
    )
    parser.add_argument(
        "--max_answer_length",
        default=40,
        type=int,
        help="The maximum length of an answer that can be generated.",
    )
    parser.add_argument(
        "--train_epoch",
        default=5,
        type=int,
        help="The number of epoch to train the model",
    )
    parser.add_argument(
        "--ckip_dir",
        default=None,
        type=str,
        help="The dir of ckip model",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="batch size for evaluating",
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help="learning rate for model training",
    )
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    args=arg_parse()

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-chinese', do_lower_case=True)
    # Added here for reproductibility
    # set_seed(9487)

    train = load_json(args.train_data)
    dev = load_json(args.dev_data)

    logging.info('tokenizing train data')
    train = convert_string2id(train, tokenizer)
    # check_ans_bound(train, tokenizer)

    logging.info('tokenizing dev data')
    dev = convert_string2id(dev, tokenizer)
    # check_ans_bound(dev, tokenizer)


    logging.info('prepare training features')
    train_features = prepare_trainV2(train, tokenizer)
    logging.info('total training data: %d'%(len(train_features)))

    logging.info('prepare dev features')
    dev_features = prepare_trainV2(dev, tokenizer, training=False)
    logging.info('total dev data: %d'%(len(dev_features)))

    tokenizer_script = Tokenizer(args.ckip_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    logging.info('loading bert pretrain model')
    config = BertConfig()
    model = BertForQAS(config).cuda()

    #shuffle traning dataset
    random.shuffle(train_features)

    train_dataset = features2dataset(train_features)
    train_ans = collect_answers_fromfeat(train_features)
    dev_dataset = features2dataset(dev_features, training=False)
    dev_ans = collect_answers(dev)

    # checking performance on traning set
    train_check_dataset = features2dataset(
        train_features[:100], training=False)
    train_check_ans = collect_answers_fromfeat(train_features[:100])

    print('start training...')
    train_model(model, (train_dataset, train_ans), (dev_dataset, dev_ans), (train_check_dataset,
                                                                            train_check_ans), tokenizer, tokenizer_script, num_train_epochs=args.train_epoch, batch_size=args.batch_size,lr=args.lr)
