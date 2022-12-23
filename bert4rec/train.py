import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from args import parser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import *
from dataset import *
import random
import os

def random_neg(l, r, s):
    # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# 시드 고정용
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  

# 모델 세이브
def save_model(model):
    args = parser.parse_args()
    check_point = {
        'net': model.state_dict()
    }
    torch.save(check_point, args.model_path)
    
class Trainer():
    def __init__(self,data_loader,user_train,user_valid,num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, batch_size,mask_prob,num_epochs,lr,device):
        self.user_train =  user_train
        self.data_loader = data_loader
        self.user_valid = user_valid
        self.num_user = num_user
        self.num_item =  num_item
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self. batch_size = batch_size
        self.mask_prob = mask_prob
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr

    def train(self):
        seed_everything(42)

        model = BERT4Rec(self.num_user, self.num_item, self.hidden_units, self.num_heads, 
                         self.num_layers, self.max_len, self.dropout_rate, self.device)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        
        max_score = -1e9
        min_loss = 1e9
        early_stopping = 0
        print("Model Train Start...")
        for epoch in range(1, self.num_epochs + 1):
            tbar = tqdm(self.data_loader)
            for step, (log_seqs, labels) in enumerate(tbar):
                model.train()
                logits = model(log_seqs)
                
                # size matching
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1).to(self.device)
                
                optimizer.zero_grad()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
            print(f"Train no.{epoch} : Loss {loss:0.5f}")
        

            NDCG = 0.0 # NDCG@10
            HIT = 0.0 # HIT@10

            num_item_sample = 100
            num_user_sample = 1000
            users = np.random.randint(0, self.num_user, num_user_sample) # 1000개만 sampling 하여 evaluation
            model.eval()
            for u in users:
                seq = (self.user_train[u] + [self.num_item + 1])[-self.max_len:]
                rated = set(self.user_train[u] + self.user_valid[u])
                item_idx = [self.user_valid[u][0]] + [random_neg(1, self.num_item + 1, rated) for _ in range(num_item_sample)]

                with torch.no_grad():
                    predictions = - model(np.array([seq]))
                    predictions = predictions[0][-1][item_idx] # sampling
                    rank = predictions.argsort().argsort()[0].item()
                
                if rank < 10: # @10
                    NDCG += 1 / np.log2(rank + 2)
                    HIT += 1 
            print(f"Train no.{epoch} >>> Loss {loss:0.5f} | NDCG@10 {NDCG/num_user_sample:0.5f} | HIT@10: {HIT/num_user_sample:0.5f}")      
            if (loss < min_loss) & ((HIT/num_user_sample) > max_score):
                save_model(model)
                min_loss = loss
                max_score = HIT/num_user_sample
                early_stopping = 0
                print(f"Best Score Appear...Loss {min_loss:0.5f} | HIT@10 {max_score:0.5f}")
            else:
                early_stopping += 1
            
            if early_stopping == 20:
                print("Early Stop!...")
                print(f"Best Score...Loss {min_loss:0.5f} | HIT@10 {max_score:0.5f}")
                break


            
    
    def inference(self,user2idx,item2idx):
        print("Model Inference Start...")
        args = parser.parse_args()
        user_list = []
        item_list = []
        model = BERT4Rec(self.num_user, self.num_item, self.hidden_units, self.num_heads, self.num_layers, self.max_len, 
                         self.dropout_rate, self.device)
        
        checkpoint = torch.load(args.model_path, map_location=self.device)
        state_dict = checkpoint['net']
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        for u in range(self.num_user):
            # 전체 유저에 대해
            seq = (self.user_train[u] + self.user_valid[u]+ [self.num_item + 1])[-self.max_len:] 
            # 시퀀스
            rated = set(self.user_train[u] + self.user_valid[u])
            # train+valid = total watched
            item_idx = [i+1 for i in  range(self.num_item)]
            # 전체 아이템에 대해 ~ +1하는 이유는 첫 예측은 제외되어야하는 부분이기에
            # 따라서 index 기준 prediction에서 0인덱스는 1아이템이라고 보면 됨

            with torch.no_grad():
                predictions = -model(np.array([seq]))
                # 마지막 시퀀스에 대한 예측
                predictions = predictions[0][-1][item_idx] # sampling
                predictions[[i-1 for i in rated]] = np.inf
                # rated에 있는 애들은 제외, -1하는 이유는 인덱스는 0이지만 아이템 인덱싱은 1부터 시작하기에
                rank = predictions.argsort().argsort().cpu().numpy()
                item_list.append(np.where(rank < 10)[0]+1)
                user_list.append([u]*10)
                
                
        u_ = np.concatenate(user_list)
        i_ = np.concatenate(item_list)
        submit_df = pd.DataFrame(data={'user': u_, 'item': i_}, columns=['user', 'item'])
        submit_df['user'] = submit_df['user'].apply(lambda x : user2idx.index[x])
        submit_df['item'] = submit_df['item'].apply(lambda x : item2idx.index[x])
        
        print("Done...!")
        return submit_df