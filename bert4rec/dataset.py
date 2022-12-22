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

class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len, mask_prob,batch_size):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob    
        self.batch_size = batch_size
        
    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        # 개별 user의 시퀀스
        seq = self.user_train[user]
        tokens = []
        labels = []
        for s in seq:
            prob = np.random.random() # TODO1: numpy를 사용해서 0~1 사이의 임의의 값을 샘플링하세요.
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # BERT 학습
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s)  # 학습에 사용
            else:
                tokens.append(s)
                labels.append(0)  # 학습에 사용 X, trivial
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)
    
def data_process(data_path):
    print("Data Load...")
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)

    # 유니한 item/use 리스트
    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    # item/user 수
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // args.batch_size

    # 인덱싱용 
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) 
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) 

    # 새롭게 인덱싱한거로 colunm 대체
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user'] 

    # train set, valid set 생성 ~ 각 유저별로 마지막 시퀀스를 valid로 가져감
    users = defaultdict(list) 
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    return user_train, user_valid, num_user, num_item, user2idx, item2idx