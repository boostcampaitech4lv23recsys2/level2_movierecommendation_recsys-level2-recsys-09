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
from dataset import * 
from model import *
from train import * 


def main():
    # 데이터 로드
    args = parser.parse_args()
    user_train, user_valid, num_user, num_item, user2idx, item2idx = data_process(args.data_path)
    seq_dataset = SeqDataset(user_train, num_user, num_item, args.max_len, args.mask_prob, args.batch_size)
    data_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) 
    
    # 학습
    
    bertmodel = Trainer(data_loader,user_train,user_valid,num_user, num_item, args.hidden_units, 
                        args.num_heads, args.num_layers, args.max_len, args.dropout_rate, 
                        args.batch_size, args.mask_prob, args.num_epochs,args.lr ,args.device)

    bertmodel.train()
    
if __name__ == "__main__":
    main()
    
