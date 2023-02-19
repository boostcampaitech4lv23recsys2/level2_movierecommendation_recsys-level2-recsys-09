import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse

from model import NeuMF, loss_function_BCEW
from data import DataLoader
from trainer import trainer, make_UIdataset
from utils import wandb_setup, seed_everything

def main(args):
    seed_everything(args.seed)

    weight_dir = os.path.join(args.save, 'model.pt')
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.save = weight_dir
    
    # wandb 설정
    wandb_setup(args)
    
    ###############################################################################
    # Load data
    ###############################################################################
    loader = DataLoader(args.data, args.seed, args.dataset_create)
    n_items = loader.load_n_items()
    
    args.data = dict()
    args.data['train'] = loader.load_data('train')
    args.data['valid'] = loader.load_data('validation')
    args.data['genre'] = loader.load_data('genre')
    ###############################################################################
    # Build the model
    ###############################################################################
    model = NeuMF(args).to(args.device)
    criterion = loss_function_BCEW
    ###############################################################################
    # Training code
    ###############################################################################
    UIdataset = make_UIdataset(args.data['train'], args.data['genre'], neg_ratio=args.neg_ratio)
    runner = trainer(model, UIdataset, criterion, args) # criterion
    runner.run()
    

if __name__ == "__main__":
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Nueral-MF for Collaborative Filtering')


    parser.add_argument('--data', type=str, default='./../input/data/train/',
                        help='Movielens dataset location')
    parser.add_argument('--dataset_create', action='store_true',
                        help='create preprocessed data file')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='drop out ratio of model')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='./saved',
                        help='path to save the final model')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    ########################################################################################
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding ')
    parser.add_argument('--n_items', type=int, default=6807,
                        help='Items count')
    parser.add_argument('--layer_dim', type=int, default=128,
                        help='layer dimensions')
    parser.add_argument('--n_users', type=int, default=31361,
                        help='users count')
    parser.add_argument('--n_genres', type=int, default=18,
                        help='genres count')
    parser.add_argument('--neg_ratio', type=int, default=50,
                        help='negative sampling ratio')
    parser.add_argument('--check_epoch', type=int, default=1,
                        help='Check Epoch')
    parser.add_argument('--top_k', type=int, default=10,
                        help='TOP K')
    ########################################################################################
    args = parser.parse_args()

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("using",args.device)
    main(args)
    