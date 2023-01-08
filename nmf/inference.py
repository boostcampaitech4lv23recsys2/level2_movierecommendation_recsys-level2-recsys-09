import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
import pandas as pd

from model import NeuMF, loss_function_BCEW
from data import DataLoader, load_inference_data, reverse_numerize
from trainer import trainer, make_UIdataset
from utils import wandb_setup, seed_everything

def main(args):
    seed_everything(args.seed)

    # Load Best Model Weight File 
    weight_dir = os.path.join(args.save, 'model.pt')
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.save = weight_dir
    
    ###############################################################################
    # Load data
    ###############################################################################
    unique_uid, unique_sid = list(), list()
    pro_dir = os.path.join(args.data,'pro_sg')
    # get user id label
    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(int(line.strip()))
    # get item id label
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(int(line.strip()))
    
    loader = DataLoader(args.data)
    n_items = loader.load_n_items()
    
    inference_data = load_inference_data(args.data,  unique_uid, unique_sid)
    args.data = dict()
    args.data['genre'] = loader.load_data('genre')
    ###############################################################################
    # Build the model
    ###############################################################################
    print("[Model] using Nue-MF")
    with open(weight_dir, 'rb') as model_state:
        model = torch.load(model_state,map_location=args.device)
    criterion = loss_function_BCEW
    
    ###############################################################################
    # Inference code
    ###############################################################################
    UI = 0
    runner = trainer(model, UI, criterion, args) 
    recon_list = runner.inference(inference_data, args.data['genre'])
    
    ###############################################################################
    # Save top 10 recommended data
    ###############################################################################
    pred = [[], []]
    for u, i in enumerate(recon_list):
        pred[0].append([u]*10)
        pred[1].append(np.argsort(i)[-10:][::-1])
        
    pred[0] = np.concatenate(pred[0])
    pred[1] = np.concatenate(pred[1])
    
    submit = pd.DataFrame(data={'ui': pred[0], 'ii': pred[1]}, columns=['ui', 'ii'])
    rshow2id = dict((i, sid) for (i, sid) in enumerate(unique_sid))
    rprofile2id = dict((i, pid) for (i, pid) in enumerate(unique_uid))
    submit_df = reverse_numerize(submit, rshow2id, rprofile2id)
    submit_df.sort_values(by='user',inplace=True)

    if not os.path.exists(args.submit_path):
        os.makedirs(args.submit_path)
    submit_df.to_csv(os.path.join(args.submit_path,"submission.csv"), index=False)

if __name__ == "__main__":
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Nueral-MF for Collaborative Filtering')

    parser.add_argument('--submit_path', type=str, default='./output')
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
    ########################################################################################
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='embedding ')
    parser.add_argument('--n_items', type=int, default=6807,
                        help='Items count')
    parser.add_argument('--layer_dim', type=int, default=256,
                        help='layer dimensions')
    parser.add_argument('--n_users', type=int, default=31361,
                        help='users count')
    parser.add_argument('--n_genres', type=int, default=18,
                        help='genres count')
    parser.add_argument('--neg_ratio', type=int, default=100,
                        help='negative sampling ratio')
    parser.add_argument('--check_epoch', type=int, default=1,
                        help='Check Epoch')
    parser.add_argument('--top_k', type=int, default=1,
                        help='TOP K')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    ########################################################################################
    args = parser.parse_args()

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("using",args.device)
    main(args)