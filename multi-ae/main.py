import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse

from model import MultiDAE, MultiVAE, loss_function_dae, loss_function_vae
from data import DataLoader
from trainer import trainer
from utils import wandb_setup, seed_everything

def main(args):
    seed_everything(args.seed)

    # TODO VAE와 DAE의 저장 파일 이름 다르게 만들기
    weight_dir = os.path.join(args.save, 'model.pt')

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.save = weight_dir

    # wandb 기록 용 layer dimension 저장
    args.layer_dims = str([args.encode_dim,args.bn_dim,args.decode_dim])

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
    args.data['test'] = loader.load_data('test')

    ###############################################################################
    # Build the model
    ###############################################################################
    # best dims
    p_dims = [args.bn_dim, args.decode_dim, n_items]
    q_dims = [n_items, args.encode_dim, args.bn_dim]
    if args.is_VAE:
        print('[Model] using Mult-VAE')
        model = MultiVAE(p_dims, q_dims, args.dropout).to(args.device)
        criterion = loss_function_vae
    else:
        print('[Model] using Mult-DAE')
        model = MultiDAE(p_dims, q_dims, args.dropout).to(args.device)
        criterion = loss_function_dae

    ###############################################################################
    # Training code
    ###############################################################################

    runner = trainer(model, criterion, args)
    runner.run()


if __name__ == "__main__":
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Mult-Autoencoders for Collaborative Filtering')


    parser.add_argument('--data', type=str, default='../../data/train/',
                        help='Movielens dataset location')
    parser.add_argument('--dataset_create', action='store_true',
                        help='create preprocessed data file')
    parser.add_argument('--is_VAE', action='store_true',
                        help='to use VAE')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='drop out ratio of model')
    parser.add_argument('--encode_dim', type=int, default=500,
                        help='dimention of encode layer')
    parser.add_argument('--bn_dim', type=int, default=200,
                        help='dimention of bottle-neck layer')
    parser.add_argument('--decode_dim', type=int, default=600,
                        help='dimention of decode layer')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='./weight',
                        help='path to save the final model')
    args = parser.parse_args()

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("using",args.device)
    main(args)