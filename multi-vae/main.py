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

def main(args):
    # TODO VAE와 DAE의 저장 파일 이름 다르게 만들기
    weight_dir = os.path.join(args.save, 'model.pt')

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.save = weight_dir
    ###############################################################################
    # Load data
    ###############################################################################
    loader = DataLoader(args.data, args.seed, args.dataset_create)
    n_items = loader.load_n_items()

    args.data = dict()
    args.data['train'] = loader.load_data('train')
    args.data['valid'] = loader.load_data('validation')
    args.data['test'] = loader.load_data('test')
    # train_data = loader.load_data('train')
    # vad_data_tr, vad_data_te = loader.load_data('validation')
    # test_data_tr, test_data_te = loader.load_data('test')

    ###############################################################################
    # Build the model
    ###############################################################################

    p_dims = [200, 600, n_items]
    if args.is_VAE:
        model = MultiVAE(p_dims).to(device)
        criterion = loss_function_vae
    else:
        model = MultiDAE(p_dims).to(device)
        criterion = loss_function_dae

    ###############################################################################
    # Training code
    ###############################################################################
    runner = trainer(model, criterion, args)
    runner.run()


if __name__ == "__main__":
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


    parser.add_argument('--data', type=str, default='../../data/train/',
                        help='Movielens dataset location')
    parser.add_argument('--dataset_create', type=bool, default=False,
                        help='create preprocessed data file')
    parser.add_argument('--is_VAE', type=bool, default=True,
                        help='if True use VAE else DAE')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
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

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    print("using",device)
    main(args)