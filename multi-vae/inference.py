import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from scipy import sparse
from model import MultiDAE, MultiVAE, loss_function_dae, loss_function_vae
from data import DataLoader, load_inference_data, reverse_numerize
from trainer import trainer

def main(args):
    weight_dir = os.path.join(args.save, 'model.pt')
    
    assert os.path.exists(weight_dir)

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

    inference_sdata = load_inference_data(args.data, unique_uid, unique_sid)

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
    recon_list = runner.inference(inference_sdata)

    ###############################################################################
    # Save top 10 recommended data
    ###############################################################################
    pred = [[], []]
    for u, i in enumerate(recon_list):
        pred[0].append([u]*10)
        pred[1].append(np.argsort(i)[-10:])
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
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    parser.add_argument('--submit_path', type=str, default='./output')
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
    args = parser.parse_args([])

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    print("using",device)
    main(args)