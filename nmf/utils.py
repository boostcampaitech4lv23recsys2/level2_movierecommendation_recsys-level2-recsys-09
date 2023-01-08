import os
import random
import numpy as np
import torch
import wandb

def wandb_setup(args, exclude = ['data','dataset_create','cuda','log_interval']):
    wandb.init(project="NMF", entity="likesubscriberecommendai")

    exclude = set(exclude)
    config = dict()
    for key, value in vars(args).items():
        if key not in exclude:
            config[key] = value
    wandb.config.update(config)

def wandb_upload(**kwargs):
    log = dict()
    for key, item in kwargs.items():
        log[key] = item
    wandb.log(log)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True