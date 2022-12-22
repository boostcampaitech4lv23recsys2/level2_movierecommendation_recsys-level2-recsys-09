import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from model import DeepFM
from util import Recall_at_k_batch
from data_loader import get_data, get_nums

from pathlib import Path
from collections import OrderedDict
import json


def main(config):
    device = torch.device(config["device"])

    train_loader, test_loader, test_dataset = get_data(config["train_ratio"])
    n_user, n_item, n_genre = get_nums()
    input_dims = [n_user, n_item, n_genre]
    
    model = DeepFM(input_dims, config["embedding_dim"], mlp_dims=config["mlp_dims"]).to(device)
    bce_loss = nn.BCELoss() # Binary Cross Entropy loss
    num_epochs = config["num_epochs"]
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    print("Training...")
    for e in tqdm(range(num_epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = bce_loss(output, y.float())
            loss.backward()
            optimizer.step()

    correct_result_sum = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        output = model(x)
        result = torch.round(output)
        correct_result_sum += (result == y).sum().float()

    acc = correct_result_sum/len(test_dataset)*100
    print("Final Acc : {:.2f}%".format(acc.item()))

    #recall = Recall_at_k_batch(correct_result_sum, test_dataset, 10)
    #print("Recall at 10 : {:.2f}%".format(recall.item()))

    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])
    torch.save(model.state_dict(), config["model_path"] + config["model_filename"])


if __name__ == '__main__':
    fname = Path("config.json")
    with fname.open('rt') as handle:
        config = json.load(handle, object_hook=OrderedDict)
    main(config)