import time

import numpy as np
import torch
from torch import optim
from scipy import sparse

from metric import NDCG_binary_at_k_batch, Recall_at_k_batch
from utils import wandb_upload

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

class trainer():
    def __init__(self, model, criterion=None, args=None):
        self.model = model
        self.criterion = criterion
        self.args = args
        self.is_VAE = args.is_VAE
        self.N = None
        self.idxlist = None
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=self.args.wd)
        self.update_count = 0
        
    def train(self, epoch, train_data):
        # Turn on training mode
        self.model.train()
        train_loss = 0.0
        start_time = time.time()

        np.random.shuffle(self.idxlist)
        
        for batch_idx, start_idx in enumerate(range(0, self.N, self.args.batch_size)):
            end_idx = min(start_idx + self.args.batch_size, self.N)
            data = train_data[self.idxlist[start_idx:end_idx]]
            data = naive_sparse2tensor(data).to(self.args.device)
            self.optimizer.zero_grad()

            if self.is_VAE:
                if self.args.total_anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 
                                    1. * self.update_count / self.args.total_anneal_steps)
                else:
                    anneal = self.args.anneal_cap

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                
                loss = self.criterion(recon_batch, data, mu, logvar, anneal)
            else:
                recon_batch = self.model(data)
                loss = self.criterion(recon_batch, data)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.update_count += 1

            if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, len(range(0, self.N, self.args.batch_size)),
                            elapsed * 1000 / self.args.log_interval,
                            train_loss / self.args.log_interval))
                

                start_time = time.time()
                train_loss = 0.0


    def evaluate(self, data_tr, data_te):
        # Turn on evaluation mode
        self.model.eval()
        total_loss = 0.0
        e_idxlist = list(range(data_tr.shape[0]))
        e_N = data_tr.shape[0]
        n100_list = []
        r10_list = []
        r20_list = []
        r50_list = []
        
        with torch.no_grad():
            for start_idx in range(0, e_N, self.args.batch_size):
                end_idx = min(start_idx + self.args.batch_size, self.N)
                data = data_tr[e_idxlist[start_idx:end_idx]]
                heldout_data = data_te[e_idxlist[start_idx:end_idx]]

                data_tensor = naive_sparse2tensor(data).to(self.args.device)
                if self.is_VAE :
                
                    if self.args.total_anneal_steps > 0:
                        anneal = min(self.args.anneal_cap, 
                                    1. * self.update_count / self.args.total_anneal_steps)
                    else:
                        anneal = self.args.anneal_cap

                    recon_batch, mu, logvar = self.model(data_tensor)

                    loss = self.criterion(recon_batch, data_tensor, mu, logvar, anneal)

                else :
                    recon_batch = self.model(data_tensor)
                    loss = self.criterion(recon_batch, data_tensor)




                total_loss += loss.item()

                # Exclude examples from training set
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.nonzero()] = -np.inf

                n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
                r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
                r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

                n100_list.append(n100)
                r10_list.append(r10)
                r20_list.append(r20)
                r50_list.append(r50)
    
        total_loss /= len(range(0, e_N, self.args.batch_size))
        n100_list = np.concatenate(n100_list)
        r10_list = np.concatenate(r10_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

        return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)


    def run(self):
        self.N = self.args.data['train'].shape[0]
        self.idxlist = list(range(self.N))

        best_r10 = -np.inf
        self.update_count = 0
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch, self.args.data['train'])
            val_loss, n100, r10, r20, r50 = self.evaluate(*self.args.data['valid'])
            print('-' * 102)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                        epoch, time.time() - epoch_start_time, val_loss,
                        n100, r10, r20, r50))
            print('-' * 102)

            n_iter = epoch * len(range(0, self.N, self.args.batch_size))


            # Save the self.model if the r10 is the best we've seen so far.
            if r10 > best_r10:
                with open(self.args.save, 'wb') as f:
                    torch.save(self.model, f)
                best_r10 = r10
            wandb_upload(dataset='valid',epoch=epoch, valid_loss=val_loss, best_recall10=best_r10,
                        ndcg100=n100, recall10=r10)


        # Load the best saved self.model.
        with open(self.args.save, 'rb') as f:
            self.model = torch.load(f)

        # Run on test data.
        test_loss, n100, r10, r20, r50 = self.evaluate(*self.args.data['test'])
        print('=' * 102)
        print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
                'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
        print('=' * 102)
        wandb_upload(dataset='test', test_loss=test_loss, test_ndcg100=n100, test_recall10=r10)

    def inference(self, inference_sdata):
        self.model.eval()
        recon_list = []
        self.update_count = 0
        e_idxlist = list(range(inference_sdata.shape[0]))
        e_N = inference_sdata.shape[0]
        with torch.no_grad():
            for start_idx in range(0, e_N, self.args.batch_size):
                end_idx = min(start_idx + self.args.batch_size, e_N)
                # print(inference_sdata[0].toarray())
                # print(e_idxlist[start_idx:end_idx])
                data = inference_sdata[e_idxlist[start_idx:end_idx]]
                data_tensor = naive_sparse2tensor(data).to(self.args.device)
                # VAE
                if self.args.total_anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 
                        1. * self.update_count / self.args.total_anneal_steps)
                else:
                    anneal = self.args.anneal_cap
                
                if self.is_VAE:
                    recon_batch, mu, logvar = self.model(data_tensor)
                else:
                    recon_batch = self.model(data_tensor)

                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.nonzero()] = -np.inf
                recon_list.append(recon_batch)
        recon_list = np.concatenate(recon_list)

        return recon_list