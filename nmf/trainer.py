import pandas as pd
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm
import os
from collections import defaultdict

from metric import recallk, ndcgk
from utils import wandb_upload


def make_UIdataset(train,genre, neg_ratio):
    """ 유저별 학습에 필요한 딕셔너리 데이터 생성 
    Args:
        train : 유저-아이템의 상호작용을 담은 행렬 
            ex) 
                array([[0., 0., 0., ..., 0., 0., 0.],
                        [0., 0., 0., ..., 0., 0., 0.],
                        [0., 0., 0., ..., 0., 0., 0.],
                        ...,
                        [0., 0., 0., ..., 0., 0., 0.],
                        [0., 0., 0., ..., 0., 0., 0.],
                        [0., 0., 0., ..., 0., 0., 0.]])
        neg_ratio : negative sampling 활용할 비율 
            ex) 3 (positive label 1개당 negative label 3개)
    Returns: 
        UIdataset : 유저별 학습에 필요한 정보를 담은 딕셔너리 
            ex) {'사용자 ID': [[positive 샘플, negative 샘플], ... , [1, 1, 1, ..., 0, 0]]}
                >>> UIdataset[3]
                    [array([   16,    17,    18, ...,  9586, 18991,  9442]),
                    array([5, 5, 5, ..., 5, 5, 5]),
                    array([4, 4, 4, ..., 5, 1, 1]),
                    array([1., 1., 1., ..., 0., 0., 0.])]
    """
    UIdataset = {}
    print("Make UIdataset ...ing..")
    for user_id, items_by_user in enumerate(train):
        UIdataset[user_id] = []
        # positive 샘플 계산 
        pos_item_ids = np.where(items_by_user > 0.5)[0]
        num_pos_samples = len(pos_item_ids)

        # negative 샘플 계산 (random negative sampling) 
        num_neg_samples = neg_ratio * num_pos_samples
        neg_items = np.where(items_by_user < 0.5)[0]
        neg_item_ids = np.random.choice(neg_items, min(num_neg_samples, len(neg_items)), replace=False)
        UIdataset[user_id].append(np.concatenate([pos_item_ids, neg_item_ids]))
        
        # feature 추출 
        # features = []
        # for item_id in np.concatenate([pos_item_ids, neg_item_ids]): 
        #     features.append(user_features['age'][user_id])
        # UIdataset[user_id].append(np.array(features))
        
        features = []
        for item_id in np.concatenate([pos_item_ids, neg_item_ids]): 
            features.append(genre['genre'][item_id])
        UIdataset[user_id].append(np.array(features))
        
        # label 저장  
        pos_labels = np.ones(len(pos_item_ids))
        neg_labels = np.zeros(len(neg_item_ids))
        UIdataset[user_id].append(np.concatenate([pos_labels, neg_labels]))
    print("UIdataset Finished !")
    return UIdataset

def make_batchdata(UIdataset, user_indices, batch_idx, batch_size):
    """ 배치 데이터로 변환 
    Args:
        user_indices : 전체 유저의 인덱스 정보 
            ex) array([ 3100,  1800, 30098, ...,  2177, 11749, 20962])
        batch_idx : 배치 인덱스 (몇번째 배치인지)
            ex) 0 
        batch_size : 배치 크기 
            ex) 256 
    Returns 
        batch_user_ids : 배치내의 유저 인덱스 정보 
            ex) [22194, 22194, 22194, 22194, 22194, ...]
        batch_item_ids : 배치내의 아이템 인덱스 정보 
            ex) [36, 407, 612, 801, 1404, ...]
        batch_feat0 : 배치내의 유저-아이템 인덱스 정보에 해당하는 feature0 정보 
            ex) [6, 6, 6, 6, 6, ...]
        batch_feat1 : 배치내의 유저-아이템 인덱스 정보에 해당하는 feature1 정보 
            ex) [4,  4,  4, 23,  4, ...]
        batch_labels : 배치내의 유저-아이템 인덱스 정보에 해당하는 label 정보 
            ex) [1.0, 1.0, 1.0, 1.0, 1.0, ...]
    """
    batch_user_indices = user_indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
    batch_user_ids = []
    batch_item_ids = []
    batch_feat0 = []
    # batch_feat1 = []
    batch_labels = []
    for user_id in batch_user_indices:
        item_ids = UIdataset[user_id][0]
        feat0 = UIdataset[user_id][1]
        # feat1 = UIdataset[user_id][2]
        labels = UIdataset[user_id][2]
        user_ids = np.full(len(item_ids), user_id)
        batch_user_ids.extend(user_ids.tolist())
        batch_item_ids.extend(item_ids.tolist())
        batch_feat0.extend(feat0.tolist())
        # batch_feat1.extend(feat1.tolist())
        batch_labels.extend(labels.tolist())
    return batch_user_ids, batch_item_ids, batch_feat0,  batch_labels # batch_feat1,

def update_avg(curr_avg, val, idx):
    """ 현재 epoch 까지의 평균 값을 계산 
    """
    return (curr_avg * idx + val) / (idx + 1)

class trainer():
    def __init__(self, model, UI, criterion=None, args=None):
        self.model = model
        self.criterion = criterion
        self.args = args
        self.idxlist = None
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr,  weight_decay=self.args.wd)
        self.update_count = 0
        self.UIdataset = UI
        
    def train(self, epoch, train_data, item_features):
        self.model.train()
        curr_loss_avg = 0.0
        
        n_users = train_data.shape[0]
        user_indices = np.arange(n_users)
        np.random.RandomState(self.args.epochs).shuffle(user_indices)
        batch_num = int(len(user_indices) / self.args.batch_size) + 1
        bar = range(batch_num)
        
        for step, batch_idx in enumerate(bar):
            user_ids, item_ids, feat0,  labels = make_batchdata(self.UIdataset, user_indices, batch_idx, self.args.batch_size) # feat1
            # 배치 사용자 단위로 학습
            user_ids = torch.LongTensor(user_ids).to(self.args.device)
            item_ids = torch.LongTensor(item_ids).to(self.args.device)
            feat0 = torch.LongTensor(feat0).to(self.args.device)
            # feat1 = torch.LongTensor(feat1).to(args.device)
            labels = torch.FloatTensor(labels).to(self.args.device)
            labels = labels.view(-1, 1)

            # grad 초기화
            self.optimizer.zero_grad()

            # 모델 forward
            output = self.model.forward(user_ids, item_ids, [feat0]) # , feat1
            output = output.view(-1, 1)

            loss = self.criterion(output, labels)

            # 역전파
            loss.backward()

            # 최적화
            self.optimizer.step()    
            if torch.isnan(loss):
                print('Loss NAN. Train finish.')
                break
            curr_loss_avg = update_avg(curr_loss_avg, loss, step)
            
            msg = f"epoch: {epoch}, "
            msg += f"loss: {curr_loss_avg.item():.5f}, "
            msg += f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
            
        rets = {'losses': np.around(curr_loss_avg.item(), 5)}
        return rets
    
    def valid_epoch(self, epoch, data, item_features, args = None):
        pred_list = []
        self.model.eval()
        
        query_user_ids = data['uid'].unique() # 추론할 모든 user array 집합
        full_item_ids = np.array([c for c in range(self.args.n_items)]) # 추론할 모든 item array 집합 
        full_item_ids_feat1 = [item_features['genre'][c] for c in full_item_ids]
        
        for user_id in query_user_ids:
            with torch.no_grad():
                user_ids = np.full(self.args.n_items, user_id)
                
                user_ids = torch.LongTensor(user_ids).to(self.args.device)
                item_ids = torch.LongTensor(full_item_ids).to(self.args.device)
                # feat0 = np.full(args.n_items, user_features['age'][user_id])
                # feat0 = torch.FloatTensor(feat0).to(args.device)
                feat1 = torch.LongTensor(full_item_ids_feat1).to(self.args.device)
                # print(feat1.shape)
                
                eval_output = self.model.forward(user_ids, item_ids, [feat1]).detach().cpu().numpy() # feat0,
                pred_u_score = eval_output.reshape(-1)   
            
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = full_item_ids[pred_u_idx]
            pred_list.append(list(pred_u[:self.args.top_k]))
            
        pred = pd.DataFrame()
        pred['uid'] = query_user_ids
        pred['predicted_list'] = pred_list
        
        # 모델 성능 확인 
        rets = self.evaluation(data, pred)
        return rets, pred

    def run(self, args = None):
        total_logs = defaultdict(list)
        best_scores  = 0
        for epoch in range(self.args.epochs+1):
            train_results = self.train(epoch, self.args.data['train'], self.args.data['genre'])
            
            # args.check_epoch 번의 epoch 마다 성능 확인 
            if epoch % self.args.check_epoch == 0: 
                valid_results, _ = self.valid_epoch(epoch, self.args.data['valid'], self.args.data['genre'])

                logs = {
                    'Train Loss': train_results['losses'],
                    f'Valid Recall@{self.args.top_k}': valid_results['recall'],
                    f'Valid NDCG@{self.args.top_k}': valid_results['ndcg'],
                    # 'Valid Coverage': valid_results['coverage'],
                    'Valid Score': valid_results['score'],
                    }

                # 검증 성능 확인 
                for key, value in logs.items():
                    total_logs[key].append(value)

                if epoch == 0:
                    print("Epoch", end=",")
                    print(",".join(logs.keys()))

                print(f"{epoch:02d}  ", end="")
                print("  ".join([f"{v:0.6f}" for v in logs.values()]))
                
                # 가장 성능이 좋은 가중치 파일을 저장 
                if best_scores <= valid_results['score']: 
                    best_scores = valid_results['score']
                    with open(self.args.save, 'wb') as f:
                        torch.save(self.model, f)
                    
    def evaluation(self, gt, pred):
        """ 
        label과 prediction 사이의 recall, coverage, competition metric 평가 함수 
        Args:
            gt : 데이터 프레임 형태의 정답 데이터 
            pred : 데이터 프레임 형태의 예측 데이터 
        Returns: 
            rets : recall, ndcg, coverage, competition metric 결과 
                ex) {'recall': 0.123024, 'ndcg': 056809, 'coverage': 0.017455, 'score': 0.106470}
        """    
        gt = gt.groupby('uid')['sid'].unique().to_frame().reset_index()
        gt.columns = ['uid', 'actual_list']

        evaluated_data = pd.merge(pred, gt, how = 'left', on = 'uid')

        evaluated_data['Recall@25'] = evaluated_data.apply(lambda x: recallk(x.actual_list, x.predicted_list), axis=1)
        evaluated_data['NDCG@25'] = evaluated_data.apply(lambda x: ndcgk(x.actual_list, x.predicted_list), axis=1)

        recall = evaluated_data['Recall@25'].mean()
        ndcg = evaluated_data['NDCG@25'] .mean()
        # coverage = (evaluated_data['predicted_list'].apply(lambda x: x[:cfg.top_k]).explode().nunique())/meta_df.index.nunique()

        score = 0.75*recall + 0.25*ndcg
        rets = {"recall" :recall, 
                "ndcg" :ndcg, 
                # "coverage" :coverage, 
                "score" :score}
        
        return rets
    
    def inference(self, inference_data, item_features):
        self.model.eval()
        recon_list = []
        query_user_ids = inference_data['uid'].unique() # 추론할 모든 user array 집합
        full_item_ids = np.array([c for c in range(self.args.n_items)]) # 추론할 모든 item array 집합 
        full_item_ids_feat1 = [item_features['genre'][c] for c in full_item_ids]
        
        for user_id in query_user_ids:
            with torch.no_grad():
                user_ids = np.full(self.args.n_items, user_id)
                
                user_ids = torch.LongTensor(user_ids).to(self.args.device)
                item_ids = torch.LongTensor(full_item_ids).to(self.args.device)
                # feat0 = np.full(args.n_items, user_features['age'][user_id])
                # feat0 = torch.FloatTensor(feat0).to(args.device)
                feat1 = torch.LongTensor(full_item_ids_feat1).to(self.args.device)
                # print(feat1.shape)
                
                eval_output = self.model.forward(user_ids, item_ids, [feat1]).detach().cpu().numpy() # feat0,
                pred_u_score = eval_output.reshape(-1)   
                # 시청했던 것들 없애줄 코드, user_id이며, 해당 시청 아이템들의 값이 index니까 그 값을 -np.inf 처리
                idx = inference_data.loc[inference_data['uid']==user_id, :]['sid']
                pred_u_score[idx] = -np.inf
                recon_list.append(pred_u_score)
        
        return recon_list