import os
import pandas as pd
from scipy import sparse
import numpy as np
import tqdm
from sklearn.preprocessing import LabelEncoder


class DataLoader():
    '''
    Load Movielens dataset
    '''
    def __init__(self, path, seed=42, dataset_create=False):
        if dataset_create:
            print("Preprocessing data and saving...")
            preprocess(path, seed)
        self.pro_dir = os.path.join(path, 'pro_sg')
        self.path = path
        assert os.path.exists(self.pro_dir), "Preprocessed files do not exist. Run data.py"
        
        self.n_items = self.load_n_items()
        
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_valid_data(datatype)
        elif datatype == 'genre':
            return self._load_genre_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        
        tp = pd.read_csv(path)
        n_users = tp['uid'].nunique()
        
        tp = tp.to_numpy()
        matrix = sparse.lil_matrix((n_users, self.n_items))
        for (p, i) in tp:
            matrix[p, i] = 1
        
        train_data = sparse.csr_matrix(matrix)
        train_data = train_data.toarray()
        print("train_data 형태: \n", train_data)
        
        return train_data
    
    def _load_valid_data(self, datatype='validation'):
        path = os.path.join(self.pro_dir, 'validation.csv'.format(datatype))

        valid = pd.read_csv(path)
        
        return valid
    
    def _load_genre_data(self, datatype='genre'):
        le = LabelEncoder()
        path = os.path.join(self.pro_dir, 'genres.tsv'.format(datatype))
        genre_data = pd.read_csv(path, sep='\t')
        
        genre_data['genre'] = le.fit_transform(genre_data['genre'])
        genre_data['item'] = le.fit_transform(genre_data['item'])
        
        # 아이템 특징 정보 추출 
        genre_data = genre_data.set_index('item')
        # 범주형 데이터를 수치형 데이터로 변경 
        genre_data['genre'] = le.fit_transform(genre_data['genre'])
        item_features = genre_data[['genre']].to_dict()
        print("item 749의 genre 정보 :", item_features['genre'][749])
        
        return item_features
    # unique_sid.txt : unique items 파일 변형에 의해서 변경
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id)# , as_index=False)
    count = playcount_groupbyid.size()

    return count

# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상) 
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def preprocess(path, seed=42):
    # Load Data
    raw_data = pd.read_csv(os.path.join(path, 'train_ratings.csv'), header=0)
    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

    # Shuffle User Indices
    unique_uid = user_activity.index
    print("(BEFORE) unique_uid:",unique_uid[:5])
    np.random.seed(seed)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    print("(AFTER) unique_uid:",unique_uid[:5])

    n_users = unique_uid.size #31360
    n_heldout_users = int(0.2 * n_users)


    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users)]
    vd_users = unique_uid[(n_users - n_heldout_users): ]

    #주의: 데이터의 수가 아닌 사용자의 수입니다!
    print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
    print("검증 데이터에 사용될 사용자 수:", len(vd_users))

    ##훈련 데이터에 해당하는 아이템들
    #Train에는 전체 데이터를 사용합니다.
    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

    ##아이템 ID
    unique_sid = pd.unique(train_plays['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(path, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)


    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    vad_data = numerize(vad_plays, profile2id, show2id)
    vad_data.to_csv(os.path.join(pro_dir, 'validation.csv'), index=False)

    print("Done!")
    

    
def load_inference_data(path,  unique_uid, unique_sid):
    raw_data = pd.read_csv(os.path.join(path, 'train_ratings.csv'), header=0)
    
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    inference_data = numerize(raw_data, profile2id, show2id)

    return inference_data

def reverse_numerize(tp, rshow2id, rprofile2id):
    user = tp['ui'].apply(lambda x: rprofile2id[x])
    item = tp['ii'].apply(lambda x: rshow2id[x])
    return pd.DataFrame(data={'user': user, 'item': item}, columns=['user', 'item'])