"""
1. Rating df 생성  
rating 데이터(train_ratings.csv)를 불러와 [user, item, rating]의 컬럼으로 구성된 데이터 프레임을 생성합니다.   

2. Genre df 생성   
genre 정보가 담긴 데이터(genres.tsv)를 불러와 genre이름을 id로 변경하고, [item, genre]의 컬럼으로 구성된 데이터 프레임을 생성합니다.    

3. Negative instances 생성   
rating 데이터는 implicit feedback data(rating :0/1)로, positive instances로 구성되어 있습니다. 따라서 rating이 없는 item중 negative instances를 뽑아서 데이터에 추가하게 됩니다.   

4. Join dfs   
rating df와 genre df를 join하여 [user, item, rating, genre]의 컬럼으로 구성된 데이터 프레임을 생성합니다.   

5. zero-based index로 mapping   
Embedding을 위해서 user,item,genre를 zero-based index로 mapping합니다.
    - user : 0-31359
    - item : 0-6806
    - genre : 0-17  

6. feature matrix X, label tensor y 생성   
[user, item, genre] 3개의 field로 구성된 feature matrix를 생성합니다.   

7. data loader 생성
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# 1. Rating df 생성
rating_data = "../../data/train/train_ratings.csv"

raw_rating_df = pd.read_csv(rating_data)
raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

users = set(raw_rating_df.loc[:, 'user'])
items = set(raw_rating_df.loc[:, 'item'])

#2. Genre df 생성
genre_data = "../../data/train/genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경

# 3. Negative instance 생성
num_negative = 50
user_group_dfs = list(raw_rating_df.groupby('user')['item'])
first_row = True
user_neg_dfs = pd.DataFrame()

for u, u_items in user_group_dfs:
    u_items = set(u_items)
    i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
    
    i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
    if first_row == True:
        user_neg_dfs = i_user_neg_df
        first_row = False
    else:
        user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)

# 4. Join dfs
joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')

# 5. user, item을 zero-based index로 mapping
users = list(set(joined_rating_df.loc[:,'user']))
users.sort()
items =  list(set((joined_rating_df.loc[:, 'item'])))
items.sort()
genres =  list(set((joined_rating_df.loc[:, 'genre'])))
genres.sort()

if len(users)-1 != max(users):
    users_dict = {users[i]: i for i in range(len(users))}
    joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
    users = list(set(joined_rating_df.loc[:,'user']))
    
if len(items)-1 != max(items):
    items_dict = {items[i]: i for i in range(len(items))}
    joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
    items =  list(set((joined_rating_df.loc[:, 'item'])))

joined_rating_df = joined_rating_df.sort_values(by=['user'])
joined_rating_df.reset_index(drop=True, inplace=True)

data = joined_rating_df

n_data = len(data)
n_user = len(users)
n_item = len(items)
n_genre = len(genres)

#6. feature matrix X, label tensor y 생성
user_col = torch.tensor(data.loc[:,'user'])
item_col = torch.tensor(data.loc[:,'item'])
genre_col = torch.tensor(data.loc[:,'genre'])

offsets = [0, n_user, n_user+n_item]
for col, offset in zip([user_col, item_col, genre_col], offsets):
    col += offset

X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
y = torch.tensor(list(data.loc[:,'rating']))


#7. data loader 생성
class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)

def get_data(train_ratio = 0.9):
    dataset = RatingDataset(X, y)
    train_ratio = 0.9

    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader, test_dataset

def get_dict_and_df():
    return items_dict, users_dict, user_group_dfs, raw_genre_df, offsets

def get_nums():
    return n_user, n_item, n_genre