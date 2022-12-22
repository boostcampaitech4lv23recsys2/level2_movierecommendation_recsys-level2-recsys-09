import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

from model import DeepFM
from data_loader import get_dict_and_df, get_nums

from pathlib import Path
from collections import OrderedDict
import json


def main(config):
    device = torch.device(config["device"])

    items_dict, users_dict, user_group_dfs, raw_genre_df, offsets = get_dict_and_df()
    n_user, n_item, n_genre = get_nums()
    input_dims = [n_user, n_item, n_genre]

    model = DeepFM(input_dims, config["embedding_dim"], mlp_dims=config["mlp_dims"]).to(device)
    model.load_state_dict(torch.load(config["model_path"] + config["model_filename"]))
    model.eval()

    # 모든 유저-아이템을 인풋으로 넣어서 결과 생성 후 랭킹 (31360 x 6807)
    u_list = []
    i_list = []
    ritems_dict = {v:k for k,v in items_dict.items()}

    print("Inferencing...")
    for u, u_items in tqdm(user_group_dfs):

        # 인코딩하기 전에 유저id 저장
        u_list.append([u]*10)

        # user_group_dfs은 인코딩 이전 값이므로 사용하기 위해 인코딩 진행
        u = users_dict[u]
        u_items = set(u_items.map(lambda x : items_dict[x]))    # 인덱스로 활용 가능!

        # user, item, genre 데이터를 인코딩하여 학습한 모델에 맞는 값으로 변환
        i_user_col = torch.tensor([u] * n_item)
        i_item_col = torch.tensor(raw_genre_df['item'].map(lambda x : items_dict[x]).values)
        i_genre_col = torch.tensor(raw_genre_df['genre'].values)
        for col, offset in zip([i_user_col, i_item_col, i_genre_col], offsets):
            col += offset
        
        x = torch.cat([i_user_col.unsqueeze(1), i_item_col.unsqueeze(1), i_genre_col.unsqueeze(1)], dim=1)
        x = x.to(device)

        output_batch = model(x)
        output_batch = output_batch.cpu().detach().numpy()

        output_batch[list(u_items)] = -1    # -np.inf, 분포 확인을 위해 교체 / 이미 본 아이템 제외
        result_batch = np.argsort(output_batch)[-10:][::-1] # 역방향 -> 정방향으로 수정
        i_list.append(list(map(lambda x : ritems_dict[x], result_batch)))  # 아이템 디코딩, ndarray는 map()이 안돼서 다른 방법 찾음

    u_list = np.concatenate(u_list)
    i_list = np.concatenate(i_list)

    submit_df = pd.DataFrame(data={'user': u_list, 'item': i_list}, columns=['user', 'item'])

    if not os.path.exists(config["ouput_path"]):
        os.makedirs(config["ouput_path"])
    submit_df.to_csv(config["ouput_path"] + config["submit_filename"], index=False)


if __name__ == '__main__':
    fname = Path("config.json")
    with fname.open('rt') as handle:
        config = json.load(handle, object_hook=OrderedDict)
    main(config)