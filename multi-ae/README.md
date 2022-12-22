# Movie Recommendation using Mult-DAE & Mult-VAE

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/younggyoseo/vae-cf-pytorch

## Installation

```
pip install -r requirements.txt
```

## How to run

1. Training
   ```
   # preprocess dataset for first time
   python main.py --dataset_create
   python main.py
   ```
2. Inference
   ```
   python inference.py
   ```
