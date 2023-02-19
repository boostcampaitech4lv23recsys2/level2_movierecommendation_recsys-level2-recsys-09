![스크린샷 2023-01-06 오전 9 15 45](https://user-images.githubusercontent.com/94108712/210904175-1db22a0d-97be-438b-8af0-24214a5342af.png)

# 9️⃣ boostcamp AI Tech 4th - RecSys

## 👪 Members
| [<img src="https://avatars.githubusercontent.com/u/94108712?v=4" width="200px">](https://github.com/KChanho) | [<img src="https://avatars.githubusercontent.com/u/22442453?v=4" width="200px">](https://github.com/sungsubae) | [<img src="https://avatars.githubusercontent.com/u/28619804?v=4" width="200px">](https://github.com/JJI-Hoon) | [<img src="https://avatars.githubusercontent.com/u/71113430?v=4" width="200px">](https://github.com/sobin98) | [<img src="https://avatars.githubusercontent.com/u/75313644?v=4" width="200px">](https://github.com/dnjstka0307) |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                          [김찬호](https://github.com/KChanho)                           |                            [배성수](https://github.com/sungsubae)                             |                        [이지훈](https://github.com/JJI-Hoon)                           |                          [정소빈](https://github.com/sobin98)                           |                            [조원삼](https://github.com/dnjstka0307) |
| 협업 관리, 인퍼런스 구현, EASE, 앙상블 | 모델 탐색, 데이터 전처리, 모델 베이스라인 개발 및 실험, 앙상블 | 모델 탐색 및 실험, Nue-MF Pytorch Project 개발 | 모델 탐색 및 실험 | EDA, DeepFM, Bert4rec |

<br /> 

## 🎬 Movie Recommendation
사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측

<br /> 


## 📄 Data
- 사용자의 영화 시청 이력 데이터 5,154,471 개
- 영화 아이템 메타 정보 데이터 6,807 개

<br /> 

## 💻 Repository Summary
![코드 구조도 drawio](https://user-images.githubusercontent.com/94108712/211151400-d7469957-c0db-48d2-8765-8ecc9e4c3270.png)

<br /> 

## 🗃 Project Process

### 🤖 Model
- AutoEncoder 계열: Multi-VAE, Multi-DAE, MSE-DAE, EASE, ALS
- 시퀀셜 모델: Sasrec, S3rec, Bert4rec

<br /> 

### 📈 Ensemble
- Top-K의 결과를 기반으로 Hard Voting 방법 사용
- 특성이 다른 시퀀셜과 AE 기반 모델을 사용하여 앙상블

<br /> 

## 🏅 Result : Public 6th > Privite 5th


|리더보드| Recall@10 | 순위 |
|:--------:|:------:|:----------:|
|public| 0.1646 | **6위** |
|private| 0.1634 | **최종 5위** |

![스크린샷 2023-01-08 오후 11 45 27](https://user-images.githubusercontent.com/94108712/217742589-5ae82226-622b-4f06-b7eb-04f68a35e03e.png)

<br /> 
