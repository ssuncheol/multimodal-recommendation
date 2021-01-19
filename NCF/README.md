# Neural Collaborative Filtering With Side Information 

Pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)] 

---

## Model

<img width="728" alt="KakaoTalk_20201222_200737563" src="https://user-images.githubusercontent.com/69955858/102883105-cb338f80-4492-11eb-8391-7c9d7da6f32a.png">

#### multimodal feature of Movielens
- Image → [512 dim vector]
- Text → [300 dim vector]

---


## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|data|str|dataset|amazon|
|image|bool|image feature|False|
|text|bool|text feature|False|
|optim|str|Optimizer|adam|
|lr|float|Learning Rate|0.001|
|epochs|int|Epoch|20|
|batch_size|int|Train batch size|1024|
|latent_dim_mf|int|Dimension of latent vectors|8|
|num_layers|int|Number of MLP's layer |1|
|num_neg|int|Number of negative sample|4|
|l2|float|L2 Regularization|0|
|gpu|str|Name of Using gpu|0|
|eval|str|evaluation protocol|ratio-split|


## Installation 

### Clone the repo 

```sh
git clone https://github.com/dltkddn0525/recommendation.git
```

### Prerequisites 

- python 3.8.3

  - pytorch 1.60
  - sklearn 0.23.2
  - numpy 1.18.5 
  - pandas 1.1.2

## Usage 

```sh
python3 main.py --optim='adam' --lr=0.001 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=1 --num_neg=4 --gpu=0
```

## Performance metrics
- [x] **NDCG@10**
- [x] **HR@10**

---

## Result

