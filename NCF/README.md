# Neural Collaborative Filtering With Side Information 

Pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)] 

---

## Model

<img width="728" alt="KakaoTalk_20201222_200737563" src="https://user-images.githubusercontent.com/69955858/105034065-53aa4d80-5a9c-11eb-9ec1-f3a0f6bd4409.png">

#### multimodal feature of Movielens
- Image → [512 dim vector]
- Text → [300 dim vector]

---


## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|data|str|dataset|amazon|
|path|str|path|/daintlab/home/tmddnjs3467/workspace|
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
|interval|int|evaluation interval if eval is ratio-split|1|


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
python3 main.py --data=amazon --image=False --text=False --optim=adam --lr=0.001 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=1 --num_neg=4 --gpu=0 --eval=ratio-split
```

## Performance metrics
- [x] **NDCG@10**
- [x] **HR@10**

---

## Result

