# Neural Collaborative Filtering With Side Information 

Pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)] 

---

## Model

<img width="728" alt="KakaoTalk_20201222_200737563" src="https://user-images.githubusercontent.com/69955858/107190147-cd937e00-6a2d-11eb-9dba-007667a3dfa2.png">

#### multimodal feature of Movielens
- Image → [512 dim vector]
- Text → [300 dim vector]

---


## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|path|str|path|/daintlab/data/recommend/Amazon-office-raw|
|top_k|int|top k recomendation|10|
|image|bool|image feature|False|
|text|bool|text feature|False|
|feature|str|raw(png) or pre(vector)|raw|
|optim|str|Optimizer|adam|
|lr|float|Learning Rate|0.001|
|epochs|int|Epoch|50|
|drop_rate|float|dropout rate|0.0|
|batch_size|int|Train batch size|1024|
|latent_dim_mf|int|Dimension of latent vectors|8|
|num_layers|int|Number of MLP's layer |1|
|num_neg|int|Number of negative sample|4|
|l2|float|L2 Regularization|0|
|gpu|int|Number of used gpus|1|
|eval|str|evaluation protocol|ratio-split|
|interval|int|evaluation interval|1|
|extractor_path|str|pretrained model path|/daintlab/data/recommend/Amazon-office-raw/resnet18.pth|
|amp|using Automatic mixed-precision|bool|True|
|port|master port|str|8888|
|address|master address|str|127.0.0.1|

## Installation 

### Clone the repo 

```sh
git clone https://github.com/dltkddn0525/recommendation.git
```

### Prerequisites 

- python 3.8.3

  - pytorch 1.60
  - numpy 1.18.5 
  - pandas 1.1.2

## Usage 

```sh
python3 main.py --path=/daintlab/data/recommmend/Amazon-office-raw --top_k=10 --image=False --text=False  --feature=raw --optim=adam --lr=0.001 --epochs=30 --drop_rate=0.3 --batch_size=256 --latent_dim_mf=8 --num_layers=1 --num_neg=4 --gpu=2 --eval=ratio-split --amp=True --port=8888 --address=127.0.0.1
```

## Performance metrics
- [x] **NDCG@10**
- [x] **HR@10**

---

## Result

