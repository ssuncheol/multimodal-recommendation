# Neural Collaborative Filtering With Side Information 

Pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)] 

---

## Data preparation 

### Movielens dataset

Data should be prepared as follows

- movie_3953.ftr (Rating data)
- movies.csv (Information of movie : genre, director)
- image_feature_vec.pickle (Image features of movie posters extracted from pretrained network(ResNet18))
- text_feature_vec.pickle (Text features of movie's title + plot extracted from pretrained network(Doc2Vec model))


Movielens Dataset can be downloaded here<br>
[Movielens dataset](https://drive.google.com/drive/folders/15T7s2DDFt1HLlwRVw4ytViKE2rAAXgsj)


## Model

![image](https://user-images.githubusercontent.com/69955858/102883105-cb338f80-4492-11eb-8391-7c9d7da6f32a.png)

- Genre → [23 dim one-hot-vector] 
- Director → [1918 dim one-hot-vector]
- Image -> [512 dim vector]
- Text -> [300 dim vector]

---


## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|optim|str|Optimizer|'adam'|
|lr|float|Learning Rate|0.001|
|epochs|int|Epoch|20|
|batch_size|int|Train batch size|1024|
|latent_dim_mf|int|Dimension of latent vectors|8|
|num_layers|int|Number of MLP's layer |1|
|num_neg|int|Number of negative sample|4|
|l2|float|L2 Regularization|0|
|gpu|str|Name of Using gpu|0|

## Usage 

```sh
python3 main.py --optim='adam' --lr=0.001 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=1 --num_neg=4 --gpu=0
```

## Performance metrics
- [x] **NDCG@10**
- [x] **HR@10**

---

## Result

