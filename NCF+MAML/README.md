# User Diverse Preferences Modeling By Multimodal Attentive Metric Learning
Pytorch implementation of [User Diverse Preferences Modeling By Multimodal Attentive Metric Learning(Liu et al., 2019)](https://dl.acm.org/doi/abs/10.1145/3343031.3350953)
[Official Code](https://github.com/liufancs/MAML#user-diverse-preferences-modeling-by-multimodal-attentive-metric-learning)

# Neural Collaborative Filtering With Side Information 
Pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars]
[Official Code](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)] 

---

#### multimodal feature of datasets
- Image → Resnet18 → [512 dim vector]
- Text → Doc2Vec → [300 dim vector]

---


## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|model|str|model type[MAML, NCF]|MAML|
|save_path|str|save path|./result|
|batch_size|int|batch size|512|
|epochs|int|train epoch|50|
|data_path|str|path of rating data|/daintlab/data/recommend/Amazon-office-raw|
|num_layers|int|number of MLP's layer|5|
|embed_dim|int|embedding dimension|64|
|MLP_dim|str|MLP's dimension|96,128,64,64,32,32|
|dropout_rate|float|droupout rate|0.2|
|lr|float|Learning Rate|0.001|
|margin|float|margin of embedding loss|
|feat_weight|float|weight of feature loss|0|
|cov_weight|float|weight of covariance loss|0|
|top_k|int|top k recomendation|10|
|num_neg|int|number of negative sample for training|4|
|load_path|str|path of saved model|None|
|eval_freq|int|evaluate performance every n epoch|50|
|feature_type|str|type of feature to use. [all, img, txt, rating]|rating|
|eval_type|str|evaluation protocol. [ratio-split, leave-one-out]|ratio-split|
|cnn_path|str|path of pretrained model|./resnet18.pth|
|ddp_port|str|ddp master port|str|22222|
|ddp_addr|str|ddp master address|str|127.0.0.1|
|fine_tuning|bool|fine tuning|False|
|hier_attention|bool|hierarchical attention|False|
|att_wd|float|-|100|

## Installation 

#### MAML
```sh
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py --model=MAML --save_path=./amazon_attention_ratio_img_wd_01_feat1_cov1 --batch_size=512 --epoch=50 --feature_type=img --eval_type=ratio-split --att_wd=0.1 --feat_weight=1 --cov_weight=1 --hier_attention=True
```
#### NCF
```sh
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --model=NCF --save_path=./final_fine/ratio/img/4_16/10  --batch_size=512 --epoch=30 --feature_type=img --num_layers=4 --embed_dim=16 --eval_type=ratio-split --dropout_rate=0.3 --eval_freq=15 --MLP_dim=48,48,24,24,16
```

The following results will be saved in ```<Your save path>```
- train.log ( epoch, total loss, embedding loss, feature loss, covariance loss )
- test.log ( epoch, hit ratio, nDCG )
- model.pth (model saved every n epoch)



## Performance metrics
- [x] **HR@10**
- [x] **NDCG@10**
---

## Result


