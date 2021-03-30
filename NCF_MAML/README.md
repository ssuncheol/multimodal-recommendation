## Arguments 

| Argument | Type | Description | Default |
|:---:|:---:|:---:|:---:|
|model|str|model type to use. [MAML, NCF]|MAML|
|save_path|str|save path|./result|
|batch_size|int|Total batch size. Each GPU will have batch_size/world_size|840|
|epochs|int|train epoch|50|
|data_path|str|path of rating data|/daintlab/data/recommend/Amazon-office-raw|
|num_layers|int|number of MLP's layer in NCF|4|
|embed_dim|int|embedding dimension|64|
|dropout_rate|float|droupout rate|0.2|
|lr|float|Learning Rate|0.001|
|margin|float|margin of embedding loss in MAML|1.0|
|feat_weight|float|weight of feature loss in MAML|1.0|
|cov_weight|float|weight of covariance loss in MAML|1.0|
|top_k|int|top k recomendation|10|
|num_neg|int|number of negative sample per positive sample for training|4|
|load_path|str|path to model checkpoint|None|
|eval_freq|int|evaluate performance every n epoch|50|
|feature_type|str|type of feature to use. [all, img, txt, rating]|rating|
|eval_type|str|evaluation protocol. [ratio-split, leave-one-out]|ratio-split|
|cnn_path|str|path to imagenet pretrained ResNet18 model. if None, randomly initialized.|'./resnet18.pth'|
|ddp_port|str|ddp master port|22222|
|ddp_addr|str|ddp master address|127.0.0.1|
|fine_tuning|bool|Whether to apply fine tuning. If False, resnet18 will be freezed|False|
|hier_attention|bool|Whether to apply hierarchical attention|False|


- [Pytorch에서 제공하는 Imagenet pretrained ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)
<hr>

## References

### User Diverse Preferences Modeling By Multimodal Attentive Metric Learning
Pytorch implementation of [User Diverse Preferences Modeling By Multimodal Attentive Metric Learning(Liu et al., 2019)](https://dl.acm.org/doi/abs/10.1145/3343031.3350953)

[Official Code](https://github.com/liufancs/MAML#user-diverse-preferences-modeling-by-multimodal-attentive-metric-learning)

### Neural Collaborative Filtering With Side Information 
Pytorch implementation of [Neural Collaborative Filtering(He et al., 2017)](https://arxiv.org/abs/1708.05031)

[Official Code](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)

