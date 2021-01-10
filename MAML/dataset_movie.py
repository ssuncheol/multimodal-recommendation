import numpy as np
import pandas as pd
import os
import json
import time
from torch.utils.data import Dataset
import pickle

def load_data(path):

    data = pd.read_feather(os.path.join(path, 'movie_3953.ftr'))
    movie_data = pd.read_csv(os.path.join(path, 'movies.csv'))

    num_user = max(data["userid"]) + 1
    num_item = max(movie_data["movieid"]) + 1

    end = time.time()
    test_negative = []
    train_ng_pool = []
    total_item = np.arange(0, num_item)

    trn_positive_len= data['train_positive'].map(len)
    tst_positive_item = data[['userid', 'test_positive']].copy()
    trn_positive_item=[]

    for user in range(num_user-1):
        train_ng_pool.append(data['train_negative'][user].tolist())
        test_negative.append(data['test_negative'][user].tolist())

        for i in range(trn_positive_len[user]):
            trn_positive_item.append([data['userid'][user],data['train_positive'][user][i]])

    trn_positive_item=pd.DataFrame(trn_positive_item)
    trn_positive_item=trn_positive_item.rename(columns={0: 'userid', 1: 'movieid'})

    with open(os.path.join(path, 'image_feature_vec.pickle'), 'rb') as f:
        img_vec = pickle.load(f)
    with open(os.path.join(path, 'text_feature_vec.pickle'), 'rb') as f:
        text_vec = pickle.load(f)

    keys=list(text_vec.keys())

    t_features = []
    v_features = []
    for i in range(len(keys)):
        t_features.append(text_vec[keys[i]])
        v_features.append(img_vec[keys[i]])

    feature = np.concatenate((t_features, v_features), axis=1)
    feature_dict={}
    for i in range(len(keys)):
        feature_dict[keys[i]]=feature[i]

    return trn_positive_item, tst_positive_item, train_ng_pool, test_negative, num_user, num_item, feature_dict




class CustomDataset_movielens(Dataset):
    '''
    Train Batch [user, item_p, item_n, feature_p, feature_n]
    user = [N]
    item_p = [N]
    item_n = [N x num_neg]
    feature_p = [N x (vis_feature_dim + text_feature_dim)]
    featuer_n = [N x num_neg x (vis_feature_dim + text_feature_dim)]
    Test Batch [user, item, feature, label]
    N = number of positive + negative item for corresponding user
    user = [1]
    item = [N]
    feature = [N x (vis_feature_dim + text_feature_dim)]
    label = [N] 1 for positive, 0 for negative
    '''

    def __init__(self, dataset, feature, negative, num_neg=4, istrain=False):
        super(CustomDataset_movielens, self).__init__()
        self.dataset = dataset
        self.feature = feature
        self.negative = negative
        self.istrain = istrain
        self.num_neg = num_neg
        self.train_dataset = None

        if istrain:
            self.train_ng_sampling()
        else:
            self.make_testset()

    def train_ng_sampling(self):
        assert self.istrain
        end = time.time()
        print(f"Negative sampling for Train. {self.num_neg} Negative samples per positive pair")
        train_negative = []
        for index, row in self.dataset.iterrows():
            user = int(row["userid"])-1
            ng_pool = self.negative[user]
            ng_item_u = []
            # Sampling num_neg samples
            for i in range(self.num_neg):
                idx = np.random.randint(0, len(ng_pool))
                ng_item_u.append(ng_pool[idx])
            train_negative.append(ng_item_u)
        self.dataset["negative"] = train_negative
        self.train_dataset = self.dataset.values.tolist()
        print(f"Negative Sampling Complete ({time.time() - end:.4f} sec)")

    def make_testset(self):
        assert not self.istrain
        users = np.unique(self.dataset["userid"])
        test_dataset = []
        for user in users:
            test_negative = self.negative[user-1]
            test_positive = self.dataset[self.dataset["userid"] == user]["test_positive"].tolist()
            item = test_positive + test_negative
            label = np.zeros(len(item))
            label[:len(test_positive)] = 1
            label = label.tolist()
            test_user = np.ones(len(item)) * user
            test_dataset.append([test_user.tolist(), item, label])

        self.dataset = test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.istrain:
            user, item_p, item_n = self.train_dataset[index]
            feature_p = self.feature[item_p]
            feature_n=[]
            for i in range(len(item_n)):
                feature_n.append(self.feature[item_n[i]])
            feature_n=np.array(feature_n)
            return user, item_p, item_n, feature_p, feature_n

        else:
            user, item, label = self.dataset[index]
            feature=[]
            for i in range(len(item)):
                feature.append(self.feature[item[i]])
            feature=np.array(feature)
            return user, item, feature, label


def inspect(df, num_inter):
    user = np.unique(df["userid"])
    x_user = []
    x_num_rating = []
    for i in user:
        if len(df[df["userid"] == i]) < num_inter:
            x_user.append(i)
            x_num_rating.append(len(df[df["userid"] == i]))

    return x_user, x_num_rating