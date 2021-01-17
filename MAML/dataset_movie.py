import numpy as np
import pandas as pd
import os
import json
import time
from torch.utils.data import Dataset
import pickle

def load_data(path, feature_type):

    data = pd.read_feather(os.path.join(path, 'movie_3953.ftr'))

    # +1 if starts with 0
    num_user = max(data["userid"])
    # num_user = max(data["userid"])+1

    test_negative = []
    train_ng_pool = []

    trn_positive_len = data['train_positive'].map(len)
    tst_positive_item = data[['userid', 'test_positive']].copy()
    trn_positive_item = []

    for user in range(num_user):
        train_ng_pool.append(data['train_negative'][user].tolist())
        test_negative.append(data['test_negative'][user].tolist())

        for i in range(trn_positive_len[user]):
            trn_positive_item.append([data['userid'][user],data['train_positive'][user][i]])

    trn_positive_item = pd.DataFrame(trn_positive_item)
    trn_positive_item = trn_positive_item.rename(columns={0: 'userID', 1: 'itemID'})
    tst_positive_item = tst_positive_item.rename(columns={'userid': 'userID', 'test_positive': 'itemID'})

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

    if feature_type == "all":
        feature = np.concatenate((t_features, v_features), axis=1)
    elif feature_type == "img":
        feature = np.array(v_features)
    elif feature_type == "txt":
        feature = np.array(t_features)
    
    item_list = keys
    # User 1 -> 0
    trn_positive_item["userID"] = trn_positive_item["userID"]-1
    tst_positive_item["userID"] = tst_positive_item["userID"]-1
    # Change trn&tst df item ID
    items = trn_positive_item["itemID"].values.tolist()
    trn_positive_item["itemID"] = trn_positive_item["itemID"].map(item_list.index)
    tst_positive_item["itemID"] = tst_positive_item["itemID"].map(item_list.index)
    # Change train ng pool item ID
    for u in range(num_user):
        train_ng_pool[u] = list(map(item_list.index, train_ng_pool[u]))
        test_negative[u] = list(map(item_list.index, test_negative[u]))

    return trn_positive_item, tst_positive_item, train_ng_pool, test_negative, num_user, len(item_list), feature


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

    def __init__(self, dataset, feature, negative, num_neg=4, istrain=False, use_feature = True):
        super(CustomDataset_movielens, self).__init__()
        self.dataset = dataset # df
        self.feature = feature # numpy
        self.negative = np.array(negative) # list->np
        self.istrain = istrain
        self.num_neg = num_neg
        self.use_feature = use_feature

        if not istrain:
            self.make_testset()
        else:
            self.dataset = np.array(self.dataset)

    def make_testset(self):
        assert not self.istrain
        users = np.unique(self.dataset["userID"])
        test_dataset = []
        for user in users:
            test_negative = self.negative[user]
            test_positive = self.dataset[self.dataset["userID"] == user]["itemID"].tolist()
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
            user, item_p = self.dataset[index]
            # Negative sampling
            ng_pool = np.array(self.negative[user])
            idx = np.random.choice(len(ng_pool),self.num_neg,replace=False)
            item_n = ng_pool[idx].tolist()
            if self.use_feature:
                feature_p = self.feature[item_p]
                feature_n = self.feature[item_n]
                return user, item_p, item_n, feature_p, feature_n
            else:
                return user, item_p, item_n, 0.0, 0.0
        else:
            user, item, label = self.dataset[index]
            if self.use_feature:
                feature = self.feature[item]
                return user, item, feature, label
            else:
                return user, item, 0.0, label

