import numpy as np
import pandas as pd
import os
import json
import time
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset
import pickle
import ast

def load_data(data_path, feature_path, feature_type):
    train_df = pd.read_feather(os.path.join(data_path, 'train_positive.ftr'))
    test_df = pd.read_feather(os.path.join(data_path, 'test_positive.ftr'))
    train_ng_pool = pd.read_feather(os.path.join(data_path, 'train_negative.ftr'))
    test_negative = pd.read_feather(os.path.join(data_path, 'test_negative.ftr'))
    
    train_df = train_df.astype('int64')
    test_df = test_df.astype('int64')
    train_df.rename(columns={"userid":"userID","train_pos":"itemID"}, inplace=True)
    test_df.rename(columns={"userid":"userID","test_pos":"itemID"}, inplace=True)
    train_ng_pool = train_ng_pool["train_negative"].tolist()
    test_negative = test_negative["test_negative"].tolist()
    
    index_info = pd.read_csv(os.path.join(data_path,'../index-info/item_index.csv'))
    
    num_user = max(train_df["userID"])+1
    num_item = max(train_df["itemID"])+1

    print(f"num user : {num_user} num item : {num_item}")
    #### Load feature data ####
    if feature_path.endswith('movielens'):
        
        with open(os.path.join(feature_path, 'image_feature_vec.pickle'), 'rb') as f:
            img_vec = pickle.load(f)
        with open(os.path.join(feature_path, 'text_feature_vec.pickle'), 'rb') as f:
            text_vec = pickle.load(f)

        index_list = index_info["itemidx"].tolist()
        id_list = index_info["itemid"].tolist()

        t_features = []
        v_features = []
        for i in range(len(index_list)):
            t_features.append(text_vec[id_list[i]])
            v_features.append(img_vec[id_list[i]])

    elif feature_path.endswith('Office'):
        doc2vec_model = Doc2Vec.load(os.path.join(feature_path, 'doc2vecFile'))
        vis_vec = np.load(os.path.join(feature_path, 'image_feature.npy'), allow_pickle=True).item()
        asin_dict = json.load(open(os.path.join(feature_path, 'asin_sample.json'), 'r'))

        text_vec = {}
        for asin in asin_dict:
            text_vec[asin] = doc2vec_model.docvecs[asin]
        import ipdb;ipdb.set_trace()
        asin_i_dic = {}
        for index, row in train_df.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin

        t_features = []
        v_features = []
        for i in range(num_item):
            t_features.append(text_vec[asin_i_dic[i]])
            v_features.append(vis_vec[asin_i_dic[i]])
    #### Load feature data ####
    #### Choose feature type ####
    if feature_type == "all":
        feature = np.concatenate((t_features, v_features), axis=1)
    elif feature_type == "img":
        feature = np.array(v_features)
    elif feature_type == "txt":
        feature = np.array(t_features)
    
    return train_df, test_df, train_ng_pool, test_negative, num_user, num_item, feature


class CustomDataset(Dataset):
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
        super(CustomDataset, self).__init__()
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
            if type(self.negative[user])==list:
                test_negative = self.negative[user]
            elif type(self.negative[user])==np.ndarray:
                test_negative = self.negative[user].tolist()
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


def inspect(df, num_inter):
    user = np.unique(df["userID"])
    x_user = []
    x_num_rating = []
    for i in user:
        if len(df[df["userID"] == i]) < num_inter:
            x_user.append(i)
            x_num_rating.append(len(df[df["userID"] == i]))

    return x_user, x_num_rating
