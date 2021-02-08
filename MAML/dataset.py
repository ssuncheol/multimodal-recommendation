import numpy as np
import pandas as pd
import os
import json
import time
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torch
import random


def load_data(data_path, feature_type):
    feature_dir = os.path.join(data_path, '../')
    train_df = pd.read_feather(os.path.join(data_path, 'train_positive.ftr'))
    test_df = pd.read_feather(os.path.join(data_path, 'test_positive.ftr'))
    train_ng_pool = pd.read_feather(os.path.join(data_path, 'train_negative.ftr'))
    test_negative = pd.read_feather(os.path.join(data_path, 'test_negative.ftr'))

    train_df = train_df.astype('int64')
    test_df = test_df.astype('int64')
    train_df.rename(columns={"userid": "userID", "train_pos": "itemID"}, inplace=True)
    test_df.rename(columns={"userid": "userID", "test_pos": "itemID"}, inplace=True)
    train_ng_pool = train_ng_pool["train_negative"].tolist()
    test_negative = test_negative["test_negative"].tolist()

    index_info = pd.read_csv(os.path.join(data_path, '../index-info/item_index.csv'))

    num_user = max(train_df["userID"]) + 1
    num_item = max(train_df["itemID"]) + 1

    index_list = index_info["itemidx"].tolist()
    id_list = index_info["itemid"].tolist()

    with open(os.path.join(feature_dir, "item_meta.json"), "rb") as f:
        meta_data = json.load(f)
    # with open(os.path.join(feature_dir, 'image_feature_vec.pickle'), 'rb') as f:
    #     image_vec = pickle.load(f)
    with open(os.path.join(feature_dir, 'text_feature_vec.pickle'), 'rb') as f:
        text_vec = pickle.load(f)
    image_path_list = []
    t_features = []
    # v_features = []
    for item_id in id_list:
        t_features.append(text_vec[item_id])
        # v_features.append(image_vec[item_id])
        img_path = meta_data[f"{item_id}"]["image_path"]
        image_path_list.append(os.path.abspath(os.path.join(feature_dir, img_path)))

    t_features = np.array(t_features)
    # v_features = np.array(v_features)
    image_path_list = np.array(image_path_list)
    images = []

    if feature_type == "all" or feature_type == "img":
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        for i in range(len(image_path_list)):
            img = Image.open(image_path_list[i]).convert("RGB")
            img = transform(img)
            images.append(img)

    print(f"Data Loaded. num user : {num_user} num item : {num_item}")
    return train_df, test_df, train_ng_pool, test_negative, num_user, num_item, t_features, torch.stack(images)


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

    def __init__(self, dataset, text_feature, images, negative, num_neg=4, istrain=False, feature_type="all"):
        super(CustomDataset, self).__init__()
        self.dataset = dataset  # df
        self.text_feature = text_feature  # numpy
        self.images = images  # Tensor
        self.negative = np.array(negative)  # list->np
        self.istrain = istrain
        self.num_neg = num_neg
        self.feature_type = feature_type

        if not istrain:
            self.make_testset()
        else:
            self.dataset = np.array(self.dataset)

    def make_testset(self):
        assert not self.istrain
        users = np.unique(self.dataset["userID"])
        test_dataset = []
        for user in users:
            if type(self.negative[user]) == list:
                test_negative = self.negative[user]
            elif type(self.negative[user]) == np.ndarray:
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

            ####
            item_idx = item_n.copy()
            item_idx.insert(0,item_p)

            t_feature_p, t_feature_n, img_p, img_n = 0.0, 0.0, torch.Tensor([0.0]), torch.Tensor([0.0])
            
            if self.feature_type == "txt" or self.feature_type == "all":
                t_feature = self.text_feature[item_idx]
                t_feature_p = t_feature[0]
                t_feature_n = t_feature[1:]

            if self.feature_type == "img" or self.feature_type == "all":
                img = self.images[item_idx]
                img_p = img[0]
                img_n = img[1:]

            return user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n

        else:
            user, item, label = self.dataset[index]

            t_feature, img = 0.0, torch.Tensor([0.0])

            if self.feature_type == "txt" or self.feature_type == "all":
                t_feature = self.text_feature[item]
            
            if self.feature_type == "img" or self.feature_type == "all":
                img = self.images[item]

            return user, item, t_feature, img, label
