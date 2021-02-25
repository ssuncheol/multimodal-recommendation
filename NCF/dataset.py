from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
import time
import os
import pandas as pd
import numpy as np
import json
import pickle

def load_data(data_path, feature_type):
    start = time.time()
    feature_dir = os.path.join(data_path, '../')
    train_df = pd.read_feather(os.path.join(data_path, 'train_positive.ftr'))
    test_df = pd.read_feather(os.path.join(data_path, 'test_positive.ftr'))
    train_ng_pool = pd.read_feather(os.path.join(data_path, 'train_negative.ftr'))
    test_negative = pd.read_feather(os.path.join(data_path, 'test_negative.ftr'))
    
    test_pos_item_num = np.array(test_df.groupby(by=['userid'], as_index=False).count()['test_pos']) # index = useridx, value = test postive item 개수
    item_num_dict = {}
    for i in test_negative['userid'].unique():    
        item_num_dict[i] = test_pos_item_num[i] + test_negative[test_negative['userid'] == i]['test_negative'].item().shape[0] # 유저별 test item 개수 dictionary. key=userid, value=testitem개수 
        
    train_df = train_df.astype('int64')
    test_df = test_df.astype('int64')
    train_df.rename(columns={"userid": "userID", "train_pos": "itemID"}, inplace=True)
    test_df.rename(columns={"userid": "userID", "test_pos": "itemID"}, inplace=True)
    train_ng_pool = train_ng_pool["train_negative"].tolist()
    test_negative = test_negative["test_negative"].tolist()

    index_info = pd.read_csv(os.path.join(data_path, '../index-info/item_index.csv'))
    num_user = max(train_df["userID"]) + 1
    num_item = index_info.shape[0]

    id_list = index_info["itemid"].tolist()

    with open(os.path.join(feature_dir, "item_meta.json"), "rb") as f:
        meta_data = json.load(f)
    with open(os.path.join(feature_dir, 'text_feature_vec.pickle'), 'rb') as f:
        text_vec = pickle.load(f)
    image_path_list = []
    t_features = []
    for item_id in id_list:
        t_features.append(text_vec[item_id])
        img_path = meta_data[f"{item_id}"]["image_path"]
        image_path_list.append(os.path.abspath(os.path.join(feature_dir, img_path)))

    t_features = np.array(t_features)
    t_features = dict(enumerate(t_features, 0))
    image_path_list = np.array(image_path_list)
    images = {}

    if feature_type == "all" or feature_type == "img":
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        for i in range(len(image_path_list)):
            img = Image.open(image_path_list[i]).convert("RGB")
            images[i] = transform(img)
        # images = torch.stack(images)
        # images = dict(enumerate(images,0))

    print(f"Data Loaded. num user : {num_user} num item : {num_item} {time.time() - start:.4f} sec")
    return train_df, test_df, train_ng_pool, test_negative, num_user, num_item, t_features, images, test_pos_item_num, item_num_dict


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

    def __init__(self, model_type, dataset, text_feature, images, negative, num_neg=4, istrain=False, feature_type="all"):
        super(CustomDataset, self).__init__()
        self.model_type = model_type
        self.dataset = dataset  # df
        self.text_feature = text_feature  # dictionary(np)
        self.images = images  # dictionary(tensor)
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
        test_dataset=[]
        for user in users:
            if type(self.negative[user]) == list:
                test_negative = self.negative[user]
            elif type(self.negative[user]) == np.ndarray:
                test_negative = self.negative[user].tolist()
            test_positive = self.dataset[self.dataset["userID"] == user]["itemID"].tolist()
            item = test_positive + test_negative
            # label = np.zeros(len(item))
            # label[:len(test_positive)] = 1
            test_user = np.ones(len(item)) * user

            testset=np.vstack((test_user,np.array(item))).T
            test_dataset.append(testset)

        test_dataset=np.concatenate(test_dataset).astype(np.int64)

        self.dataset = test_dataset



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.istrain:
            user, item_p = self.dataset[index]
            # Negative sampling
            ng_pool = np.array(self.negative[user])
            idx = np.random.choice(len(ng_pool), self.num_neg, replace=False)
            # idx = random.sample(list(range(0,len(ng_pool))), self.num_neg)
            item_n = ng_pool[idx].tolist()

            ####
            item_idx = item_n.copy()
            item_idx.insert(0, item_p)
            t_feature, img = [0.0,0.0,0.0,0.0,0.0], torch.Tensor([0.0,0.0,0.0,0.0,0.0])
            t_feature_p, t_feature_n, img_p, img_n = 0.0, [0.0,0.0,0.0,0.0], torch.Tensor([0.0]), torch.zeros(self.num_neg,1)

            if self.feature_type == "txt" or self.feature_type == "all":
                # t_feature = self.text_feature[item_idx]
                t_feature = []
                for i in item_idx:
                    t_feature.append(self.text_feature[i])
                t_feature_p = t_feature[0]
                t_feature_n = np.array(t_feature[1:])

            if self.feature_type == "img" or self.feature_type == "all":
                # img = self.images[item_idx]
                img = []
                for j in item_idx:
                    img.append(self.images[j])
                img_p = img[0]
                img_n = torch.stack(img[1:])
            if self.model_type == 'MAML':
                return user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n
            else:
                user = np.repeat(user, self.num_neg + 1)
                rating = np.repeat(0., self.num_neg + 1)
                rating[0] = 1.
                return user, item_idx, rating, t_feature, torch.stack(img)
        else:
            user, item = self.dataset[index]

            t_feature, img = [0.0], torch.Tensor([0.0])

            if self.feature_type == "txt" or self.feature_type == "all":
                for i in [item]:
                    t_feature = np.array(self.text_feature[i])

            if self.feature_type == "img" or self.feature_type == "all":
                for j in [item]:
                    img = self.images[j]

            return user, item, t_feature, img
