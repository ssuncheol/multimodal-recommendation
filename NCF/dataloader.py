import torch
import random
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import random

class UserItemTrainDataset(Dataset):
    def __init__(self, train_u, train_i, train_r, neg_pool_dict, num_neg, **kwargs):
        self.users = train_u
        self.items = train_i
        self.ratings = train_r
        self.neg_pool_dict = neg_pool_dict
        self.image_dict = None
        self.text_dict = None
        self.num_neg = num_neg
        if kwargs['image'] is not None:
            self.image_dict = kwargs['image']
        if kwargs['text'] is not None:
            self.text_dict = kwargs['text']

    def __getitem__(self, index):
        negative_users = np.array(np.repeat(self.users[index], self.num_neg))
        negative_items = random.sample(self.neg_pool_dict[self.users[index]].tolist(), self.num_neg)
        negative_ratings = np.repeat(0, self.num_neg)
        negative_img = []
        negative_txt = []

        if (self.image_dict is not None) & (self.text_dict is not None):
            for i in negative_items:
                negative_img.append(self.image_dict[self.items[i].item()])
                negative_txt.append(self.text_dict[self.items[i].item()])
            negative_img.append(self.image_dict[self.items[index].item()])
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), negative_img, np.concatenate((negative_txt, [self.text_dict[self.items[index].item()]]))
        elif self.image_dict is not None:
            for i in negative_items:
                negative_img.append(self.image_dict[self.items[i].item()])
            negative_img.append(self.image_dict[self.items[index].item()])
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), negative_img
        elif self.text_dict is not None:
            for i in negative_items:
                negative_txt.append(self.text_dict[self.items[i].item()])
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), torch.cat((torch.FloatTensor(negative_txt), torch.FloatTensor(self.text_dict[self.items[index].item()]).unsqueeze(0)))
        else:
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]]))
        
    def __len__(self):
        return len(self.users)

class Make_Dataset(object):
    def __init__(self, df_train_p, df_train_n, df_test_p, df_test_n):
        self.train_data =self._train_data(df_train_p, df_train_n)
        self.evaluate_data = self._evaluate_data(df_test_p, df_test_n)
    
    def _train_data(self, df_train_p, df_train_n):
        users = np.array(df_train_p['userid'])
        items = np.array(df_train_p['train_pos'])
        ratings = np.repeat(1, len(df_train_p['userid'])).reshape(-1)
        neg_pool_dict = {}
        for i in df_train_n['userid'].unique():
            neg_pool_dict[i] = df_train_n.iloc[i]['train_negative'] # user별로 해당하는 negative item array를 갖는 dictionary 생성.
        return users, items, ratings, neg_pool_dict
    
    def _evaluate_data(self, df_test_p, df_test_n):
        test_pos_item_num = np.array(df_test_p.groupby(by=['userid'], as_index=False).count()['test_pos'])
        item_num_dict = {}
        df_test = pd.DataFrame({'userid':[], 'test_pos':[]})
        user = []
        item = []
        for i in df_test_p['userid'].unique():
            user.extend(df_test_p[df_test_p['userid'] == i]['userid'])
            item.extend(df_test_p[df_test_p['userid'] == i]['test_pos'])
            negative_item_list = df_test_n[df_test_n['userid'] == i]['test_negative'].item()
            user_list = np.ones(len(negative_item_list)) * i
            user.extend(user_list)
            item.extend(negative_item_list)
            item_num_dict[i] = test_pos_item_num[i] + df_test_n[df_test_n['userid'] == i]['test_negative'].item().shape[0] # 유저별 test item 개수 dictionary. key=userid, value=testitem개수 
        return user, item, item_num_dict, test_pos_item_num

class UserItemtestDataset(Dataset):
    def __init__(self, test_user, test_item, **kwargs):
        self.image_dict = None
        self.text_dict = None
        self.test_user = test_user
        self.test_item = test_item

        if kwargs['image'] is not None:
            self.image_dict = kwargs["image"]
        if kwargs['text'] is not None:
            self.text_dict = kwargs["text"]
            
    def __getitem__(self, index): 
        user, item = self.test_user[index], self.test_item[index]
        
        if (self.image_dict is not None) & (self.text_dict is not None):
            image_f = self.image_dict[item]
            text_f = self.text_dict[item]
            if type(image_f) != type(torch.Tensor([])):
                image_f = torch.FloatTensor(image_f)
            return torch.LongTensor([user]), torch.LongTensor([item]), image_f, torch.FloatTensor(text_f)
        elif self.image_dict is not None:
            image_f = self.image_dict[item]
            if type(image_f) != type(torch.Tensor([])):
                image_f = torch.FloatTensor(image_f)
            return torch.LongTensor([user]), torch.LongTensor([item]), image_f
        elif self.text_dict is not None:
            text_f = self.text_dict[item]
            return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor(text_f)
        else:
            return torch.LongTensor([user]), torch.LongTensor([item])

    def __len__(self):
        return len(self.test_user) 
