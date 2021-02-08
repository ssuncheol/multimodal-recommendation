import torch
import random
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import random

class UserItemTrainDataset(Dataset):
    def __init__(self, df_train_p, df_train_n, num_neg, **kwargs):
        self.users = np.array(df_train_p['userid'])
        self.items = np.array(df_train_p['train_pos'])
        self.ratings = np.repeat(1, len(df_train_p['userid'])).reshape(-1)
        self.image_dict = None
        self.text_dict = None
        self.df_train_n = df_train_n
        self.df_train_p = df_train_p
        self.num_neg = num_neg
        if 'image' in kwargs.keys() :
            self.image_dict = kwargs['image']
        if 'text' in kwargs.keys():
            self.text_dict = kwargs['text']

    def __getitem__(self, index):
        negative_users = np.array(np.repeat(self.users[index], self.num_neg))
        negative_items = random.sample(list(self.df_train_n[self.df_train_n['userid'] == self.users[index]]["train_negative"].item()), self.num_neg)
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
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), torch.cat((torch.FloatTensor(negative_text), torch.FloatTensor(self.text_dict[self.item_tensor[index].item()].unsqueeze(0))))
        else:
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]]))
        
    def __len__(self):
        return len(self.users)

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, **kwargs):
        
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        
        self.image_dict = None
        self.text_dict = None
        if 'image' in kwargs.keys() :
            self.image_dict = kwargs['image']

        if 'text' in kwargs.keys():
            self.text_dict = kwargs['text']

    def __getitem__(self, index): 
        if (self.image_dict is not None) & (self.text_dict is not None):
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], torch.FloatTensor(self.image_dict[self.item_tensor[index].item()]), torch.FloatTensor(self.text_dict[self.item_tensor[index].item()])
        elif self.image_dict is not None:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], torch.FloatTensor(self.image_dict[self.item_tensor[index].item()])
        elif self.text_dict is not None:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], torch.FloatTensor(self.text_dict[self.item_tensor[index].item()])
        else:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class Make_Dataset(object):
    def __init__(self, df_test_p, df_test_n):
        # self.df_train_p = df_train_p
        self.df_test_p = df_test_p
        self.df_test_n = df_test_n
        # self.trainset = self._trainset(df_train_p)
        self.evaluate_data = self._evaluate_data(df_test_p, df_test_n)
  
    # def _trainset(self, df_train_p):
    #     user = np.array(df_train_p['userid'])
    #     item = np.array(df_train_p['train_pos'])
    #     rating = np.repeat(1, len(df_train_p['userid'])).reshape(-1)
    #     return user, item, rating  
     
    def _evaluate_data(self, df_test_p, df_test_n):
        eval_dataset = []
        for u in df_test_p['userid'].unique():
            test_positive_item = np.array(df_test_p[df_test_p['userid'] == u]['test_pos'])
            test_negative_item = df_test_n[df_test_n['userid'] == u]['test_negative'].item()
            test_item = test_positive_item.tolist() + test_negative_item.tolist()
            label = np.zeros(len(test_item))
            label[:len(test_positive_item)] = 1
            test_user = np.ones(len(test_item)) * u
            eval_dataset.append([test_user.tolist(), test_item, label.tolist()])
        return eval_dataset
    

class UserItemtestDataset(Dataset):
    def __init__(self, test_dataset, **kwargs):
        self.image_dict2 = None
        self.text_dict2 = None
        self.test_dataset = test_dataset

        if "image" in kwargs.keys():
            self.image_dict2 = kwargs["image"]
        if "text" in kwargs.keys():
            self.text_dict2 = kwargs["text"]
            
    def __getitem__(self, index): 
        user, item, label = self.test_dataset[index]
        image_f = []
        text_f =[]
        if (self.image_dict2 is not None) & (self.text_dict2 is not None):
            for i in item:
                image_f.append(self.image_dict2[i])
                text_f.append(self.text_dict2[i])
            return user, item, image_f, text_f, label
        elif self.image_dict2 is not None:
            for i in item:
                image_f.append(self.image_dict2[i])
            return user, item, image_f, label
        elif self.text_dict2 is not None:
            for i in item:
                text_f.append(self.text_dict2[i])
            return user, item, text_f, label
        else:
            return user, item, label

    def __len__(self):
        return len(self.test_dataset)  
