import torch
import random
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset 

#seed 
random.seed(42)

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)



class Make_Dataset(object):
    def __init__(self, ratings):
        #args : ratings: pd.DataFrame, which contains 5 columns = ['userid', 'train_positive', 'train_negative', 'test_rating','test_negative']
        
        assert 'userid' in ratings.columns
        assert 'train_positive' in ratings.columns
        assert 'train_negative' in ratings.columns
        assert 'test_positive' in ratings.columns
        assert 'test_negative' in ratings.columns

        self.ratings = ratings
        self.positive_len = ratings["train_positive"].map(len)
        self.trainset = self._trainset(ratings)
        self.evaluate_data = self._evaluate_data(ratings)
        
        
    def _trainset(self, ratings):
        #make train data
        user = np.array(np.repeat(ratings["userid"], self.positive_len))
        item = np.array([item for items in ratings['train_positive'] for item in items])
        print(item)
        rating = np.repeat(1, self.positive_len.sum()).reshape(-1)
        print(min(item), max(item), item.shape)
        print(min(user), max(user), user.shape)
        return user, item, rating

    
    def _evaluate_data(self, ratings):
        #make evaluate data
        test_user = np.array(ratings["userid"])
        test_item = np.array(ratings["test_positive"])
        
        test_negative_user = np.array(np.repeat(ratings["userid"], 99))
        test_negative_item = np.array(list(ratings["test_negative"])).reshape(-1)
        print(min(test_negative_item), max(test_negative_item), test_negative_item.shape)
        return [torch.LongTensor(test_user), torch.LongTensor(test_item), torch.LongTensor(test_negative_user),
                torch.LongTensor(test_negative_item)]
    
class SampleGenerator(object):
    def __init__(self, user, item, rating, ratings, positive_len,num_neg):
        self.user = user # 전처리한 데이터
        self.item = item # 전처리한 데이터
        self.rating = rating # 전처리한 데이터
        self.ratings = ratings # 원본 데이터
        self.num_neg = num_neg
        self.positive_len = positive_len
        self.train_user, self.train_item, self.train_rating = self.total_train(ratings, positive_len, num_neg)
        
    def total_train(self,ratings,positive_len, num_neg):
        positive_len.rename("len", inplace = True)
        ratings = pd.concat([ratings,positive_len], axis = 1)

        negative_user = np.array(np.repeat(ratings["userid"], ratings["len"] * num_neg)).reshape(-1) 
        ratings["negative_items"] = ratings.apply(lambda x : np.random.choice(list(x["train_negative"]), x["len"] * num_neg), axis = 1)
        negative_item = np.array([item for items in ratings['negative_items'] for item in items])
        negative_rating = np.repeat(0, ratings["len"].sum() * num_neg).reshape(-1)
        
        train_user = np.hstack((self.user,negative_user))
        train_item = np.hstack((self.item,negative_item))
        train_rating = np.hstack((self.rating,negative_rating))
 
        return train_user, train_item, train_rating
    
    
    def instance_a_train_loader(self, batch_size):
        user = self.train_user
        item = self.train_item
        rating = self.train_rating        
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating))

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
    
    