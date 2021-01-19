import torch
import random
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset 

#seed 
random.seed(42)

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
    def __init__(self, df_train_p, df_train_n, df_test_p, df_test_n):
        self.positive_len = df_train_p.groupby(by=['userid'], as_index=False).count()['train_pos']
        self.positive_len.columns = ['positive_len']
        self.negative_len = df_test_n["test_negative"].map(len)
        self.df_train_p = df_train_p
        self.df_train_n = df_train_n
        self.df_test_p = df_test_p
        self.df_test_n = df_test_n
        self.trainset = self._trainset(df_train_p)
        self.evaluate_data = self._evaluate_data(df_test_p, df_test_n)
        
    def _trainset(self, df_train_p):
        #make train data
        user = np.array(df_train_p['userid'])
        item = np.array(df_train_p['train_pos'])
        rating = np.repeat(1, len(df_train_p['userid'])).reshape(-1)
        return user, item, rating
        
     
    def _evaluate_data(self, df_test_p, df_test_n):
        #make evaluate data
        test_user = np.array(df_test_p['userid'])
        test_item = np.array(df_test_p['test_pos'])
        test_negative_user = np.array(np.repeat(df_test_n["userid"], self.negative_len))
        test_negative_item = np.array([item for items in df_test_n['test_negative'] for item in items])
        return [torch.LongTensor(test_user), torch.LongTensor(test_item), torch.LongTensor(test_negative_user),
                torch.LongTensor(test_negative_item)]
    
     
class SampleGenerator(object):
    def __init__(self, user, item, rating, df_train_n, positive_len, num_neg):
        self.user = user # 전처리한 데이터
        self.item = item # 전처리한 데이터
        self.rating = rating # 전처리한 데이터
        self.df_train_n = df_train_n # 원본 데이터
        self.num_neg = num_neg
        self.positive_len = positive_len
        self.train_user, self.train_item, self.train_rating = self.total_train(df_train_n, positive_len, num_neg)
        
    def total_train(self, df_train_n, positive_len, num_neg):
        positive_len.rename("len", inplace=True)
        df_train_n = pd.concat([df_train_n, positive_len], axis=1)

        negative_user = np.array(np.repeat(df_train_n["userid"], df_train_n["len"] * num_neg)).reshape(-1) 
        df_train_n["negative_items"] = df_train_n.apply(lambda x : np.random.choice(list(x["train_negative"]), x["len"] * num_neg), axis=1)
        negative_item = np.array([item for items in df_train_n['negative_items'] for item in items])
        negative_rating = np.repeat(0, df_train_n["len"].sum() * num_neg).reshape(-1)
        
        train_user = np.hstack((self.user,negative_user))
        train_item = np.hstack((self.item,negative_item))
        train_rating = np.hstack((self.rating,negative_rating))
        
        return train_user, train_item, train_rating
    
    
    def instance_a_train_loader(self, batch_size, **kwargs):
        user = self.train_user
        item = self.train_item
        rating = self.train_rating
        
        if ("image" in kwargs.keys()) & ("text" in kwargs.keys()):
            print("IMAGE TEXT")
            dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating),
                                        image=kwargs["image"],text=kwargs["text"])
        elif "image" in kwargs.keys():
            print("IMAGE")
            dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating),
                                        image=kwargs["image"])
        elif "text" in kwargs.keys():
            print("TEXT")
            dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating),
                                        text=kwargs["text"])
            
            
        else:            
            print("NOT")
            dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating))
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)



class UserItemtestDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, **kwargs):
        
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.image_dict2 = None
        self.text_dict2 = None


        if "image" in kwargs.keys():
            self.image_dict2 = kwargs["image"]
        if "text" in kwargs.keys():
            self.text_dict2 = kwargs["text"]
            
    def __getitem__(self, index): 
        if (self.image_dict2 is not None) & (self.text_dict2 is not None):
            return self.user_tensor[index], self.item_tensor[index], torch.FloatTensor(self.image_dict2[self.item_tensor[index].item()]), torch.FloatTensor(self.text_dict2[self.item_tensor[index].item()])
        elif self.image_dict2 is not None:
            return self.user_tensor[index], self.item_tensor[index], torch.FloatTensor(self.image_dict2[self.item_tensor[index].item()])
        elif self.text_dict2 is not None:
            return self.user_tensor[index], self.item_tensor[index], torch.FloatTensor(self.text_dict2[self.item_tensor[index].item()])
        else:
            return self.user_tensor[index], self.item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)




class testGenerator(object):
    def __init__(self,test_user,test_item, **kwargs):
        self.user = test_user
        self.item = test_item
        

    def instance_a_test_loader(self, batch_size, **kwargs):
        user = self.user
        item = self.item
        
        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            print("TEST IMAGE TEXT")
            dataset = UserItemtestDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        image=kwargs['image'],text=kwargs['text'])
        elif ('image' in kwargs.keys()):
            print("TEST IMAGE")
            dataset = UserItemtestDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        image=kwargs['image'])
        elif ('text' in kwargs.keys()):
            print("TEST TEXT")
            dataset = UserItemtestDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        text=kwargs['text'])
        else:            
            print("TEST")
            dataset = UserItemtestDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)