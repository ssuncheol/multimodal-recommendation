import os
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn
import argparse
import time
import random
from dataloader import Make_Dataset, SampleGenerator, UserItemtestDataset
from utils import optimizer
from model import NeuralCF
from evaluate import Engine
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

def my_collate_tst_2(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    image = [items[2] for items in batch]
    image = torch.Tensor(image)
    text = [items[3] for items in batch]
    text = torch.Tensor(text)
    label = [items[4] for items in batch]
    label = torch.Tensor(label)
    
    return [user, item, image, text, label]

def my_collate_tst_1(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    feature = [items[2] for items in batch]
    feature = torch.Tensor(feature)
    label = [items[3] for items in batch]
    label = torch.Tensor(label)

    return [user, item, feature, label]

def my_collate_tst_0(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    label = [items[2] for items in batch]
    label = torch.Tensor(label)
    
    return [user, item, label]

def main():
    wandb.init(project="amazon leave one out")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="amazon",
                help='dataset')
    parser.add_argument('--path',
                type=str,
                default='/daintlab/home/tmddnjs3467/workspace',
                help='path')
    parser.add_argument('--top_k',
                type=int,
                default=10,
                help='top_k')
    parser.add_argument('--image',
                type=bool,
                default=False,
                help='image')
    parser.add_argument('--text',
                type=bool,
                default=False,
                help='text')    
    parser.add_argument('--optim',
                type=str,
                default='adam',
                help='optimizer')
    parser.add_argument('--lr',
                type=float,
                default=0.001,
                help='learning rate')
    parser.add_argument('--epochs',
                type=int,
                default=50,
                help='learning rate')
    parser.add_argument('--drop_rate',
                type=float,
                default=0.0,
                help= 'dropout rate')
    parser.add_argument('--batch_size',
                type=int,
                default=1024,
                help='train batch size')
    parser.add_argument('--latent_dim_mf',
                type=int,
                default=8,
                help='latent_dim_mf')
    parser.add_argument('--num_layers',
                type=int,
                default=1,
                help='num layers')
    parser.add_argument('--num_neg',
                type=int,
                default=4,
                help='negative sample')
    parser.add_argument('--l2',
                type=float,
                default=0.0,
                help='l2_regularization')
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')
    parser.add_argument('--eval',
                type=str,
                default='ratio-split',
                help='protocol')
    parser.add_argument('--interval',
                type=int,
                default=1,
                help='evaluation interval')
    args = parser.parse_args()
    wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    

    # Select data 

    if args.data == "amazon":

        df_train_p = pd.read_feather("%s/Amazon-office-raw/%s/train_positive.ftr" % (args.path, args.eval))
        df_train_n = pd.read_feather("%s/Amazon-office-raw/%s/train_negative.ftr" % (args.path, args.eval))
        df_test_p = pd.read_feather("%s/Amazon-office-raw/%s/test_positive.ftr" % (args.path, args.eval))
        df_test_n = pd.read_feather("%s/Amazon-office-raw/%s/test_negative.ftr" % (args.path, args.eval))
        user_index_info = pd.read_csv("%s/Amazon-office-raw/index-info/user_index.csv" % args.path)
        item_index_info = pd.read_csv("%s/Amazon-office-raw/index-info/item_index.csv" % args.path)
        img_feature = pd.read_pickle('%s/Amazon-office-raw/image_feature_vec.pickle' % args.path)
        txt_feature = pd.read_pickle('%s/Amazon-office-raw/text_feature_vec.pickle' % args.path)
        num_user = 54084
        num_item = 18316
        
        ## reindex 때문에 feature dict 다시 만들기 위한 과정
        user_index_dict = {}
        item_index_dict = {}
        img_dict = {}
        txt_dict = {}
        # for i, j in zip(user_index_info['useridx'], user_index_info['userid']):
        #     user_index_dict[i] = j
        for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
            item_index_dict[i] = j
        for i in item_index_dict.keys():
            img_dict[i] = img_feature[item_index_dict[i]] 
            txt_dict[i] = txt_feature[item_index_dict[i]]
        image_shape = 512
        text_shape = 300
        
    elif args.data == "movie":

        df_train_p = pd.read_feather("%s/Movielens-raw/%s/train_positive.ftr" % (args.path, args.eval))
        df_train_n = pd.read_feather("%s/Movielens-raw/%s/train_negative.ftr" % (args.path, args.eval))
        df_test_p = pd.read_feather("%s/Movielens-raw/%s/test_positive.ftr" % (args.path, args.eval))
        df_test_n = pd.read_feather("%s/Movielens-raw/%s/test_negative.ftr" % (args.path, args.eval))
        img_feature = pd.read_pickle('%s/movielense/image_feature_vec.pickle' % args.path)
        txt_feature = pd.read_pickle('%s/movielense/text_feature_vec.pickle' % args.path)
        user_index_info = pd.read_csv("%s/Movielens-raw/index-info/user_index.csv" % args.path)
        item_index_info = pd.read_csv("%s/Movielens-raw/index-info/item_index.csv" % args.path)
        
        ## reindex 때문에 feature dict 다시 만들기 위한 과정
        user_index_dict = {}
        item_index_dict = {}
        img_dict = {}
        txt_dict = {}
        # for i, j in zip(user_index_info['useridx'], user_index_info['userid']):
        #     user_index_dict[i] = j
        for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
            item_index_dict[i] = j
        for i in item_index_dict.keys():
            img_dict[i] = img_feature[item_index_dict[i]] 
            txt_dict[i] = txt_feature[item_index_dict[i]]
        
        num_user = 6040
        num_item = 3659
        image_shape = 512
        text_shape = 300
    else:
        print("데이터가 존재하지 않습니다.")
        return 0

    MD = Make_Dataset(df_train_p, df_train_n, df_test_p, df_test_n)
    user, item, rating = MD.trainset
    eval_dataset = MD.evaluate_data
                    
    #NCF model
    if (args.image == True) & (args.text == True):
        print("IMAGE TEXT MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, image=image_shape, text=text_shape)    
    
    elif args.image == True:
        print("IMAGE MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, image=image_shape)  
    
    elif args.text == True:
        print("TEXT MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, text=text_shape)  

    else:
        print("MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers)
    
    model = nn.DataParallel(model)
    model = model.cuda()
    print(model)

    optim = optimizer(optim=args.optim, lr=args.lr, model=model, weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    
    wandb.watch(model)

    N = []
    for epoch in range(args.epochs):
        print('Epoch {} starts !'.format(epoch+1))
        print('-' * 80)
        t1 = time.time()
        model.train()
        total_loss = 0
        sample = SampleGenerator(user=user, item=item, 
                                 rating=rating, df_train_n=df_train_n, 
                                 positive_len=MD.positive_len, num_neg=args.num_neg)
        if (args.image == True) & (args.text == True):               
            train_loader = sample.instance_a_train_loader(args.batch_size, image=img_dict, text=txt_dict)
        elif args.image == True:              
            train_loader = sample.instance_a_train_loader(args.batch_size, image=img_dict)
        elif args.text == True:           
            train_loader = sample.instance_a_train_loader(args.batch_size, text=txt_dict)
        else :                
            train_loader = sample.instance_a_train_loader(args.batch_size)
        
        print("Train Loader 생성 완료")
        for batch_id, batch in enumerate(train_loader):
            optim.zero_grad()
            if (args.image == True) & (args.text == True):
                users, items, ratings, image, text = batch[0], batch[1], batch[2], batch[3], batch[4]             
                users, items, ratings, image, text = users.cuda(), items.cuda(), ratings.cuda(), image.cuda(), text.cuda()
                output = model(users, items, image=image, text=text)
            elif args.image == True: 
                users, items, ratings, image = batch[0], batch[1], batch[2], batch[3]                  
                users, items, ratings, image = users.cuda(), items.cuda(), ratings.cuda(), image.cuda()
                output = model(users, items, image=image)
            elif args.text == True:                   
                users, items, ratings, text = batch[0], batch[1], batch[2], batch[3]
                users, items, ratings, text = users.cuda(), items.cuda(), ratings.cuda(), text.cuda()
                output = model(users, items, text=text)
            else :                   
                users, items, ratings = batch[0], batch[1], batch[2]
                users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                output = model(users, items)
             
            loss = criterion(output, ratings)
            loss.backward()
            optim.step()
            loss = loss.item()
            wandb.log({'Batch Loss': loss})
            total_loss += loss

        t2 = time.time()
        print("train : ", t2 - t1) 

        engine = Engine(args.top_k)
        if args.data == "amazon":
            a=time.time()
            if (args.image == True) & (args.text == True):            
                test_dataset = UserItemtestDataset(eval_dataset, image=img_dict, text=txt_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_2, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval, interval=args.interval)
            elif args.image == True:
                test_dataset = UserItemtestDataset(eval_dataset, image=img_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_1, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, eval=args.eval, interval=args.interval)
            elif args.text == True:
                test_dataset = UserItemtestDataset(eval_dataset, text=txt_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_1, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, text=txt_dict, eval=args.eval, interval=args.interval)                
            else:
                test_dataset = UserItemtestDataset(eval_dataset)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_0, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, eval=args.eval, interval=args.interval)  
            b=time.time()
            print('test:', b-a) 

        else:
            a=time.time() 
            if (args.image == True) & (args.text == True):            
                test_dataset = UserItemtestDataset(eval_dataset, image=img_dict, text=txt_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_2, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval, interval=args.interval)
            elif args.image == True:
                test_dataset = UserItemtestDataset(eval_dataset, image=img_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_1, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, eval=args.eval, interval=args.interval)
            elif args.text == True:
                test_dataset = UserItemtestDataset(eval_dataset, text=txt_dict)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_1, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, text=txt_dict, eval=args.eval, interval=args.interval)                
            else:
                test_dataset = UserItemtestDataset(eval_dataset)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst_0, pin_memory =True)
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, eval=args.eval, interval=args.interval)  
            b=time.time()
            print('test:' ,b-a) 
        
        if (epoch + 1) % args.interval == 0: 
            wandb.log({"epoch" : epoch,
                        "HR" : hit_ratio,
                        "HR2" : hit_ratio2,
                        "NDCG" : ndcg})

if __name__ == '__main__':
    main()
        
