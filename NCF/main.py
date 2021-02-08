import os
import pandas as pd
import numpy as np
from comet_ml import Experiment
import torch
import torch.nn as nn
import argparse
import time
import random
from dataloader import Make_Dataset, UserItemtestDataset, UserItemTrainDataset
from utils import optimizer
from model2 import NeuralCF
from evaluate import Engine
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collate import my_collate_trn_0, my_collate_trn_1, my_collate_trn_2, my_collate_tst_0, my_collate_tst_1, my_collate_tst_2 

import warnings
warnings.filterwarnings("ignore")

def main():
    experiment = Experiment(project_name='recommend', api_key='Bc3OhH0UQZebqFKyM77eLZnAm')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                type=str,
                default='/daintlab/home/tmddnjs3467/workspace/Amazon-office-raw',
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
    parser.add_argument('--feature',
                type=str,
                default='raw',
                help='raw(png) or pre(vector)')
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
    experiment.log_parameters(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # data load 
    df_train_p = pd.read_feather("%s/%s/train_positive.ftr" % (args.path, args.eval))
    df_train_n = pd.read_feather("%s/%s/train_negative.ftr" % (args.path, args.eval))
    df_test_p = pd.read_feather("%s/%s/test_positive.ftr" % (args.path, args.eval))
    df_test_n = pd.read_feather("%s/%s/test_negative.ftr" % (args.path, args.eval))
    user_index_info = pd.read_csv("%s/index-info/user_index.csv" % args.path)
    item_index_info = pd.read_csv("%s/index-info/item_index.csv" % args.path)

    user_index_dict = {}
    item_index_dict = {}
    img_dict = {}
    txt_dict = {}
    
    # image 쓸 건가
    if args.image:
        # raw image를 쓸 것인지, 전처리 해놓은 feature vector를 쓸 지.
        if args.feature == 'raw':
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
            img_list = os.listdir('%s/image' % args.path)
            for i in img_list:
                img_dict[item_index_info[item_index_info['itemid'] == i.split('.')[0]]['itemidx'].item()] = transform(Image.open(os.path.join('%s/image/%s' % (args.path, i))).convert('RGB'))
        else:
            img_feature = pd.read_pickle('%s/image_feature_vec.pickle' % args.path)
            for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
                item_index_dict[i] = j
            for i in item_index_dict.keys():
                img_dict[i] = img_feature[item_index_dict[i]] 
    # text 쓸 건가
    if args.text:
        txt_feature = pd.read_pickle('%s/text_feature_vec.pickle' % args.path)
        for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
            item_index_dict[i] = j
        for i in item_index_dict.keys():
            txt_dict[i] = txt_feature[item_index_dict[i]]
      
    num_user = df_train_p['userid'].nunique()
    num_item = item_index_info.shape[0]

    image_shape = 512
    text_shape =300
    
    # data 전처리
    MD = Make_Dataset(df_test_p, df_test_n)
    # user, item, rating = MD.trainset
    eval_dataset = MD.evaluate_data
    
    #NCF model
    if (args.image) & (args.text):
        print("IMAGE TEXT MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, feature=args.feature, image=image_shape, text=text_shape)    
    
    elif args.image:
        print("IMAGE MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, feature=args.feature, image=image_shape)  
    
    elif args.text:
        print("TEXT MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, feature=args.feature, text=text_shape)  

    else:
        print("MODEL")
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, feature=args.feature)
    
    model = nn.DataParallel(model)
    model = model.cuda()
    print(model)

    optim = optimizer(optim=args.optim, lr=args.lr, model=model, weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    
    # experiment.log_model(model)

    # train loader 생성
    if (args.image) & (args.text):               
        train_dataset = UserItemTrainDataset(df_train_p, df_train_n, args.num_neg, image=img_dict, text=txt_dict)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_trn_2, pin_memory =True)
    elif args.image:              
        train_dataset = UserItemTrainDataset(df_train_p, df_train_n, args.num_neg, image=img_dict)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_trn_1, pin_memory =True)
    elif args.text:           
        train_dataset = UserItemTrainDataset(df_train_p, df_train_n, args.num_neg, text=txt_dict)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_trn_1, pin_memory =True)
    else :                
        train_dataset = UserItemTrainDataset(df_train_p, df_train_n, args.num_neg)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_trn_0, pin_memory =True)

    step = 0
    # train 및 eval 시작
    for epoch in range(args.epochs):
        with experiment.train():
            print('Epoch {} starts !'.format(epoch+1))
            print('-' * 80)
            # model.train()
            total_loss = 0
            t1 = time.time()
            for batch_id, batch in enumerate(train_loader):
                # print("Train Loader 생성 완료 %.5f" % (time.time() - t1))
                optim.zero_grad()
                if (args.image) & (args.text):
                    users, items, ratings, image, text = batch[0], batch[1], batch[2], batch[3], batch[4]             
                    users, items, ratings, image, text = users.cuda(), items.cuda(), ratings.cuda(), image.cuda(), text.cuda()
                    output = model(users, items, image=image, text=text)
                elif args.image: 
                    users, items, ratings, image = batch[0], batch[1], batch[2], batch[3]                  
                    users, items, ratings, image = users.cuda(), items.cuda(), ratings.cuda(), image.cuda()
                    output = model(users, items, image=image)
                elif args.text:                   
                    users, items, ratings, text = batch[0], batch[1], batch[2], batch[3]
                    users, items, ratings, text = users.cuda(), items.cuda(), ratings.cuda(), text.cuda()
                    output = model(users, items, text=text)
                else :                   
                    users, items, ratings = batch[0], batch[1], batch[2]
                    users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                    output = model(users, items)
                
                step += 1
                loss = criterion(output, ratings)
                loss.backward()
                optim.step()
                loss = loss.item()
                experiment.log_metric('Batch Loss', loss, step=step)
                total_loss += loss

            t2 = time.time()
            print("train : ", t2 - t1) 

        engine = Engine(args.top_k)
        t3 = time.time()
        if (args.image) & (args.text):            
            test_dataset = UserItemtestDataset(eval_dataset, image=img_dict, text=txt_dict)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=my_collate_tst_2, pin_memory =True)
            hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval, interval=args.interval)
        elif args.image:
            test_dataset = UserItemtestDataset(eval_dataset, image=img_dict)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=my_collate_tst_1, pin_memory =True)
            hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, eval=args.eval, interval=args.interval)
        elif args.text:
            test_dataset = UserItemtestDataset(eval_dataset, text=txt_dict)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=my_collate_tst_1, pin_memory =True)
            hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, text=txt_dict, eval=args.eval, interval=args.interval)                
        else:
            test_dataset = UserItemtestDataset(eval_dataset)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=my_collate_tst_0, pin_memory =True)
            hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, eval=args.eval, interval=args.interval)  
        t4 = time.time()
        print('test:', t4 - t3) 
        
        if (epoch + 1) % args.interval == 0: 
            experiment.log_metrics({"epoch" : epoch,
                        "HR" : hit_ratio,
                        "HR2" : hit_ratio2,
                        "NDCG" : ndcg}, epoch=(epoch+1))
        
if __name__ == '__main__':
    main()
        
