import os
import pandas as pd
import numpy as np
import wandb
from gensim.models.doc2vec import Doc2Vec
import torch
import torch.nn as nn
import argparse
import time
import random
from sklearn.preprocessing import LabelEncoder 
from dataloader import Make_Dataset, SampleGenerator, testGenerator
from utils import optimizer
from model import NeuralCF
from evaluate import Engine
from metrics import MetronAtK

import warnings
warnings.filterwarnings("ignore")

def main():
    #wandb.init(project="Real Total NCF")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="amazon",
                help='dataset')
    parser.add_argument('--path',
                type=str,
                default='/daintlab/home/tmddnjs3467/workspace',
                help='path')
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
                help='train epochs')
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
    #wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    

    # Select data 

    if args.data == "amazon":

        df_train_p = pd.read_feather("%s/Amazon-office-raw/%s/train_positive.ftr" % (args.path, args.eval))
        df_train_n = pd.read_feather("%s/Amazon-office-raw/%s/train_negative.ftr" % (args.path, args.eval))
        df_test_p = pd.read_feather("%s/Amazon-office-raw/%s/test_positive.ftr" % (args.path, args.eval))
        df_test_n = pd.read_feather("%s/Amazon-office-raw/%s/test_negative.ftr" % (args.path, args.eval))
        user_index_info = pd.read_csv("%s/Amazon-office-raw/index-info/user_index.csv" % args.path)
        item_index_info = pd.read_csv("%s/Amazon-office-raw/index-info/item_index.csv" % args.path)
        txt_feature = pd.read_pickle('%s/Amazon-office-raw/text_feature_vec.pickle' % args.path)
        num_user = 101187
        num_item = 18371
        
        user_index_dict={}
        item_index_dict={}
        txt_dict={}
        for i, j in zip(user_index_info['useridx'], user_index_info['userid']):
            user_index_dict[i] = j
        for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
            item_index_dict[i] = j
        for i in item_index_dict.keys():
            # img_dict[i] = img_feature[item_index_dict[i]] 
            txt_dict[i] = txt_feature[item_index_dict[i]]
        
        # ##feature들 있을 때 쓰는 코드
        # itemid_asin_dict = dict(zip(total["itemID"], total["asin"]))
        # userid_reviewerID_dict = dict(zip(total["userID"], total["reviewerID"]))
        # img_dict = {}
        # txt_dict = {}
        # for i in list(itemid_asin_dict.keys()):
        #     img_dict[i] = image_feature[itemid_asin_dict[i]]
        # for j in list(userid_reviewerID_dict.keys()):
        #     txt_dict[j] = text_feature.infer_vector([userid_reviewerID_dict[j]])
        # image_shape = 4096
        # text_shape = 512
        
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
        for i, j in zip(user_index_info['useridx'], user_index_info['userid']):
            user_index_dict[i] = j
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
    test_user, test_item, test_negative_user, test_negative_item = MD.evaluate_data
    
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
    
    #wandb.watch(model)

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
            #import pdb; pdb.set_trace()
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

        engine = Engine()
        if args.data == "amazon":
            #if args.feature == True:
            a=time.time() 
            evaluate_data = testGenerator(test_user, test_item)
            evaluate_data_neg = testGenerator(test_negative_user, test_negative_item)
            if (args.image == True) & (args.text == True):
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, image=img_dict, text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, image=img_dict, text=txt_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval, interval=args.interval)
            elif args.image == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, image=img_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, image=img_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, image=img_dict, eval=args.eval, interval=args.interval)
            elif args.text == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, text=txt_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, text=txt_dict, eval=args.eval, interval=args.interval)                
            else:
                a=time.time() 
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, eval=args.eval, interval=args.interval)  
            b=time.time()
            print('test:', b-a) 
                

        else:
            a=time.time() 
            evaluate_data = testGenerator(test_user, test_item)
            evaluate_data_neg = testGenerator(test_negative_user, test_negative_item)
            if (args.image == True) & (args.text == True):
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, image=img_dict, text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, image=img_dict, text=txt_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval, interval=args.interval)
            elif args.image == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, image=img_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, image=img_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, image=img_dict, eval=args.eval, interval=args.interval)
            elif args.text == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100, text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100, text=txt_dict) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, text=txt_dict, eval=args.eval, interval=args.interval)                
            else:
                a=time.time() 
                test_loader = evaluate_data.instance_a_test_loader(len(test_user) // 100)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user) // 100) 
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, test_negative_loader, epoch_id=epoch, eval=args.eval, interval=args.interval)  
            b=time.time()
            print('test:' ,b-a) 
        if args.eval == 'ratio-split':
            if (epoch + 1) % args.interval == 0: 
                #wandb.log({"epoch" : epoch,
                #            "HR" : hit_ratio,
                #            "HR2" : hit_ratio2,
                #            "NDCG" : ndcg})
                N.append(ndcg)
        else:
            #wandb.log({"epoch" : epoch,
            #            "HR" : hit_ratio,
            #            "NDCG" : ndcg})
            N.append(ndcg)

if __name__ == '__main__':
    main()
        
