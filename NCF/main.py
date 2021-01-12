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

from dataloader import Make_Dataset, SampleGenerator
from utils import optimizer
from model import NeuralCF
from evaluate import Engine
from metrics import MetronAtK

import warnings
warnings.filterwarnings("ignore")

# amazon preprocessing
def amazon(train, test):
    train = train[["userID","itemID"]]
    test = test[["userID","itemID"]]
    train_item = set(train["itemID"])
    test_item = set(test["itemID"])
    items = train_item | test_item
    print("User : {} ~ {}".format(min(train["userID"]), max(train["userID"])))
    print("Item : {} ~ {}".format(min(items), max(items)))
    train = train.groupby(["userID"])["itemID"].apply(list).to_frame(name = "train_positive")
    test = test.groupby(["userID"])["itemID"].apply(list).to_frame(name = "test_positive")
    data = pd.merge(train,test, on = "userID")
    data["test_negative"] = data.apply(lambda x : list(items - set(x["train_positive"]) - set(x["test_positive"])), axis = 1)
    data["train_negative"] = data.apply(lambda x : list(items - set(x["train_positive"]) - set(x["test_positive"])), axis = 1)
    data["userid"] = data.index
    image_feature = np.load("/daintlab/data/amazon_office/image_feature.npy", allow_pickle=True)
    image_feature = image_feature.item()
    text_feature = Doc2Vec.load("/daintlab/data/amazon_office/doc2vecFile")
    return data, image_feature, text_feature  


def main():
    wandb.init(project="NCF-side")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="amazon",
                help='dataset')
    parser.add_argument('--feature',
                type=bool,
                default=False,
                help='feature')
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
    args = parser.parse_args()
    wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.data == "amazon":
        train = pd.read_csv("/daintlab/data/amazon_office/train.csv")
        test = pd.read_csv("/daintlab/data/amazon_office/test.csv")
        total = pd.concat([train,test])
        num_user = 4874
        num_item = 2406
        item2image = dict(zip(total["itemID"], total["asin"]))
        user2review = dict(zip(total["userID"], total["reviewerID"]))
        data, image_feature, text_feature = amazon(train,test)
        image_shape = 4096
        text_shape = 512

    elif args.data == "movie":
        data = pd.read_feather("/daintlab/data/movielens/movie_3953.ftr")
        image_feature = pd.read_pickle('/daintlab/data/movielens/image_feature_vec.pickle')
        text_feature = pd.read_pickle('/daintlab/data/movielens/text_feature_vec.pickle')
        data["test_positive"] = data["test_positive"].apply(lambda x : [x])
        num_user = 6041
        num_item = 3953
        image_shape = 512
        text_shape = 300


    
    else:
        print("데이터가 존재하지 않습니다.")
        return 0
    
    MD = Make_Dataset(ratings = data)
    user, item, rating = MD.trainset
    evaluate_data = MD.evaluate_data


    #NCF model
    # feature X
    if args.feature == False:
        print("FEATURE X")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data)
    
    # feature O
    else:
        print("FEATURE O")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data,image=image_shape,text=text_shape)
    
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
        sample = SampleGenerator(user = user, item = item, 
                                 rating = rating, ratings = data, 
                                 positive_len = MD.positive_len, num_neg = args.num_neg)
        train_loader = sample.instance_a_train_loader(args.batch_size)
        
        print("Train Loader 생성 완료")
        for batch_id, batch in enumerate(train_loader):
            users, items, ratings = batch[0], batch[1], batch[2]
            ratings = ratings.float()
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            optim.zero_grad() 
            
            if args.feature == True:
                
                image = []
                text = []
                if args.data == "amazon":
                    for i in items :
                        image_vector = item2image[i.item()]
                        image.append(image_feature[image_vector])  
                    for u in users :
                        text_vector = user2review[u.item()]
                        text.append(text_feature.infer_vector([text_vector]))
                else:
                    for i in items :
                        image.append(image_feature[i.item()]) 
                        text.append(text_feature[i.item()])                                  
                image= torch.FloatTensor(image)
                image = image.cuda()
                text= torch.FloatTensor(text)
                text = text.cuda()
            
                output = model(users,items,image=image,text=text,data=args.data)
            
            
            else:                           
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
            hit_ratio,ndcg = engine.evaluate(model,evaluate_data,epoch_id=epoch,
                                            feature=args.feature,image=item2image,text=user2review,
                                            image_feature=image_feature,text_feature=text_feature,
                                           data=args.data)
        else:
            hit_ratio,ndcg = engine.evaluate(model,evaluate_data,
                                             epoch_id=epoch,
                                             image_feature=image_feature,
                                             text_feature=text_feature,
                                             feature=args.feature,
                                             data=args.data)
        wandb.log({"epoch" : epoch,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        N.append(ndcg)

if __name__ == '__main__':
    main()
        