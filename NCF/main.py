import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import random
from dataloader import Make_Dataset, SampleGenerator
from utils import optimizer
from model import NeuralCF
from evaluate import Engine
from metrics import MetronAtK
import wandb
import warnings
from sklearn.preprocessing import LabelEncoder 
warnings.filterwarnings("ignore")

def main():
    wandb.init(project="Multimodal")
    parser = argparse.ArgumentParser()
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
                default=20,
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

    data = pd.read_feather("/daintlab/data/movielens/movie_3953.ftr")
    data2 = pd.read_csv("/daintlab/data/movielens/movies.csv", header=None)
    
    image_feature = pd.read_pickle('/daintlab/data/movielens/image_feature_vec.pickle')
    text_feature = pd.read_pickle('/daintlab/data/movielens/text_feature_vec.pickle')
    
    
    
    
    subdata = data2.iloc[:, [0, 1, 7, 8]]
    subdata.columns = ['ID', 'REIndex', 'Genre', 'Director']

    Genre = subdata['Genre']
    
    G = []
    for i in list(Genre):
        try:
            G.extend(i.split(', '))
        except:
            print(i)
    df = pd.DataFrame(G)
    Genre = df[0].unique()
    
    
    
    genre_dic = {}
    one_hot_vector = {}
    for i, j in zip(list(subdata['Genre']), list(subdata['ID'])):
        for genre_name in Genre:
            genre_dic[genre_name] = 0
        
        try:
            genre_list_of_item = i.split(', ')
            for k in genre_list_of_item:
                genre_dic[k] += 1
            v = list(genre_dic.values())
            one_hot_vector[j] = v
            
        except:
            print(j)
            one_hot_vector[j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            
    #print(one_hot_vector)

    data1 = data2.iloc[:, [0, 1, 7, 8]]
    data1.columns = ['ID', 'index', 'Genre', 'Director']
    
    labelencoder =LabelEncoder()
    data1['Director_num'] = labelencoder.fit_transform(data1['Director'].astype(str))
    
    data1['Director_num'] = data1['Director_num'].apply(lambda x:x+1)
    
    print(np.min(data1['Director_num']))
    print(np.max(data1['Director_num']))
    
    dic_director = {}
    for i in range(len(data1['ID'])):
        dic_director[data1['ID'][i]] = data1['Director_num'][i]
    

      
    MD = Make_Dataset(ratings = data)
    user, item, rating = MD.trainset
    evaluate_data = MD.evaluate_data


    #NCF model
    model = NeuralCF(num_users= 6041,num_items = 3953, num_director=1918,num_genre=23,image=512,text=300,
                     embedding_size = args.latent_dim_mf,
                     num_layers = args.num_layers)
    
    #model = nn.DataParallel(model)
    model = model.cuda()
    
    print(model)
    optim = optimizer(optim=args.optim, lr=args.lr, model=model, weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    wandb.watch(model)


    N = []
    patience = 0
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
            director = []
            genre=[]
            image = []
            text = []
            for i in items :
                director.append(dic_director[i.item()])
                genre.append(one_hot_vector[i.item()])
                image.append(image_feature[i.item()]) 
                text.append(text_feature[i.item()])      
                   
            ratings = ratings.float()
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            
            optim.zero_grad() 
            director = torch.LongTensor(director)
            director = director.cuda()
            genre = torch.LongTensor(genre)
            genre = genre.cuda()
            image= torch.FloatTensor(image)
            image = image.cuda()
            text= torch.FloatTensor(text)
            text = text.cuda()
            
            output = model(users, items, director,genre,image,text)
            
            
            loss = criterion(output, ratings)
            loss.backward()
            optim.step()
            loss = loss.item()
            wandb.log({'Batch Loss': loss})
            total_loss += loss

        t2 = time.time()
        print("train : ", t2 - t1) 
 
        engine = Engine()
        hit_ratio,ndcg = engine.evaluate(model,evaluate_data,dic_director,one_hot_vector,image_feature,text_feature,epoch_id=epoch)
        wandb.log({"epoch" : epoch,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        N.append(ndcg)

        if N[-1] < max(N):
            if patience == 5:
                print("Patience = ")
                print("ndcg = {:.4f}".format(max(N)))
                break
            else:
                patience += 1
                print("Patience = {} ndcg = {:.4f}".format(patience, max(N)))
        else:
            patience = 0
            print("Patience = {}".format(patience))

    
if __name__ == '__main__':
    file_name = "/daintlab/data/movielens/movie_3953.ftr"
    if os.path.exists(file_name):
        print("Data 존재")
        main()
    else:
        print("데이터 없음")
        