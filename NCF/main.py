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

# amazon preprocessing!
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
    image_feature = np.load("/daintlab/home/tmddnjs3467/workspace/amazon_office/image_feature.npy", allow_pickle=True)
    image_feature = image_feature.item()
    text_feature = Doc2Vec.load("/daintlab/home/tmddnjs3467/workspace/amazon_office/doc2vecFile")
    return data, image_feature, text_feature  
     

def main():
    wandb.init(project="Real Total NCF")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="amazon",
                help='dataset')
    #parser.add_argument('--feature',
    #            type=bool,
    #            default=False,
    #            help='feature')
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
    

    # Select data 

    if args.data == "amazon":

        train = pd.read_csv("/daintlab/home/tmddnjs3467/workspace/amazon_office/train.csv")
        test = pd.read_csv("/daintlab/home/tmddnjs3467/workspace/amazon_office/test.csv")
        total = pd.concat([train,test])
        num_user = 4874
        num_item = 2406
        image1_feature = dict(zip(total["itemID"], total["asin"]))
        text1_feature = dict(zip(total["userID"], total["reviewerID"]))
        data, image_feature, text_feature = amazon(train,test)
        
        img_dict = {}
        txt_dict = {}
        for i in list(image1_feature.keys()):
            img_dict[i] = image_feature[image1_feature[i]]
        for j in list(text1_feature.keys()):
            txt_dict[j] = text_feature.infer_vector([text1_feature[j]])
        import pdb; pdb.set_trace()
        #print(txt_dict)
        image_shape = 4096
        text_shape = 512
        
    elif args.data == "movie":

        data = pd.read_feather("/daintlab/home/tmddnjs3467/workspace/movielense/movie_3953.ftr")
        img_dict = pd.read_pickle('/daintlab/home/tmddnjs3467/workspace/movielense/image_feature_vec.pickle')
        txt_dict = pd.read_pickle('/daintlab/home/tmddnjs3467/workspace/movielense/text_feature_vec.pickle')
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
    test_user,test_item,test_negative_user,test_negative_item = MD.evaluate_data
    
    #NCF model
    if (args.image == True) & (args.text == True):
        print("IMAGE TEXT MODEL")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data,image=image_shape,text=text_shape)    
    
    elif args.image == True:
        print("IMAGE MODEL")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data,image=image_shape)  
    
    elif args.text == True:
        print("TEXT MODEL")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data,text=text_shape)  

    else:
        print("MODEL")
        model = NeuralCF(num_users=num_user,num_items=num_item,
                        embedding_size=args.latent_dim_mf,
                        num_layers=args.num_layers,data=args.data)
    
    
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
        if (args.image == True) & (args.text == True):               
            train_loader = sample.instance_a_train_loader(args.batch_size,image=img_dict,text=txt_dict)
        elif args.image == True:              
            train_loader = sample.instance_a_train_loader(args.batch_size,image=img_dict)
        elif args.text == True:           
            train_loader = sample.instance_a_train_loader(args.batch_size,text=txt_dict)
        else :                
            train_loader = sample.instance_a_train_loader(args.batch_size)
            
            
        
        
        print("Train Loader 생성 완료")
        for batch_id, batch in enumerate(train_loader):
            #import pdb; pdb.set_trace()
            optim.zero_grad()
            if (args.image == True) & (args.text == True):
                users, items, ratings, image, text = batch[0], batch[1], batch[2], batch[3], batch[4]             
                users, items, ratings, image, text = users.cuda(), items.cuda(), ratings.cuda(), image.cuda(), text.cuda()
                output = model(users,items,image=image, text=text)
            elif args.image == True: 
                users, items, ratings, image = batch[0], batch[1], batch[2], batch[3]                  
                users, items, ratings, image = users.cuda(), items.cuda(), ratings.cuda(), image.cuda()
                output = model(users,items,image=image)
            elif args.text == True:                   
                users, items, ratings, text = batch[0], batch[1], batch[2], batch[3]
                users, items, ratings, text = users.cuda(), items.cuda(), ratings.cuda(), text.cuda()
                output = model(users,items,text=text)
            else :                   
                users, items, ratings = batch[0], batch[1], batch[2]
                users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                output = model(users, items)
             
            
            #if (args.image == True) & (args.text == True): 
            #    output = model(users,items,image=image, text=text)
            #elif args.image == True:
            #    output = model(users,items,image=image)
            #elif args.text == True:
            #    output = model(users,items,text=text)
            #else:                           
            #    output = model(users, items)
                #print(output)
            loss = criterion(output, ratings)
            loss.backward()
            optim.step()
            loss = loss.item()
            wandb.log({'Batch Loss': loss})
            total_loss += loss

        t2 = time.time()
        print("train : ", t2 - t1) 


        """         if (args.image == True) & (args.text == True):               
            train_loader = sample.instance_a_train_loader(args.batch_size,image=img_dict,text=txt_dict)
        elif args.image == True:              
            train_loader = sample.instance_a_train_loader(args.batch_size,image=img_dict)
        elif args.text == True:           
            train_loader = sample.instance_a_train_loader(args.batch_size,text=txt_dict)
        else :                
            train_loader = sample.instance_a_train_loader(args.batch_size)
            """
        engine = Engine()
        if args.data == "amazon":
            #if args.feature == True:
            a=time.time() 
            evaluate_data = testGenerator(test_user,test_item)
            evaluate_data_neg = testGenerator(test_negative_user,test_negative_item)
            if (args.image == True) & (args.text == True):
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,image=img_dict,text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,image=img_dict,text=txt_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,image=img_dict,text=txt_dict,data=args.data)
            elif args.image == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,image=img_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,image=img_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,image=img_dict,data=args.data)
            elif args.text == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,text=txt_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,text=txt_dict,data=args.data)                
            


            else:
                a=time.time() 
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,data=args.data)  
            b=time.time()
            print('test:' ,b-a) 
                

        else:
            a=time.time() 
            evaluate_data = testGenerator(test_user,test_item)
            evaluate_data_neg = testGenerator(test_negative_user,test_negative_item)
            if (args.image == True) & (args.text == True):
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,image=img_dict,text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,image=img_dict,text=txt_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,image=img_dict,text=txt_dict,data=args.data)
            elif args.image == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,image=img_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,image=img_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,image=img_dict,data=args.data)
            elif args.text == True:
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100,text=txt_dict)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100,text=txt_dict) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,text=txt_dict,data=args.data)                
            else:
                a=time.time() 
                test_loader = evaluate_data.instance_a_test_loader(len(test_user)//100)
                test_negative_loader = evaluate_data_neg.instance_a_test_loader(len(test_negative_user)//100) 
                hit_ratio,ndcg = engine.evaluate(model,test_loader,test_negative_loader,epoch_id=epoch,data=args.data)  
            b=time.time()
            print('test:' ,b-a) 
         
        wandb.log({"epoch" : epoch,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        N.append(ndcg)



if __name__ == '__main__':
    main()
        