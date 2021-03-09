#from comet_ml import Experiment
import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from utils import optimizer
from model import ACF
import dataset as D
from metric import get_performance
import wandb
import random
import warnings
warnings.filterwarnings("ignore")



def main():
    wandb.init(project="AttCF")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                type=str,
                default='/daintlab/data/recommend/Amazon-office-raw',
                help='path')
    parser.add_argument('--top_k',
                type=int,
                default=10,
                help='top_k')
    parser.add_argument('--optim',
                type=str,
                default='adam',
                help='optimizer')
    parser.add_argument('--epochs',
                type=int,
                default=5,
                help='epoch')
    parser.add_argument('--batch_size',
                type=int,
                default=256,
                help='batch size')
    parser.add_argument('--dim',
                type=int,
                default=128,
                help='dimension')    
    parser.add_argument('--lr',
                type=float,
                default=0.001,
                help='learning rate')    
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')
    parser.add_argument('--num_sam',
                type=int,
                default=4,
                help='num of pos sample')

    parser.add_argument('--feature_type',
                        default='all', 
                        type=str,
                        help='Type of feature to use. [all, img, txt]')
    parser.add_argument('--eval_type', 
                        default='leave-one-out', 
                        type=str,
                        help='Evaluation protocol. [ratio-split, leave-one-out]')

    global args
    global sd
    global train_len
    global test_len
    
    args = parser.parse_args()
    wandb.config.update(args)

    args = parser.parse_args()
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path,args.eval_type)
        
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, images = D.load_data(data_path, args.feature_type)
    train_len = len(train_df)
    test_len = num_user
    
    train_dataset = D.CustomDataset(train_df, test_df, images, negative=train_ng_pool, istrain=True, feature_type=args.feature_type, num_sam=args.num_sam)
    test_dataset = D.CustomDataset(train_df, test_df, images, negative=test_negative, istrain=False, feature_type=args.feature_type, num_sam=args.num_sam)
  
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=my_collate,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=my_collate_tst,pin_memory =True)
    
    # Model
    acf = ACF(num_user, num_item, images, args.dim)
    acf = torch.nn.DataParallel(acf)
    acf = acf.cuda()
    print(acf)

    # Optimizer
    optim = optimizer(optim=args.optim, lr=args.lr, model=acf)

    # Train & Eval
    
    for epoch in range(args.epochs):
        sd = np.random.randint(2021)
        start = time.time()
        train(acf, train_loader, epoch,optim)
        end = time.time()
        print("{}/{} Train Time : {}".format(epoch+1,args.epochs,end-start))
        if (epoch+1) == args.epochs:
            start = time.time()
            test(acf, test_loader, epoch)
            end = time.time()
            print("{}/{} Evaluate Time : {}".format(epoch+1,args.epochs,end-start))
        

def my_loss(pos, neg):
    cus_loss = - torch.sum(torch.log(torch.sigmoid(pos - neg) + 1e-10))
    return cus_loss

def train(model, train_loader, epoch, optim):
    model.train()
    
    for i, (users, item_p, item_n, positives, img_p) in enumerate(train_loader):
        s = time.time()
        print("user : ",users.shape)
        print("pos : ",item_p.shape)
        print("neg : ",item_n.shape)
        print("poss : ",positives.shape)
        users, item_p, item_n,positives,img_p = users.cuda(), item_p.cuda(), item_n.cuda(), positives.cuda(), img_p.cuda()
        score_j, score_k = model(users,item_p,item_n,positives,img_p,args.num_sam)
        loss = my_loss(score_j,score_k)

        optim.zero_grad()
        loss.backward()
        optim.step()
        wandb.log({'Batch Loss': loss})
        e = time.time()
        print("{}/{} iter loss : {} time : {}".format(i,round(train_len/args.batch_size),loss,e-s))
        


def test(model, test_loader, epoch):
    model.eval()
    hr1 = []
    hr2 = []
    ndcg = []
    for i, (test_users,  test_negative, positiveset, test_positiveset, test_img_p) in enumerate(test_loader):
        with torch.no_grad():
            sss = time.time()
            pos_len = len(positiveset[0])
            print("User Index : ",test_users)
            #print("Positive : ",test_positive)
            print("Positiveset : ",positiveset.shape)
            print("Test_Positiveset : ",test_positiveset.shape)
            print("Negative : ",test_negative.shape)
            
            #test_index = test_positive.numpy().reshape(-1)
            test_index = test_positiveset.numpy().reshape(-1)
            test_negative_index = test_negative.numpy().reshape(-1)
            test_users,  test_negative ,positiveset,test_img_p = test_users.cuda(),  test_negative.cuda(), positiveset.cuda(), test_img_p.cuda()
            pos_score = []
            for p in range(len(test_index)):
                score_p, _ = model(test_users.view(-1),test_positiveset[:,p].view(-1),test_positiveset[:,p].view(-1),positiveset,test_img_p,pos_len)
                score_p = score_p.detach().cpu().numpy().tolist()
                pos_score.extend(score_p)
            neg_score = []
            for n in range(len(test_negative_index)):
                score_n, _ = model(test_users.view(-1),test_negative[:,n].view(-1),test_negative[:,n].view(-1),positiveset,test_img_p,pos_len)
                score_n = score_n.detach().cpu().numpy().tolist()
                neg_score.extend(score_n)
            positive_score = pd.Series(pos_score,index = test_index)
            negative_score = pd.Series(neg_score,index = test_negative_index)
            print("Test Score : ",positive_score)
            print("Neg Score : ",negative_score)
            test_score = pd.concat([positive_score,negative_score])
            test_score = test_score.sort_values(ascending=False)[:args.top_k]
            performance = get_performance(gt_item=test_index.tolist(),recommends=test_score.index.tolist())
            hr1.append(performance[0])
            print("hr1 : ",performance[0])
            hr2.append(performance[1])
            print("hr2 : ",performance[1])
            ndcg.append(performance[2])
            print("ndcg : ",performance[2])
            eee = time.time()
            print("{}/{}번째 Time : {}".format(i,test_len,eee-sss))
            

    print("hr1 = {}, hr2 = {}, ndcg = {}".format(np.mean(hr1),np.mean(hr2),np.mean(ndcg)))
    wandb.log({"epoch" : epoch,
            "HR" : np.mean(hr1),
            "HR2" : np.mean(hr2),
            "NDCG" : np.mean(ndcg)})

def my_collate(batch):
    ss = time.time()
    user = [item[0] for item in batch]
    user = torch.LongTensor(user).view(-1,1)
    item_p = [item[1] for item in batch]
    item_p = torch.LongTensor(item_p).view(-1,1)
    item_n = [item[2] for item in batch]
    item_n = torch.LongTensor(item_n).view(-1,1)
    
    random.seed(sd)
    print(sd)
    pos_set = np.array([random.sample(set(item[3]), args.num_sam) for item in batch])
    pos_set = torch.LongTensor(pos_set)
    print("PS : ",pos_set)
  
    random.seed(sd)
    print(sd)
    img_p = [random.sample(set(item[4]), args.num_sam) for item in batch]
    img_p = [torch.cat(item) for item in img_p]
    img_p = [torch.unsqueeze(item,0) for item in img_p]
    img_p = torch.cat(img_p)    
  
    ee = time.time()
    #print("Collate Time : {}".format(ee-ss))
    
    return [user, item_p, item_n, pos_set, img_p]

def my_collate_tst(batch):
    ss = time.time()
    user = [item[0] for item in batch]
    user = torch.LongTensor(user)
    #item_p = [item[1] for item in batch]
    #item_p = torch.LongTensor(item_p)
    neg_set = [item[1] for item in batch]
    neg_set = torch.LongTensor(neg_set)
    pos_set = torch.LongTensor([item[2] for item in batch])
    test_pos_set = torch.LongTensor([item[3] for item in batch])
    img_p = [item[4] for item in batch]
    img_p = torch.cat(img_p)    
    
    ee = time.time()
    #print("Collate Time : {}".format(ee-ss))
    #import pdb;pdb.set_trace()
    return [user, neg_set, pos_set, test_pos_set, img_p]

if __name__ == '__main__':
    main()
        