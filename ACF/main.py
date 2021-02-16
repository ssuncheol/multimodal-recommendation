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
import warnings
warnings.filterwarnings("ignore")



def main():
    wandb.init(project="ACF")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                type=str,
                default='/daintlab/home/yeeun0501/Jaecheol/Multimodal-Rec/Attentive_Collaborative_Filtering/data/Movielens-raw',
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
                default=20,
                help='epoch')
    parser.add_argument('--batch_size',
                type=int,
                default=256,
                help='batch size')
    parser.add_argument('--dim',
                type=int,
                default=64,
                help='dimension')    
    parser.add_argument('--lr',
                type=float,
                default=0.01,
                help='learning rate')    
    parser.add_argument('--reg',
                type=float,
                default=0.01,
                help='l2_regularization')
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')
    parser.add_argument('--num_sam',
                type=int,
                default=1,
                help='num of pos sample')

    parser.add_argument('--feature_type', default='all', type=str,
                        help='Type of feature to use. [all, img, txt]')
    parser.add_argument('--eval_type', default='leave-one-out', type=str,
                        help='Evaluation protocol. [ratio-split, leave-one-out]')

    global args
    args = parser.parse_args()
    wandb.config.update(args)

    args = parser.parse_args()
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path,args.eval_type)
        
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, images = D.load_data(data_path, args.feature_type)
    #import pdb; pdb.set_trace()
    train_dataset = D.CustomDataset(train_df, test_df, images, negative=train_ng_pool, istrain=True, feature_type=args.feature_type, num_sam=args.num_sam)
    test_dataset = D.CustomDataset(train_df, test_df, images, negative=test_negative, istrain=False, feature_type=args.feature_type, num_sam=args.num_sam)
  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=my_collate_trn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                              collate_fn=my_collate_tst, pin_memory =True)
    
    # Model
    acf = ACF(num_user, num_item, images, args.dim)
    acf = torch.nn.DataParallel(acf)
    acf = acf.cuda()
    print(acf)

    # Optimizer
    optim = optimizer(optim=args.optim, lr=args.lr, model=acf)

    # Train & Eval
    for epoch in range(args.epochs):
        start = time.time()
        train(acf, train_loader, epoch,optim)
        end = time.time()
        print("{}/{} Train Time : {}".format(epoch+1,args.epochs,end-start))
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
        users, item_p, item_n,positives,img_p = users.cuda(), item_p.cuda(), item_n.cuda(), positives.cuda(), img_p.cuda()
        print("user shape", users.shape)
        print("item_p shape", item_p.shape)
        print("item_n shape", item_n.shape)
        print("positives shape", positives.shape)
        print("image shape", img_p.shape)
        score_j, score_k = model(users,item_p,item_n,positives,img_p)
        loss = my_loss(score_j,score_k)

        optim.zero_grad()
        loss.backward()
        optim.step()
        e = time.time()
        loss = loss.item()
        wandb.log({'Batch Loss': loss})
        print("{} iter loss : {} time : {}".format(i,loss,e-s))
    


def test(model, test_loader, epoch):
    model.eval()
    hr1 = []
    hr2 = []
    ndcg = []
    for i, (users, test_positive, test_negative, positives, img_p) in enumerate(test_loader):
        with torch.no_grad():
     
            test_index = test_positive.numpy().reshape(-1)
            test_negative_index = test_negative.numpy().reshape(-1)
            test_positive = test_positive.view(-1)
   
            users, test_positive, test_negative ,positives,img_p = users.cuda(), test_positive.cuda(), test_negative.cuda(), positives.cuda(), img_p.cuda()
            
            score_j, _ = model(users,test_positive,test_positive,positives,img_p)
            _, score_k = model(users,test_negative,test_negative,positives,img_p)

            positive_score = pd.Series(score_j.detach().cpu().numpy(),index = test_index)
            negative_score = pd.Series(score_k.detach().cpu().numpy(),index = test_negative_index)
            test_score = pd.concat([positive_score,negative_score])
            test_score = test_score.argsort()[:10]
            performance = get_performance(gt_item=test_index.tolist(),recommends=test_score.tolist())
            hr1.append(performance[0])
            hr2.append(performance[1])
            ndcg.append(performance[2])

    print("hr1 = {}, hr2 = {}, ndcg = {}".format(np.mean(hr1),np.mean(hr2),np.mean(ndcg)))
    wandb.log({"epoch" : epoch,
            "HR" : np.mean(hr1),
            "HR2" : np.mean(hr2),
            "NDCG" : np.mean(ndcg)})

def my_collate_trn(batch):
    c = 1
    user = [torch.LongTensor(item[0]) for item in batch]
    user = torch.cat(user)
    item_p = [item[1] for item in batch]
    item_p = torch.LongTensor(item_p)
    item_n = [item[2] for item in batch]
    item_n = torch.LongTensor(item_n).view(-1)
    np.random.seed(c)
    pos_set = [torch.LongTensor(item[3][np.random.choice(len(item[3]), args.num_sam)]) for item in batch]
    pos_set = torch.cat(pos_set)
  
    np.random.seed(c)
    img_p = [torch.FloatTensor(item[4][np.random.choice(len(item[4]), args.num_sam)]) for item in batch]
    img_p = torch.cat(img_p)
    
    c+=1

    return [user, item_p, item_n, pos_set, img_p]

def my_collate_tst(batch):
    count = 1
    user = [torch.LongTensor(item[0]) for item in batch]
    user = torch.cat(user)
    item_p = [item[1] for item in batch]
    item_p = torch.LongTensor(item_p)
    item_n = [item[2] for item in batch]
    item_n = torch.LongTensor(item_n).view(-1)
    np.random.seed(count)
    pos_set = [torch.LongTensor(item[3][np.random.choice(len(item[3]), args.num_sam)]) for item in batch]
    pos_set = torch.cat(pos_set)
  
    np.random.seed(count)
    img_p = [torch.FloatTensor(item[4][np.random.choice(len(item[4]), args.num_sam)]) for item in batch]
    img_p = torch.cat(img_p)

    count +=1
    return [user, item_p, item_n, pos_set, img_p]
    
    
    
  


if __name__ == '__main__':
    main()
        