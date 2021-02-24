import os
import json
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
from model import NeuralCF
from evaluate import Engine
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def my_collate_trn(batch):
    user = [element for items in batch for element in items[0]]
    user = torch.LongTensor(user)
    item = [element for items in batch for element in items[1]]
    item = torch.LongTensor(item)
    rating = [element for items in batch for element in items[2]]
    rating = torch.FloatTensor(rating)
    ## feature가 1개일 때.
    if len(batch[0]) == 4:
        feature = [element for items in batch for element in items[3]]
        if type(feature[0]) == type(torch.Tensor([])):
            feature = torch.stack(feature)
        else:
            feature = torch.Tensor(feature)
        return [user, item, rating, feature]
    ## feature가 2개일 때.
    if len(batch[0]) == 5:
        image = [element for items in batch for element in items[3]]
        if type(image[0]) == type(torch.Tensor([])):
            image = torch.stack(image)
        else:
            image = torch.Tensor(image)
        text = [element for items in batch for element in items[4]]
        text = torch.Tensor(text)
        return [user, item, rating, image, text]
    return [user, item, rating]

# model 저장 함수
def save(ckpt_dir, net, optim, epoch, image_type):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
              '%s/model_epoch%d_%s.pth' % (ckpt_dir, epoch, image_type))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                type=str,
                default='/daintlab/data/recommend/Amazon-office-raw',
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
                default=30,
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
    parser.add_argument('--extractor_path',
                type=str,
                default='/daintlab/data/recommend/Amazon-office-raw/resnet18.pth',
                help='path of feature extractor(pretrained model)')
    parser.add_argument('--amp',
                type=bool,
                default=True,
                help='using amp(Automatic mixed-precision)')
    args = parser.parse_args()
    return args
def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    experiment = Experiment(api_key="Bc3OhH0UQZebqFKyM77eLZnAm", project_name='dp final')
    experiment.log_parameters(args)
    
    # data load 
    df_train_p = pd.read_feather("%s/%s/train_positive.ftr" % (args.path, args.eval))
    df_train_n = pd.read_feather("%s/%s/train_negative.ftr" % (args.path, args.eval))
    df_test_p = pd.read_feather("%s/%s/test_positive.ftr" % (args.path, args.eval))
    df_test_n = pd.read_feather("%s/%s/test_negative.ftr" % (args.path, args.eval))
    user_index_info = pd.read_csv("%s/index-info/user_index.csv" % args.path)
    item_index_info = pd.read_csv("%s/index-info/item_index.csv" % args.path)
    meta_data = json.load(open(os.path.join('%s/item_meta.json' % args.path), 'rb'))

    user_index_dict = {}
    item_index_dict = {}
    
    img_dict = None
    txt_dict = None
    
    image_shape = None
    text_shape = None
    # image 쓸 건가
    if args.image:
        img_dict = {}
        image_shape = 512
        # raw image를 쓸 것인지, 전처리 해놓은 feature vector를 쓸 지.
        if args.feature == 'raw':
            id_list = item_index_info["itemid"].tolist()
            transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.5,), (0.5,))])
            for i in range(len(id_list)):
                img_dict[i] = transform(Image.open(os.path.join('%s' % args.path, '%s' % meta_data[str(id_list[i])]['image_path'])).convert('RGB'))
        else:
            img_feature = pd.read_pickle('%s/image_feature_vec.pickle' % args.path)
            for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
                item_index_dict[i] = j
            for i in item_index_dict.keys():
                img_dict[i] = img_feature[item_index_dict[i]] 
    # text 쓸 건가
    if args.text:
        txt_dict = {}
        text_shape = 300
        txt_feature = pd.read_pickle('%s/text_feature_vec.pickle' % args.path)
        for i, j in zip(item_index_info['itemidx'], item_index_info['itemid']):
            item_index_dict[i] = j
        for i in item_index_dict.keys():
            txt_dict[i] = txt_feature[item_index_dict[i]]
      
    num_user = df_train_p['userid'].nunique()
    num_item = item_index_info.shape[0]
    
    print('num of user ', num_user)
    print('num of item ', num_item)

    # data 전처리
    dt = time.time()
    MD = Make_Dataset(df_train_p, df_train_n, df_test_p, df_test_n)
    train_u, train_i, train_r, neg_pool_dict = MD.train_data
    test_u, test_i, item_num_dict, test_pos_item_num = MD.evaluate_data
    print('데이터 전처리', (time.time() - dt))
    
    #NCF model
    model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.latent_dim_mf, dropout=args.drop_rate,
                        num_layers=args.num_layers, feature=args.feature, image=image_shape, text=text_shape, extractor_path=args.extractor_path)
    model = nn.DataParallel(model)
    model = model.cuda()
    print('model 생성 완료.')

    optim = optimizer(optim=args.optim, lr=args.lr, model=model, weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss().cuda()
    
    # amp
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # train, test loader 생성
    train_dataset = UserItemTrainDataset(train_u, train_i, train_r, neg_pool_dict, args.num_neg, image=img_dict, text=txt_dict)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_trn, pin_memory =True)
    test_dataset = UserItemtestDataset(test_u, test_i, image=img_dict, text=txt_dict)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 5, shuffle=False, num_workers=4)
    
    print('dataloader 생성 완료.')
    # train 및 eval 시작
    for epoch in range(args.epochs):
        with experiment.train():
            print('Epoch {} starts !'.format(epoch+1))
            print('-' * 80)
            model.train()
            total_loss = 0
            t1 = time.time()
            for batch_id, batch in enumerate(train_loader):
                # print("Train Loader 생성 완료 %.5f" % (time.time() - t1))
                optim.zero_grad()
                if (args.image) & (args.text):
                    users, items, ratings, image, text = batch[0], batch[1], batch[2], batch[3], batch[4]             
                    users, items, ratings, image, text = users.cuda(), items.cuda(), ratings.cuda(), image.cuda(), text.cuda()    
                elif args.image: 
                    users, items, ratings, image = batch[0], batch[1], batch[2], batch[3]                  
                    users, items, ratings, image = users.cuda(), items.cuda(), ratings.cuda(), image.cuda()
                    text = None
                elif args.text:                   
                    users, items, ratings, text = batch[0], batch[1], batch[2], batch[3]
                    users, items, ratings, text = users.cuda(), items.cuda(), ratings.cuda(), text.cuda()
                    image = None
                else :                   
                    users, items, ratings = batch[0], batch[1], batch[2]
                    users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                    image = None
                    text = None
        
                if args.amp:  
                    with torch.cuda.amp.autocast():
                        output = model(users, items, image=image, text=text)
                        loss = criterion(output, ratings)  
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    output = model(users, items, image=image, text=text)
                    loss = criterion(output, ratings)
                    loss.backward()
                    optim.step()
                
            t2 = time.time()
            print("train : ", t2 - t1) 
        if (epoch + 1) % args.interval == 0:
            with experiment.test():
                engine = Engine(args.top_k, item_num_dict, test_pos_item_num, num_item, num_user)
                t3 = time.time()
                hit_ratio, hit_ratio2, ndcg = engine.evaluate(model, test_loader, epoch_id=epoch, image=img_dict, text=txt_dict, eval=args.eval)
                t4 = time.time()
                print('test:', t4 - t3) 
            
                # ckpt_dir = '%s/ckpt_dir' % args.path
                # save(ckpt_dir, model, optim, args.interval, args.feature)
                experiment.log_metrics({"epoch" : epoch,
                                "HR" : hit_ratio,
                                "HR2" : hit_ratio2,
                                "NDCG" : ndcg}, epoch=(epoch+1))
    experiment.end()

if __name__ == '__main__':
    args = get_args()
    main(args)
        
