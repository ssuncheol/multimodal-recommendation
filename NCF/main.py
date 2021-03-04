from comet_ml import Experiment
import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
from utils import Logger, AverageMeter, str2bool
from model import MAML, NeuralCF
from loss import Embedding_loss, Feature_loss, Covariance_loss
import dataset as D
from metric import get_performance
import resnet_tv as resnet
import torch.distributed as dist
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MAML', type=str,
                    help='model type')
parser.add_argument('--save_path', default='./result', type=str,
                    help='savepath')
parser.add_argument('--batch_size', default=512, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='train epoch')
parser.add_argument('--data_path', default='/daintlab/data/recommend/Movielens-raw', type=str,
                    help='Path to rating data')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of used layers in NCF')
parser.add_argument('--embed_dim', default=64, type=int,
                    help='Embedding Dimension')
parser.add_argument('--dropout_rate', default=0.2, type=float,
                    help='Dropout rate')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--margin', default=1.0, type=float,
                    help='Margin for embedding loss')
parser.add_argument('--feat_weight', default=0, type=float,
                    help='Weight of feature loss')
parser.add_argument('--cov_weight', default=0, type=float,
                    help='Weight of covariance loss')
parser.add_argument('--top_k', default=10, type=int,
                    help='Top k Recommendation')
parser.add_argument('--num_neg', default=4, type=int,
                    help='Number of negative samples for training')
parser.add_argument('--load_path', default=None, type=str,
                    help='Path to saved model')
parser.add_argument('--eval_freq', default=50, type=int,
                    help='evaluate performance every n epoch')
parser.add_argument('--feature_type', default='img', type=str,
                    help='Type of feature to use. [all, img, txt]')
parser.add_argument('--feature_data_type', default='raw', type=str,
                    help='raw(png) or pre(vector)')                    
parser.add_argument('--eval_type', default='ratio-split', type=str,
                    help='Evaluation protocol. [ratio-split, leave-one-out]')
parser.add_argument('--cnn_path', default='./resnet18.pth', type=str,
                    help='Path to feature data')
parser.add_argument('--ddp_port', default='22222', type=str,
                    help='DDP Port')
parser.add_argument('--ddp_addr', default='127.0.0.1', type=str,
                    help='DDP Address')
args = parser.parse_args()


def main(rank, args):
    # Initialize Each Process
    init_process(rank, args.world_size)
    
    # Set save path
    save_path = args.save_path
    if not os.path.exists(save_path) and dist.get_rank() == 0:
        os.makedirs(save_path)
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if dist.get_rank() == 0:
        experiment = Experiment(project_name='recommend', disabled=True)
        experiment.log_parameters(args)
    else:
        experiment = Experiment(disabled=True)

    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path, args.eval_type)
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, text_feature, images, test_pos_item_num, item_num_dict = D.load_data(
        data_path, args.feature_type) 
    train_dataset = D.CustomDataset(args.model, train_df, text_feature, images, negative=train_ng_pool, num_neg=args.num_neg,
                                    istrain=True, feature_type=args.feature_type)
    test_dataset = D.CustomDataset(args.model, test_df, text_feature, images, negative=test_negative, num_neg=None,
                                   istrain=False, feature_type=args.feature_type)

    args.batch_size = int(args.batch_size / args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                              collate_fn=my_collate_trn, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=4,
                             collate_fn=my_collate_tst, pin_memory=True)

    # Model
    t_feature_dim = text_feature[0].shape[-1]
    if args.model == 'MAML':
        model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate, args.feature_type, t_feature_dim,
                    args.cnn_path, rank).cuda(rank)
    else:
        model = NeuralCF(num_users=num_user, num_items=num_item, 
                        embedding_size=args.embed_dim, dropout=args.dropout_rate,
                        num_layers=args.num_layers, feature_data_type=args.feature_data_type, feature_type=args.feature_type, text=t_feature_dim, 
                        extractor_path=args.cnn_path, rank=rank).cuda(rank)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained Model Loaded")

    # Optimizer
    if args.model == "MAML":
        if args.feature_type != "rating":
            optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                        {'params': model.module.embedding_item.parameters()},
                                        {'params': model.module.feature_fusion.parameters()},
                                        {'params': model.module.attention.parameters(), 'weight_decay': 100.0}],
                                        lr=args.lr)
        else:
            optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                        {'params': model.module.embedding_item.parameters()},
                                        {'params': model.module.attention.parameters(), 'weight_decay': 100.0}],
                                        lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
       
    scaler = torch.cuda.amp.GradScaler()
    
    # Loss
    if args.model == "MAML":
        embedding_loss = Embedding_loss(margin=args.margin, num_item=num_item).cuda(rank)
        feature_loss = Feature_loss().cuda(rank)
        covariance_loss = Covariance_loss().cuda(rank)
    else:
        criterion = nn.BCEWithLogitsLoss().cuda(rank)
    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')

    # Train & Eval
    for epoch in range(args.epoch):
        start = time.time()
        train_sampler.set_epoch(epoch)
        if args.model == "MAML":
            train(model=model, model_type=args.model, optimizer=optimizer, scaler=scaler, train_loader=train_loader, train_logger=train_logger,
                epoch=epoch, embedding_loss=embedding_loss, feature_loss=feature_loss, covariance_loss=covariance_loss, experiment=experiment)
        else:
            train(model=model, model_type=args.model, optimizer=optimizer, scaler=scaler, train_loader=train_loader, train_logger=train_logger,
                epoch=epoch, criterion=criterion, experiment=experiment)
        print('epoch time : ', time.time() - start, 'sec/epoch => ', (time.time() - start) / 60, 'min/epoch')
        # Save and test Model every n epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            start = time.time()
            test(model=model, model_type=args.model, test_loader=test_loader, test_logger=test_logger, epoch=epoch, 
                test_pos_item_num=test_pos_item_num, item_num_dict=item_num_dict, experiment=experiment)
            # if rank == 0:
            #     torch.save(model.state_dict(), f"{save_path}/model_{epoch + 1}.pth")
            print('test time : ', time.time() - start, 'sec/epoch => ', (time.time() - start) / 60, 'min')
    cleanup()
    experiment.end()


def train(model, model_type, optimizer, scaler, train_loader, train_logger, epoch, **kwargs):
    model.train()
    experiment = kwargs['experiment']
    total_loss = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()
    end = time.time()
    if model_type == "MAML":
        embed_loss = AverageMeter()
        feat_loss = AverageMeter()
        cov_loss = AverageMeter()
        embedding_loss = kwargs['embedding_loss']
        feature_loss = kwargs['feature_loss']
        covariance_loss = kwargs['covariance_loss']
    else: # NCF
        criterion = kwargs['criterion']
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if model_type == "MAML":
                (user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n) = data
                user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n = \
                    user.cuda(dist.get_rank()), item_p.cuda(dist.get_rank()), \
                    item_n.cuda(dist.get_rank()), t_feature_p.cuda(dist.get_rank()), \
                    t_feature_n.cuda(dist.get_rank()), img_p.cuda(dist.get_rank()), img_n.cuda(dist.get_rank())
                a_u, a_i, a_i_feature, dist_a = model(user, torch.hstack([item_p.unsqueeze(1), item_n]), \
                                                    torch.hstack([t_feature_p.unsqueeze(1), t_feature_n]),
                                                    torch.hstack([img_p.unsqueeze(1), img_n]))
            else: # NCF
                (user, item, rating, t_feature, img) = data
                user, item, rating, t_feature, img = user.cuda(dist.get_rank()), item.cuda(dist.get_rank()), rating.cuda(dist.get_rank()), t_feature.cuda(dist.get_rank()), img.cuda(dist.get_rank())
                score = model(user, item, image=img, text=t_feature, feature_type=args.feature_type)
            # Loss
            if model_type == "MAML":
                loss_e = embedding_loss(dist_a[:, 0], dist_a[:, 1:])
                if args.feature_type != "rating":
                    loss_f = feature_loss(a_i[:, 0], a_i_feature[:, 0], a_i[:, 1:], a_i_feature[:, 1:])
                    loss_c = covariance_loss(a_u[:, 0], a_i[:, 0], a_i[:, 1:])
                    loss = loss_e + (args.feat_weight * loss_f) + (args.cov_weight * loss_c)

                    rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())
                    rd_train_loss_e = reduce_tensor(loss_e.data, dist.get_world_size())
                    rd_train_loss_c = reduce_tensor(loss_c.data, dist.get_world_size())
                    rd_train_loss_f = reduce_tensor(loss_f.data, dist.get_world_size())
                else:
                    loss_f = torch.zeros(1)
                    loss_c = torch.zeros(1)
                    loss = loss_e
                    rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())
                    rd_train_loss_e = reduce_tensor(loss_e.data, dist.get_world_size())
                    rd_train_loss_c = loss_c
                    rd_train_loss_f = loss_f
            else: # NCF
                loss = criterion(score, rating)
                rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if model_type == "MAML":
            total_loss.update(rd_train_loss.item(), user.shape[0])
            embed_loss.update(rd_train_loss_e.item(), user.shape[0])
            feat_loss.update(rd_train_loss_f.item(), user.shape[0])
            cov_loss.update(rd_train_loss_c.item(), user.shape[0])
            iter_time.update(time.time() - end)
            end = time.time()
            # experiment.log_metric("embed_loss",embed_loss.avg, step=epoch)
            # experiment.log_metric("feat_loss",feat_loss.avg, step=epoch)
            # experiment.log_metric("cov_loss",cov_loss.avg, step=epoch)
        else:
            total_loss.update(rd_train_loss.item(), user.shape[0]//5)
            iter_time.update(time.time() - end)
            end = time.time()            
        experiment.log_metric("total_loss", total_loss.avg, step=epoch)
            
        if i % 10 == 0 and dist.get_rank() == 0:
            if model_type == "MAML":
                print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                    Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                    Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")
            else: # NCF
                print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                    Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")
                
    # if dist.get_rank() == 0:
    #     if model_type == "MAML":
    #         train_logger.write([epoch, total_loss.avg, embed_loss.avg,
    #                             feat_loss.avg, cov_loss.avg])
    #     else: # NCF
    #         train_logger.write([epoch, total_loss.avg])

def test(model, model_type, test_loader, test_logger, epoch, test_pos_item_num, item_num_dict, experiment):
    model.eval()
    hr = AverageMeter()
    hr2 = AverageMeter()
    ndcg = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()
    
    score_cat=torch.tensor([]).cuda(dist.get_rank())
    
    end = time.time()
    user_count = 0
    for i, (user, item, feature, image) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            user, item, feature, image = user.squeeze(-1), item.squeeze(-1), feature.squeeze(-1), image.squeeze(-1)
            user, item, feature, image = \
                user.cuda(dist.get_rank(), non_blocking=True), item.cuda(dist.get_rank(), non_blocking=True), \
                feature.cuda(dist.get_rank(), non_blocking=True), image.cuda(dist.get_rank(), non_blocking=True)
            if model_type == "MAML":
                _, _, _, score = model(user, item, feature, image)
            else: # NCF
                score = model(user, item, image=image, text=feature, feature_type=args.feature_type)
            score_cat=torch.cat((score_cat, score))

            if i%1000==0 and dist.get_rank()==0:
                print(f"test iter : {i}/{len(test_loader)}")
                    
            while(len(score_cat) >= item_num_dict[user_count]): 
                score_sub_tensor = score_cat[:item_num_dict[user_count]]
                score_cat = score_cat[item_num_dict[user_count]:]
                if model_type == "MAML":
                    _, indices = torch.topk(-score_sub_tensor, args.top_k)
                else: # NCF
                    _, indices = torch.topk(score_sub_tensor, args.top_k)    
                recommends = indices
                gt_item = torch.tensor(range(test_pos_item_num[user_count])).cuda(dist.get_rank())
                performance = get_performance(gt_item, recommends)
                performance = torch.tensor(performance).cuda(dist.get_rank())
                hr.update(performance[0])
                hr2.update(performance[1])
                ndcg.update(performance[2])
                user_count += 1
                if user_count == len(item_num_dict.keys()):
                    break    
                    
    print('남은 스코어', score_cat.shape)
    iter_time.update(time.time() - end)
    end = time.time()

    experiment.log_metric("hit-ratio", hr.avg, step=epoch)
    experiment.log_metric("hit-ratio2", hr2.avg, step=epoch)
    experiment.log_metric("ndcg", ndcg.avg, step=epoch)

    if dist.get_rank() == 0:
        print(f"{user_count} Users tested. Iteration time : {iter_time.avg:.5f}/user Data time : {data_time.avg:.5f}/user")
    if dist.get_rank() == 0:
        print(f"Epoch : [{epoch + 1}/{args.epoch}] Hit Ratio : {hr.avg:.4f} nDCG : {ndcg.avg:.4f} Hit Ratio 2 : {hr2.avg:.4f} Test Time : {iter_time.avg:.4f}/user")
        # test_logger.write([epoch, float(hr.avg), float(hr2.avg), float(ndcg.avg)])


def my_collate_trn(batch):
    if len(batch[0]) == 7:
        user = [item[0] for item in batch]
        user = torch.LongTensor(user)
        item_p = [item[1] for item in batch]
        item_p = torch.LongTensor(item_p)
        item_n = [item[2] for item in batch]
        item_n = torch.LongTensor(item_n)
        t_feature_p = [item[3] for item in batch]
        t_feature_p = torch.FloatTensor(t_feature_p)
        t_feature_n = [item[4] for item in batch]
        t_feature_n = torch.FloatTensor(t_feature_n)
        img_p = [item[5] for item in batch]
        img_p = torch.stack(img_p)
        img_n = [item[6] for item in batch]
        img_n = torch.stack(img_n)

        return [user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n]
    else:
        user = [element for item in batch for element in item[0]]
        user = torch.LongTensor(user)
        items = [element for item in batch for element in item[1]]
        items = torch.LongTensor(items)
        rating = [element for item in batch for element in item[2]]
        rating = torch.FloatTensor(rating)
        t_feature = [element for item in batch for element in item[3]]
        t_feature = torch.FloatTensor(t_feature)
        img = [element for item in batch for element in item[4]]
        img = torch.stack(img)
    
        return [user, items, rating, t_feature, img]

def my_collate_tst(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    t_feature = [items[2] for items in batch]
    t_feature = torch.FloatTensor(t_feature)
    img = [items[3] for items in batch]
    img = torch.stack(img)
    return [user, item, t_feature, img]


def init_process(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.ddp_addr
    os.environ['MASTER_PORT'] = args.ddp_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
  
def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    args.world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=args.world_size, args=(args,))
    #main()
