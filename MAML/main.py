from comet_ml import Experiment
import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
from utils import Logger, AverageMeter, str2bool
from model import MAML
from loss import Embedding_loss, Feature_loss, Covariance_loss
import dataset as D
from metric import get_performance
import resnet_tv as resnet
import torch.distributed as dist

parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./result', type=str,
                    help='savepath')
parser.add_argument('--batch_size', default=568, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='train epoch')
parser.add_argument('--data_path', default='/daintlab/data/recommend/Amazon-office-raw', type=str,
                    help='Path to rating data')
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
parser.add_argument('--feature_type', default='all', type=str,
                    help='Type of feature to use. [all, img, txt]')
parser.add_argument('--eval_type', default='ratio-split', type=str,
                    help='Evaluation protocol. [ratio-split, leave-one-out]')
parser.add_argument('--cnn_path', default='./pretrained_model/resnet18.pth', type=str,
                    help='Path to feature data')
args = parser.parse_args()


def main(rank, args):
    # Initialize Each Process
    init_process(rank, args.world_size)
    hyper_params={
        "batch_size":args.batch_size,
        "epoch":args.epoch,
        "embed_dim":args.embed_dim,
        "dropout_rate":args.dropout_rate,
        "learning_rate":args.lr,
        "margin":args.margin,
        "feat_weight":args.feat_weight,
        "cov_weight":args.cov_weight,
        "top_k":args.top_k,
        "num_neg":args.num_neg,
        "eval_freq":args.eval_freq,
        "feature_type":args.feature_type,
        "eval_type":args.eval_type
    }
    # Set save path
    save_path = args.save_path
    if not os.path.exists(save_path) and dist.get_rank() == 0:
        os.makedirs(save_path)
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if dist.get_rank() == 0:
        experiment = Experiment(api_key="ZSGBzbLxOxZe4qEZ917ZbOV1m",project_name='recommend',log_code=False,
        auto_param_logging=False, auto_metric_logging=False, log_env_details=False, log_git_metadata=False,
        log_git_patch=False)
        experiment.log_parameters(hyper_params)
    else:
        experiment=Experiment(api_key="ZSGBzbLxOxZe4qEZ917ZbOV1m",disabled=True)
    
    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path, args.eval_type)
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, text_feature, images = D.load_data(
        data_path, args.feature_type)

    train_dataset = D.CustomDataset(train_df, text_feature, images, negative=train_ng_pool, num_neg=args.num_neg,
                                    istrain=True, feature_type=args.feature_type)
    test_dataset = D.CustomDataset(test_df, text_feature, images, negative=test_negative, num_neg=None,
                                   istrain=False, feature_type=args.feature_type)
    args.batch_size = int(args.batch_size/args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                              collate_fn=my_collate_trn, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                             collate_fn=my_collate_tst, pin_memory=True)

    # Model
    t_feature_dim = text_feature[0].shape[-1]
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate, args.feature_type, t_feature_dim,args.cnn_path).cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint,strict=False)
        print("Pretrained Model Loaded")

    # Optimizer
    if args.feature_type != "rating":
        optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                      {'params': model.module.embedding_item.parameters()},
                                      {'params': model.module.feature_fusion.parameters()},
                                      {'params': model.module.attention.parameters(), 'weight_decay': 100.0}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                      {'params': model.module.embedding_item.parameters()},
                                      {'params': model.module.attention.parameters(), 'weight_decay': 100.0}], lr=args.lr)
    scaler=torch.cuda.amp.GradScaler()

    # Loss
    embedding_loss = Embedding_loss(margin=args.margin, num_item=num_item).cuda(rank)
    feature_loss = Feature_loss().cuda(rank)
    covariance_loss = Covariance_loss().cuda(rank)

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')
    hit_record_logger = Logger(f'{save_path}/hitrecord.log')

    # Train & Eval
    for epoch in range(args.epoch):
        start=time.time()
        train_sampler.set_epoch(epoch)
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer,scaler, train_loader, train_logger, epoch, experiment)
        print('epoch time : ', time.time()-start, 'sec/epoch => ', (time.time()-start)/60, 'min/epoch')
        # Save and test Model every n epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            start=time.time()
            test(model, test_loader, test_logger, epoch, hit_record_logger, experiment)
            torch.save(model.state_dict(), f"{save_path}/model_{epoch + 1}.pth")
            print('test time : ', time.time()-start, 'sec/epoch => ', (time.time()-start)/60, 'min')
    cleanup()
    experiment.end()

def train(model, embedding_loss, feature_loss, covariance_loss, optimizer,scaler, train_loader, train_logger, epoch, experiment):
    model.train()
    total_loss = AverageMeter()
    embed_loss = AverageMeter()
    feat_loss = AverageMeter()
    cov_loss = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()
    end = time.time()
    for i, (user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n =\
             user.cuda(dist.get_rank()), item_p.cuda(dist.get_rank()), \
             item_n.cuda(dist.get_rank()), t_feature_p.cuda(dist.get_rank()), \
             t_feature_n.cuda(dist.get_rank()), img_p.cuda(dist.get_rank()), img_n.cuda(dist.get_rank())

            a_u,a_i,a_i_feature,dist_a=model(user,torch.hstack([item_p.unsqueeze(1),item_n]), \
                torch.hstack([t_feature_p.unsqueeze(1),t_feature_n]),torch.hstack([img_p.unsqueeze(1),img_n]))

            # Loss
            loss_e = embedding_loss(dist_a[:,0], dist_a[:,1:])
            if args.feature_type != "rating":
                loss_f = feature_loss(a_i[:,0], a_i_feature[:,0], a_i[:,1:],a_i_feature[:,1:])
                loss_c = covariance_loss(a_u[:,0], a_i[:,0],a_i[:,1:])
                loss = loss_e + (args.feat_weight * loss_f) + (args.cov_weight * loss_c)
            else:
                loss_f = torch.zeros(1)
                loss_c = torch.zeros(1)
                loss = loss_e
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())
        rd_train_loss_e = reduce_tensor(loss_e.data, dist.get_world_size())
        rd_train_loss_c = reduce_tensor(loss_c.data, dist.get_world_size())
        rd_train_loss_f = reduce_tensor(loss_f.data, dist.get_world_size())

        total_loss.update(rd_train_loss.item(),user.shape[0])
        embed_loss.update(rd_train_loss_e.item(),user.shape[0])
        feat_loss.update(rd_train_loss_f.item(),user.shape[0])
        cov_loss.update(rd_train_loss_c.item(),user.shape[0])
        iter_time.update(time.time() - end)
        end = time.time()

        experiment.log_metric("total_loss",total_loss.avg, step=epoch)
        experiment.log_metric("embed_loss",embed_loss.avg, step=epoch)
        experiment.log_metric("feat_loss",feat_loss.avg, step=epoch)
        experiment.log_metric("cov_loss",cov_loss.avg, step=epoch)
        
        if i % 10 == 0 and dist.get_rank() == 0:
            print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")
    if dist.get_rank() == 0:
        train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                            feat_loss.avg, cov_loss.avg])

def test(model, test_loader, test_logger, epoch, hit_record_logger, experiment):
    model.eval()
    hr = AverageMeter()
    hr2 = AverageMeter()
    ndcg = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()
    end = time.time()
    for i, (user, item, feature, image, label) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            user, item, feature, image, label = user.squeeze(0), item.squeeze(0), feature.squeeze(0), image.squeeze(
                0), label.squeeze(0)
            user, item, feature, image, label = \
                user.cuda(dist.get_rank(),non_blocking=True), item.cuda(dist.get_rank(),non_blocking=True), \
                feature.cuda(dist.get_rank(),non_blocking=True), image.cuda(dist.get_rank(),non_blocking=True), \
                label.cuda(dist.get_rank(),non_blocking=True)
            _, _, _, score = model(user, item, feature, image)
            pos_idx = label.nonzero()
            _, indices = torch.topk(-score, args.top_k)
            recommends = torch.take(item, indices).cpu().numpy()
            gt_item = item[pos_idx].cpu().numpy()
            performance = get_performance(gt_item, recommends.tolist())
            performance = torch.tensor(performance).cuda(dist.get_rank())    

            rd_hr = reduce_tensor(performance[0].data, dist.get_world_size())
            rd_hr2 = reduce_tensor(performance[1].data, dist.get_world_size())
            rd_ndcg = reduce_tensor(performance[2].data, dist.get_world_size())

            hr.update(performance[0].cpu().numpy())
            hr2.update(performance[1].cpu().numpy())
            ndcg.update(performance[2].cpu().numpy())
            iter_time.update(time.time() - end)
            end = time.time()

            experiment.log_metric("hit-ratio",hr.avg,step=epoch)
            experiment.log_metric("hit-ratio2",hr2.avg,step=epoch)
            experiment.log_metric("ndcg",ndcg.avg,step=epoch)

            if epoch + 1 == args.epoch and dist.get_rank() == 0:
                hit_record_logger.write([user[0].item(), len(gt_item), performance[0]])
            if i % 50 == 0 and dist.get_rank() == 0:
                print(
                    f"{i + 1} Users tested. Iteration time : {iter_time.avg:.5f}/user Data time : {data_time.avg:.5f}/user")
    if dist.get_rank() == 0:
        print(
            f"Epoch : [{epoch + 1}/{args.epoch}] Hit Ratio : {hr.avg:.4f} nDCG : {ndcg.avg:.4f} Hit Ratio 2 : {hr2.avg:.4f} Test Time : {iter_time.avg:.4f}/user")
        test_logger.write([epoch, hr.avg, hr2.avg, ndcg.avg])


def my_collate_trn(batch):
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

def my_collate_tst(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    t_feature = [items[2] for items in batch]
    t_feature = torch.FloatTensor(t_feature)
    img = [items[3] for items in batch]
    img = torch.stack(img)
    label = [items[4] for items in batch]
    label = torch.FloatTensor(label)
    return [user, item, t_feature, img, label]

def init_process(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

if __name__ == "__main__":
    args.world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs = args.world_size, args = (args,))
