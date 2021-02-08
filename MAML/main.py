# from comet_ml import Experiment
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
parser.add_argument('--batch_size', default=256, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=100, type=int,
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
parser.add_argument('--cnn_path', default=None, type=str,
                    help='Path to feature data')
args = parser.parse_args()


def main():
    # Set save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save configuration
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path, args.eval_type)
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, text_feature, images = D.load_data(
        data_path, args.feature_type)
    train_dataset = D.CustomDataset(train_df, text_feature, images, negative=train_ng_pool, num_neg=args.num_neg,
                                    istrain=True, feature_type=args.feature_type)
    test_dataset = D.CustomDataset(test_df, text_feature, images, negative=test_negative, num_neg=None,
                                   istrain=False, feature_type=args.feature_type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                              collate_fn=my_collate_trn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                             collate_fn=my_collate_tst, pin_memory=True)

    # Model
    t_feature_dim = text_feature.shape[-1]
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate, args.feature_type, t_feature_dim,args.cnn_path).cuda()

    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint,strict=False)
        print("Pretrained Model Loaded")

    # Optimizer
    if args.feature_type != "rating":
        optimizer = torch.optim.Adam([{'params': model.embedding_user.parameters()},
                                      {'params': model.embedding_item.parameters()},
                                      {'params': model.feature_fusion.parameters()},
                                      {'params': model.attention.parameters(), 'weight_decay': 100.0}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{'params': model.embedding_user.parameters()},
                                      {'params': model.embedding_item.parameters()},
                                      {'params': model.attention.parameters(), 'weight_decay': 100.0}], lr=args.lr)
    scaler=torch.cuda.amp.GradScaler()

    # Loss
    embedding_loss = Embedding_loss(margin=args.margin, num_item=num_item).cuda()
    feature_loss = Feature_loss().cuda()
    covariance_loss = Covariance_loss().cuda()

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')
    hit_record_logger = Logger(f'{save_path}/hitrecord.log')

    # Train & Eval
    for epoch in range(args.epoch):
        start=time.time()
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer, scaler, train_loader, train_logger, epoch)
        print('epoch time : ', time.time()-start, 'sec/epoch => ', (time.time()-start)/60, 'min/epoch')
        # Save and test Model every n epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            test(model, test_loader, test_logger, epoch, hit_record_logger)
            torch.save(model.state_dict(), f"{save_path}/model_{epoch + 1}.pth")

def train(model, embedding_loss, feature_loss, covariance_loss, optimizer, scaler, train_loader, train_logger, epoch):
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
             user.cuda(non_blocking=True), item_p.cuda(non_blocking=True), \
             item_n.cuda(non_blocking=True), t_feature_p.cuda(non_blocking=True), \
             t_feature_n.cuda(non_blocking=True), img_p.cuda(non_blocking=True), img_n.cuda(non_blocking=True)

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

        total_loss.update(loss.item())
        embed_loss.update(loss_e.item())
        feat_loss.update(loss_f.item())
        cov_loss.update(loss_c.item())
        iter_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 :
            print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")
    train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                        feat_loss.avg, cov_loss.avg])

def test(model, test_loader, test_logger, epoch, hit_record_logger):
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
                user.cuda(non_blocking=True), item.cuda(non_blocking=True), \
                feature.cuda(non_blocking=True), image.cuda(non_blocking=True), label.cuda(non_blocking=True)
            _, _, _, score = model(user, item, feature, image)
            pos_idx = label.nonzero()
            _, indices = torch.topk(-score, args.top_k)
            recommends = torch.take(item, indices).cpu().numpy()
            gt_item = item[pos_idx].cpu().numpy()
            performance = get_performance(gt_item, recommends.tolist())
            hr.update(performance[0])
            hr2.update(performance[1])
            ndcg.update(performance[2])
            iter_time.update(time.time() - end)
            end = time.time()
            if epoch + 1 == args.epoch:
                hit_record_logger.write([user[0].item(), len(gt_item), performance[0]])
            if i % 50 == 0:
                print(
                    f"{i + 1} Users tested. Iteration time : {iter_time.avg:.5f}/user Data time : {data_time.avg:.5f}/user")
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



if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
