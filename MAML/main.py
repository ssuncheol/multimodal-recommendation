# from comet_ml import Experiment
import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.multiprocessing as mp

from utils import Logger, AverageMeter, str2bool
from model import MAML
from loss import Embedding_loss, Feature_loss, Covariance_loss
import dataset_amazon as D_a
import dataset_movie as D_m
from metric import get_performance


parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./result', type=str,
                    help='savepath')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=1000, type=int,
                    help='train epoch')
parser.add_argument('--data_path', default='/daintlab/data/recommend/amazon_review/Office', type=str,
                    help='Path to dataset')
parser.add_argument('--dataset', default='amazon', type=str,
                    help='Dataset : amazon or movielens')
parser.add_argument('--embed_dim', default=64, type=int,
                    help='Embedding Dimension')
parser.add_argument('--dropout_rate', default=0.2, type=float,
                    help='Dropout rate')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--margin', default=1.6, type=float,
                    help='Margin for embedding loss')
parser.add_argument('--feat_weight', default=7, type=float,
                    help='Weight of feature loss')
parser.add_argument('--cov_weight', default=5, type=float,
                    help='Weight of covariance loss')
parser.add_argument('--top_k', default=10, type=int,
                    help='Top k Recommendation')
parser.add_argument('--num_neg', default=4, type=int,
                    help='Number of negative samples for training')
parser.add_argument('--load_path', default=None, type=str,
                    help='Path to saved model')
parser.add_argument('--use_feature', default=True, type=str2bool,
                    help='Whether to use auxiliary information of items')
parser.add_argument('--eval_freq', default=50, type=int,
                    help='evaluate performance every n epoch')
parser.add_argument('--feature_type', default='all', type=str,
                    help='Type of feature to use. [all, img, txt]')

args = parser.parse_args()


def main():
    # Set Comet ML
    # experiment = Experiment("xlgnV9PHcoau26wzSxohP8ToM",
    #                         project_name="MAML",
    #                         log_git_metadata=False,
    #                         log_git_patch=False)
    # hyperparameters = {
    #     "dataset" : args.dataset,
    #     "epoch" : args.epoch,
    #     "margin" : args.margin,
    #     "lr" : args.lr,
    #     "feature_weight" : args.feat_weight,
    #     "covariance_weight" : args.cov_weight,
    #     "use_feature" : args.use_feature,
    #     "feature_type" : args.feature_type
    # }
    # experiment.log_parameters(hyperparameters)

    # Set save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save configuration
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load dataset
    print("Loading Dataset")
    path = args.data_path
    dataset=args.dataset
    if dataset=='amazon':
        train_df, test_df, train_ng_pool, test_negative, num_user, num_item, feature = D_a.load_data(path, args.feature_type)
        train_dataset = D_a.CustomDataset_amazon(train_df, feature, negative=train_ng_pool, num_neg=args.num_neg, istrain=True, use_feature=args.use_feature)
        test_dataset = D_a.CustomDataset_amazon(test_df, feature, negative=test_negative, num_neg=None, istrain=False, use_feature=args.use_feature)
    elif dataset=='movielens':
        train_df, test_df, train_ng_pool, test_negative, num_user, num_item, feature = D_m.load_data(path, args.feature_type)
        train_dataset = D_m.CustomDataset_movielens(train_df, feature, negative=train_ng_pool, num_neg=args.num_neg, istrain=True, use_feature=args.use_feature)
        test_dataset = D_m.CustomDataset_movielens(test_df, feature, negative=test_negative, num_neg=None, istrain=False, use_feature=args.use_feature)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=my_collate_trn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=my_collate_tst, pin_memory =True)

    # Model
    feature_dim = feature.shape[-1]
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate, args.use_feature, feature_dim).cuda()
    print(model)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint)
        print("Pretrained Model Loaded")

    # Optimizer
    if args.use_feature:
        optimizer = torch.optim.Adam([{'params': model.embedding_user.parameters()},
                                        {'params': model.embedding_item.parameters()},
                                        {'params': model.feature_fusion.parameters()},
                                        {'params': model.attention.parameters(), 'weight_decay':100.0}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{'params': model.embedding_user.parameters()},
                                        {'params': model.embedding_item.parameters()},
                                        {'params': model.attention.parameters(), 'weight_decay':100.0}], lr=args.lr)

    # Loss
    embedding_loss = Embedding_loss(margin=args.margin, num_item=num_item).cuda()
    feature_loss = Feature_loss().cuda()
    covariance_loss = Covariance_loss().cuda()

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')

    # Train & Eval
    for epoch in range(args.epoch):
        # with experiment.train():
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer, train_loader, train_logger, epoch)
        # Save and test Model every n epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            # with experiment.test():
            test(model, test_loader, test_logger, epoch)
            torch.save(model.state_dict(), f"{save_path}/model_{epoch + 1}.pth")


def train(model, embedding_loss, feature_loss, covariance_loss, optimizer, train_loader, train_logger, epoch):
    model.train()
    total_loss = AverageMeter()
    embed_loss = AverageMeter()
    feat_loss = AverageMeter()
    cov_loss = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()

    end = time.time()
    for i, (user, item_p, item_n, feature_p, feature_n) in enumerate(train_loader):
        data_time.update(time.time() - end)
        user, item_p, item_n, feature_p, feature_n = user.cuda(), item_p.cuda(), item_n.cuda(), feature_p.cuda(), feature_n.cuda()

        p_u, q_i, q_i_feature, dist_p = model(user, item_p, feature_p)
        _, q_k, q_k_feature, dist_n = model(user, item_n, feature_n)
        # Loss
        loss_e = embedding_loss(dist_p, dist_n)
        if args.use_feature:
            loss_f = feature_loss(q_i, q_i_feature, q_k, q_k_feature)
            loss_c = covariance_loss(p_u, q_i, q_k)
            loss = loss_e + (args.feat_weight * loss_f) + (args.cov_weight * loss_c)
        else:
            loss_f = torch.zeros(1)
            loss_c = torch.zeros(1)
            loss = loss_e

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        embed_loss.update(loss_e.item())
        feat_loss.update(loss_f.item())
        cov_loss.update(loss_c.item())
        iter_time.update(time.time() - end)
        end = time.time()

        # Comet ml
        # experiment.log_metric("Train total loss", loss.item(), step=i, epoch= epoch)
        # experiment.log_metric("Train embedding loss", loss_e.item(), step=i, epoch= epoch)
        # experiment.log_metric("Train feature loss", loss_f.item(), step=i, epoch= epoch)
        # experiment.log_metric("Train covariance loss", loss_c.item(), step=i, epoch= epoch)

        if i % 10 == 0:
            print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")

    train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                        feat_loss.avg, cov_loss.avg])


def test(model, test_loader, test_logger, epoch):
    model.eval()
    hr = AverageMeter()
    ndcg = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()
    end = time.time()
    for i, (user, item, feature, label) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            user, item, feature, label = user.squeeze(0), item.squeeze(0), feature.squeeze(0), label.squeeze(0)
            user, item, feature, label = user.cuda(), item.cuda(), feature.cuda(), label.cuda()
            _, _, _, score = model(user, item, feature)

            pos_idx = label.nonzero()
            _, indices = torch.topk(-score, args.top_k)
            recommends = torch.take(item, indices).cpu().numpy()
            gt_item = item[pos_idx].cpu().numpy()
            performance = get_performance(gt_item, recommends)
            hr.update(performance[0])
            ndcg.update(performance[1])

            iter_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print(f"{i + 1} Users tested. Iteration time : {iter_time.avg:.5f}/user Data time : {data_time.avg:.5f}/user")

    print(
        f"Epoch : [{epoch + 1}/{args.epoch}] Hit Ratio : {hr.avg:.4f} nDCG : {ndcg.avg:.4f} Test Time : {iter_time.avg:.4f}/user")
    test_logger.write([epoch, hr.avg, ndcg.avg])
    # experiment.log_metric("Test HR@K", hr.avg, epoch= epoch)
    # experiment.log_metric("Test nDCG@K", ndcg.avg, epoch= epoch)

def my_collate_trn(batch):
    user = [item[0] for item in batch]
    user = torch.LongTensor(user)
    item_p = [item[1] for item in batch]
    item_p = torch.LongTensor(item_p)
    item_n = [item[2] for item in batch]
    item_n = torch.LongTensor(item_n)
    feature_p = [item[3] for item in batch]
    feature_p = torch.Tensor(feature_p)
    feature_n = [item[4] for item in batch]
    feature_n = torch.Tensor(feature_n)
    return [user, item_p, item_n, feature_p, feature_n]


def my_collate_tst(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    feature = [items[2] for items in batch]
    feature = torch.Tensor(feature)
    label = [items[3] for items in batch]
    label = torch.Tensor(label)
    return [user, item, feature, label]


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()