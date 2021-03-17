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
from model_attention import MAML
from loss import Embedding_loss, Feature_loss, Covariance_loss
import dataset as D
from metric import get_performance
import torch.distributed as dist


parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./bufftoon_loo_rating_4', type=str,
                    help='savepath')
parser.add_argument('--batch_size', default=840, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=50, type=int,
                    help='train epoch')
parser.add_argument('--data_path', default='/daintlab/data/bufftoon', type=str,
                    help='Path to rating data')
parser.add_argument('--embed_dim', default=128, type=int,
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
parser.add_argument('--feature_type', default='rating', type=str,
                    help='Type of feature to use. [rating, all, img, txt]')
parser.add_argument('--eval_type', default='leave-one-out', type=str,
                    help='Evaluation protocol. [ratio-split, leave-one-out]')
parser.add_argument('--cnn_path', default='./pretrained_model/resnet18.pth', type=str,
                    help='Path to feature data')
parser.add_argument('--fine_tuning', default=False, type=bool,
                    help='Fine tuning')
parser.add_argument('--hier_attention', default=False, type=bool,
                    help='Hierarchical attention')
parser.add_argument('--ddp_port', default='88888', type=str,
                    help='DDP Port')
parser.add_argument('--att_wd', default=100, type=float)
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


    # Load dataset
    print("Loading Dataset")
    data_path = os.path.join(args.data_path, args.eval_type)
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, text_feature, images = D.load_data(
        data_path, args.feature_type)
    train_dataset = D.CustomDataset(train_df, text_feature, images, negative=train_ng_pool, num_neg=args.num_neg,
                                    istrain=True, feature_type=args.feature_type)
    test_dataset = D.CustomDataset(test_df, text_feature, images, negative=test_negative, num_neg=None,
                                   istrain=False, feature_type=args.feature_type)

    args.batch_size = int(args.batch_size / args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                              collate_fn=my_collate_trn, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size/4), shuffle=False, num_workers=4,
                             collate_fn=my_collate_tst, pin_memory=True)

    # Model
    t_feature_dim = text_feature[0].shape[-1]
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate, args.feature_type, t_feature_dim,
                 args.cnn_path,args.fine_tuning,rank).cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained Model Loaded")

    # Optimizer
    if args.feature_type != "rating":
        optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                      {'params': model.module.embedding_item.parameters()},
                                      {'params': model.module.feature_fusion.parameters()},
                                      {'params': model.module.attention.parameters(), 'weight_decay': args.att_wd}],
                                     lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{'params': model.module.embedding_user.parameters()},
                                      {'params': model.module.embedding_item.parameters()},
                                      {'params': model.module.attention.parameters(), 'weight_decay': args.att_wd}],
                                     lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # Loss
    embedding_loss = Embedding_loss(margin=args.margin, num_item=num_item).cuda(rank)
    feature_loss = Feature_loss().cuda(rank)
    covariance_loss = Covariance_loss().cuda(rank)

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')
    # Train & Eval
    for epoch in range(args.epoch):
        start = time.time()
        train_sampler.set_epoch(epoch)
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer, scaler, train_loader, train_logger,
             epoch, args.hier_attention)
        print('epoch time : ', time.time() - start, 'sec/epoch => ', (time.time() - start) / 60, 'min/epoch')
        # Save and test Model every n epoch
        if (epoch + 1) % args.eval_freq == 0:
            if dist.get_rank() == 0:
                start = time.time()
                test(model, test_loader, test_logger, epoch, args.hier_attention)
                torch.save(model.state_dict(), f"{save_path}/model_{epoch + 1}.pth")
                print('test time : ', time.time() - start, 'sec/epoch => ', (time.time() - start) / 60, 'min')


def train(model, embedding_loss, feature_loss, covariance_loss, optimizer, scaler, train_loader, train_logger, epoch, hier_attention):
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
            user, item_p, item_n, t_feature_p, t_feature_n, img_p, img_n = \
                user.cuda(dist.get_rank()), item_p.cuda(dist.get_rank()), \
                item_n.cuda(dist.get_rank()), t_feature_p.cuda(dist.get_rank()), \
                t_feature_n.cuda(dist.get_rank()), img_p.cuda(dist.get_rank()), img_n.cuda(dist.get_rank())

            a_u, a_i, a_i_feature, dist_a = model(user, torch.hstack([item_p.unsqueeze(1), item_n]), \
                                                  torch.hstack([t_feature_p.unsqueeze(1), t_feature_n]),
                                                  torch.hstack([img_p.unsqueeze(1), img_n]),hier_attention)

            # Loss
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss.update(rd_train_loss.item(), user.shape[0])
        embed_loss.update(rd_train_loss_e.item(), user.shape[0])
        feat_loss.update(rd_train_loss_f.item(), user.shape[0])
        cov_loss.update(rd_train_loss_c.item(), user.shape[0])
        iter_time.update(time.time() - end)
        end = time.time()


        if i % 10 == 0 and dist.get_rank() == 0:
            print(f"[{epoch + 1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")
    if dist.get_rank() == 0:
        train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                            feat_loss.avg, cov_loss.avg])


def test(model, test_loader, test_logger, epoch, hier_attention):
    model.eval()
    hr_10 = AverageMeter()
    hr2_10 = AverageMeter()
    ndcg_10 = AverageMeter()
    hr_5 = AverageMeter()
    hr2_5 = AverageMeter()
    ndcg_5 = AverageMeter()
    hr_3 = AverageMeter()
    hr2_3 = AverageMeter()
    ndcg_3 = AverageMeter()
    hr_1 = AverageMeter()
    hr2_1 = AverageMeter()
    ndcg_1 = AverageMeter()


    data_time = AverageMeter()
    iter_time = AverageMeter()
    user_cat = torch.tensor([]).cuda(dist.get_rank())
    score_cat=torch.tensor([]).cuda(dist.get_rank())
    label_cat=torch.tensor([]).cuda(dist.get_rank())
    item_cat = torch.tensor([]).cuda(dist.get_rank())
    end = time.time()

    for i, (user, item, feature, image, label) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            user, item, feature, image, label = user.squeeze(0), item.squeeze(0), feature.squeeze(0), image.squeeze(
               0), label.squeeze(0)
            user, item, feature, image, label = \
                user.cuda(dist.get_rank(), non_blocking=True), item.cuda(dist.get_rank(), non_blocking=True), \
                feature.cuda(dist.get_rank(), non_blocking=True), image.cuda(dist.get_rank(), non_blocking=True), \
                label.cuda(dist.get_rank(), non_blocking=True)
            _, _, _, score = model(user, item, feature, image, hier_attention)
            user_cat=torch.cat((user_cat,user))
            score_cat=torch.cat((score_cat, score))
            label_cat=torch.cat((label_cat, label))
            item_cat=torch.cat((item_cat, item))

            if i%10==0 and dist.get_rank()==0:
                print(f"test iter : {i}/{len(test_loader)}")

    for k in np.unique(user_cat.cpu()):
        index=np.where(np.array(user_cat.cpu().detach())==k)
        label=label_cat[index]
        score=score_cat[index]
        item=item_cat[index]
        performance_list=[]
        for m in [10,5,3,1] :
            pos_idx = label.nonzero()
            _, indices = torch.topk(-score, m)
            recommends = torch.take(item, indices).cpu().detach().numpy()
            gt_item = item[pos_idx].cpu().detach().numpy()
            performance = get_performance(gt_item, recommends.tolist())
            performance = torch.tensor(performance).cuda(dist.get_rank())
            performance_list.append(performance)
        hr_10.update(performance_list[0][0].cpu().detach().numpy())
        hr2_10.update(performance_list[0][1].cpu().detach().numpy())
        ndcg_10.update(performance_list[0][2].cpu().detach().numpy())
        hr_5.update(performance_list[1][0].cpu().detach().numpy())
        hr2_5.update(performance_list[1][1].cpu().detach().numpy())
        ndcg_5.update(performance_list[1][2].cpu().detach().numpy())
        hr_3.update(performance_list[2][0].cpu().detach().numpy())
        hr2_3.update(performance_list[2][1].cpu().detach().numpy())
        ndcg_3.update(performance_list[2][2].cpu().detach().numpy())
        hr_1.update(performance_list[3][0].cpu().detach().numpy())
        hr2_1.update(performance_list[3][1].cpu().detach().numpy())
        ndcg_1.update(performance_list[3][2].cpu().detach().numpy())

        iter_time.update(time.time() - end)
        end = time.time()

        if k % 50 == 0 and dist.get_rank() == 0:
            print(
                f"{k + 1} Users tested. Iteration time : {iter_time.avg:.5f}/user Data time : {data_time.avg:.5f}/user")
    if dist.get_rank() == 0:
        print(
            f"Epoch : [{epoch + 1}/{args.epoch}] Hit Ratio : {hr_10.avg:.4f} nDCG : {ndcg_10.avg:.4f} Hit Ratio 2 : {hr2_10.avg:.4f} Test Time : {iter_time.avg:.4f}/user")
        test_logger.write([epoch, hr_10.avg, hr2_10.avg, ndcg_10.avg,\
                           hr_5.avg, hr2_5.avg, ndcg_5.avg,\
                           hr_3.avg,hr2_3.avg, ndcg_3.avg,\
                           hr_1.avg, hr2_1.avg, ndcg_1.avg])


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
    os.environ['MASTER_PORT'] = args.ddp_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


if __name__ == "__main__":
    args.world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=args.world_size, args=(args,))
    #main()
