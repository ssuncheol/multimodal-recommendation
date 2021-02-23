import torch
import time
from metric import get_performance
import numpy as np
import torch.distributed as dist

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

class Engine(object):
    def __init__(self, top_k, item_num_dict, test_pos_item_num, max, num_user, rank, world_size):
        self.top_k = top_k
        self.item_num_dict = item_num_dict
        self.test_pos_item_num = test_pos_item_num
        self.max = max # test item 개수의 최대값
        self.num_user = num_user
        self.rank = rank
        self.world_size = world_size
    def evaluate(self, model, test_loader, epoch_id, **kwargs):
        #Evaluate model
        t0 = time.time()
        
        model.eval()

        score_tensor = torch.tensor([]).cuda()
        hr_list = []
        hr2_list = []
        ndcg_list = []
        for i , data in enumerate(test_loader): 
            with torch.no_grad():    
                if (kwargs['image'] is not None) & (kwargs['text'] is not None):
                    user, item, image_f, text_f = data
                    user, item, image_f, text_f = user.squeeze(1), item.squeeze(1), image_f.squeeze(1), text_f.squeeze(1)
                    user, item, image_f, text_f = user.cuda(non_blocking=True), item.cuda(non_blocking=True), image_f.cuda(non_blocking=True), text_f.cuda(non_blocking=True)
                elif kwargs['image'] is not None:
                    user, item, image_f = data
                    user, item, image_f = user.squeeze(1), item.squeeze(1), image_f.squeeze(1)
                    user, item, image_f = user.cuda(non_blocking=True), item.cuda(non_blocking=True), image_f.cuda(non_blocking=True)
                    text_f = None
                elif kwargs['text'] is not None:
                    user, item, text_f = data
                    user, item, text_f = user.squeeze(1), item.squeeze(1), text_f.squeeze(1)
                    user, item, text_f = user.cuda(non_blocking=True), item.cuda(non_blocking=True), text_f.cuda(non_blocking=True)
                    image_f = None
                else:
                    user, item = data
                    user, item = user.squeeze(1), item.squeeze(1)
                    user, item = user.cuda(non_blocking=True), item.cuda(non_blocking=True)
                    image_f = None
                    text_f = None
                
                score = model(user, item, image=image_f, text=text_f)
                score_tensor = torch.cat((score_tensor, score))

        start = 0
        end = 0
        for i in range(self.num_user):
            end = start + self.item_num_dict[i]
            score_sub_tensor = score_tensor[start:end]
            _, indices = torch.topk(score_sub_tensor, self.top_k)
            recommends = indices.cpu().numpy()
            gt_item = np.array(range(self.test_pos_item_num[i]))
            performance = get_performance(gt_item, recommends.tolist())
            performance = torch.tensor(performance).cuda(self.rank)

            rd_hr = reduce_tensor(performance[0].data, self.world_size)
            rd_hr2 = reduce_tensor(performance[1].data, self.world_size)
            rd_ndcg = reduce_tensor(performance[2].data, self.world_size)

            hr_list.append(rd_hr.item())
            hr2_list.append(rd_hr2.item())
            ndcg_list.append(rd_ndcg.item())
            start = end
        hit_ratio, hit_ratio2, ndcg = np.mean(hr_list), np.mean(hr2_list), np.mean(ndcg_list)
        print('[Evluating Epoch {}] HR = {:.4f}, HR2 = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, hit_ratio2, ndcg))
        
        t1 = time.time()
        print("evaluate time:", t1 - t0)  
        return hit_ratio, hit_ratio2, ndcg
        
