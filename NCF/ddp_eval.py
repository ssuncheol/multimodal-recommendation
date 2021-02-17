import torch
import time
from metric import get_performance
import numpy as np
import torch.distributed as dist
# from main import reduce_tensor
class Engine(object):
    def __init__(self, top_k):
        self.top_k = top_k
        
    def evaluate(self, model, test_loader, epoch_id, **kwargs):
        #Evaluate model
        t0 = time.time()
        
        model.eval()
 
        hr_list = []
        hr2_list = []
        ndcg_list = []
        for i , data in enumerate(test_loader): 
            with torch.no_grad():    
                if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
                    user, item, image_f, text_f, label = data
                    user, item, image_f, text_f, label = user.squeeze(0), item.squeeze(0), image_f.squeeze(0), text_f.squeeze(0), label.squeeze(0)
                    user, item, image_f, text_f, label = user.cuda(dist.get_rank()), item.cuda(dist.get_rank()), image_f.cuda(dist.get_rank()), text_f.cuda(dist.get_rank()), label.cuda(dist.get_rank())
                elif 'image' in kwargs.keys():
                    user, item, image_f, label = data
                    user, item, image_f, label = user.squeeze(0), item.squeeze(0), image_f.squeeze(0), label.squeeze(0)
                    user, item, image_f, label = user.cuda(dist.get_rank()), item.cuda(dist.get_rank()), image_f.cuda(dist.get_rank()), label.cuda(dist.get_rank())
                    text_f = None
                elif 'text' in kwargs.keys():
                    user, item, text_f, label = data
                    user, item, text_f, label = user.squeeze(0), item.squeeze(0), text_f.squeeze(0), label.squeeze(0)
                    user, item, text_f, label = user.cuda(dist.get_rank()), item.cuda(dist.get_rank()), text_f.cuda(dist.get_rank()), label.cuda(dist.get_rank())
                    image_f = None
                else:
                    user, item, label = data
                    user, item, label = user.squeeze(0), item.squeeze(0), label.squeeze(0)
                    user, item, label = user.cuda(dist.get_rank()), item.cuda(dist.get_rank()), label.cuda(dist.get_rank())
                    image_f = None
                    text_f = None
                
                score = model(user, item, image=image_f, text=text_f)
                pos_idx = label.nonzero()
                _, indices = torch.topk(score, self.top_k)
                recommends = torch.take(item, indices).cpu().numpy()
                gt_item = item[pos_idx].cpu().numpy()
                performance = get_performance(gt_item, recommends.tolist())
                performance = torch.tensor(performance).cuda(dist.get_rank())

                rd_hr = reduce_tensor(performance[0].data, dist.get_world_size())
                rd_hr2 = reduce_tensor(performance[1].data, dist.get_world_size())
                rd_ndcg = reduce_tensor(performance[2].data, dist.get_world_size())

                # hr_list.append(performance[0])
                # hr2_list.append(performance[1])
                # ndcg_list.append(performance[2])

                hr_list.append(rd_hr.item())
                hr2_list.append(rd_hr2.item())
                ndcg_list.append(rd_ndcg.item())

        if dist.get_rank() == 0:
            hit_ratio, hit_ratio2, ndcg = np.mean(hr_list), np.mean(hr2_list), np.mean(ndcg_list)
            print('[Evluating Epoch {}] HR = {:.4f}, HR2 = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, hit_ratio2, ndcg))
            
            t1 = time.time()
            print("evaluate time:", t1 - t0)  
        return hit_ratio, hit_ratio2, ndcg
