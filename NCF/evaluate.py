import torch
import time
from metric import get_performance
import numpy as np

1

class Engine(object):
    def __init__(self, top_k):
        self.top_k = top_k
        
    def evaluate(self, model, test_loader, epoch_id, **kwargs):
        #Evaluate model
        a=time.time()
        if (epoch_id + 1) % kwargs['interval'] != 0:
            return 0, 0, 0
        
        model.eval()

        hr_list = []
        hr2_list = []
        ndcg_list = []
        
        for i , data in enumerate(test_loader): 

            with torch.no_grad():    
                if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
                    user, item, image_f, text_f, label = data
                    user, item, image_f, text_f, label = user.squeeze(0), item.squeeze(0), image_f.squeeze(0), text_f.squeeze(0), label.squeeze(0)
                    user, item, image_f, text_f, label = user.cuda(), item.cuda(), image_f.cuda(), text_f.cuda(), label.cuda()

                    score = model(user, item, image=image_f, text=text_f)
                elif 'image' in kwargs.keys():
                    user, item, image_f, label = data
                    user, item, image_f, label = user.squeeze(0), item.squeeze(0), image_f.squeeze(0), label.squeeze(0)
                    user, item, image_f, label = user.cuda(), item.cuda(), image_f.cuda(), label.cuda()

                    score = model(user, item, image=image_f)
                elif 'text' in kwargs.keys():
                    user, item, text_f, label = data
                    user, item, text_f, label = user.squeeze(0), item.squeeze(0), text_f.squeeze(0), label.squeeze(0)
                    user, item, text_f, label = user.cuda(), item.cuda(), text_f.cuda(), label.cuda()

                    score = model(user, item, text=text_f)
                else:
                    user, item, label = data
                    user, item, label = user.squeeze(0), item.squeeze(0), label.squeeze(0)
                    user, item, label = user.cuda(), item.cuda(), label.cuda()
                    
                    score = model(user, item)
                
                pos_idx = label.nonzero()
                _, indices = torch.topk(score, self.top_k)
                recommends = torch.take(item, indices).cpu().numpy()
                gt_item = item[pos_idx].cpu().numpy()
                performance = get_performance(gt_item, recommends.tolist())
                hr_list.append(performance[0])
                hr2_list.append(performance[1])
                ndcg_list.append(performance[2])
        
        hit_ratio, hit_ratio2, ndcg = np.mean(hr_list), np.mean(hr2_list), np.mean(ndcg_list)
        print('[Evluating Epoch {}] HR = {:.4f}, HR2 = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, hit_ratio2, ndcg))
        
        b=time.time()
        print("evaluate time:",b-a)  
        return hit_ratio, hit_ratio2, ndcg
