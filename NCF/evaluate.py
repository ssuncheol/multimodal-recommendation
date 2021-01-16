import torch
import time
from metrics import MetronAtK
from itertools import cycle

class Engine(object):
    def __init__(self):
        self._metron = MetronAtK(top_k=10)
        
    def evaluate(self, model,test_loader,test_negative_loader, epoch_id,**kwargs):
        #Evaluate model
        a=time.time()
        model.eval()
        if kwargs["feature"] == True:
            t_test_users=[]
            t_negative_users=[]
            t_test_items=[]
            t_negative_items=[]
            test_score=[]
            negative_score=[]
            i=0
            for batch1,batch2 in zip(test_loader,cycle(test_negative_loader)):
                with torch.no_grad():
             
                        test_users, test_items , image1 = batch1
                        negative_users, negative_items,image2 = batch2
                        
                        test_scores = model(test_users, test_items,image=image1)
                        negative_scores = model(negative_users, negative_items,image=image2)

                        test_scores = test_scores.cpu()
                        negative_scores = negative_scores.cpu()

                        t_test_users.extend(test_users.detach().numpy())
                        t_test_items.extend(test_items.detach().numpy())
                        t_negative_users.extend(negative_users.detach().numpy())
                        t_negative_items.extend(negative_items.detach().numpy())
                        test_score.extend(test_scores.detach().numpy())
                        negative_score.extend(negative_scores.detach().numpy())
                        print(i)
                        i+=1
      
            
            #import pdb; pdb.set_trace()
            
            self._metron.subjects = [t_test_users,
                                t_test_items,
                                test_score,
                                t_negative_users,
                                t_negative_items,
                                negative_score]
            hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
            print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, ndcg))
            
            b=time.time()
            print("evaluate time:",b-a)  
            return hit_ratio, ndcg
            
            
