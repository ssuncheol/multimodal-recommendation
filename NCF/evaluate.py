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
            test_score=[]
            negative_Score=[]
            for batch1,batch2 in zip(test_loader,cycle(test_negative_loader)):
                with torch.no_grad():
             
                        test_users, test_items , image1 = batch1
                        negative_users, negative_items,image2 = batch2
                        
                        test_users = test_users.cuda()
                        test_items = test_items.cuda()
                        negative_users = negative_users.cuda()
                        negative_items = negative_items.cuda()
                        
                        image1 =image1.cuda()
                        image2 = image2.cuda()
                        test_scores = model(test_users, test_items,image=image1)
                        negative_scores = model(negative_users, negative_items,image=image2)
                        b=time.time()
                        print("time:",b-a)
                        import pdb; pdb.set_trace()
      
            
            
                        
                        test_users = test_users.cpu()
                        test_items = test_items.cpu()
                        test_scores = test_scores.cpu()
                        negative_users = negative_users.cpu()
                        negative_items = negative_items.cpu()
                        negative_scores = negative_scores.cpu()
                        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                            test_items.data.view(-1).tolist(),
                                            test_scores.data.view(-1).tolist(),
                                            negative_users.data.view(-1).tolist(),
                                            negative_items.data.view(-1).tolist(),
                                            negative_scores.data.view(-1).tolist()]
                hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
                print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, ndcg))
                return hit_ratio, ndcg
                
            
