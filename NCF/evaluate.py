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
 
        t_test_users=[]
        t_negative_users=[]
        t_test_items=[]
        t_negative_items=[]
        test_score=[]
        negative_score=[]
        dataloader_iterator = iter(test_loader)
        
        for i , data1 in enumerate(test_negative_loader): 

            try :
                data2 = next(dataloader_iterator)
            except StopIteration: 
                dataloader_iterator = iter(test_loader)
                data2 = next(dataloader_iterator)
                
            with torch.no_grad():
                if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
                    test_users, test_items , image1, text1 = data2
                    negative_users, negative_items, image2, text2 = data1
                    
                    test_scores = model(test_users, test_items,image=image1,text=text1)
                    negative_scores = model(negative_users, negative_items,image=image2,text=text2)
                elif 'image' in kwargs.keys():
                    test_users, test_items , image1 = data2
                    negative_users, negative_items,image2 = data1
                    
                    test_scores = model(test_users, test_items,image=image1)
                    negative_scores = model(negative_users, negative_items,image=image2)                
                elif 'text' in kwargs.keys():
                    test_users, test_items , text1 = data2
                    negative_users, negative_items,text2 = data1
                    
                    test_scores = model(test_users, test_items,text=text1)
                    negative_scores = model(negative_users, negative_items,text=text2)                       

                else:
                    test_users, test_items = data2
                    negative_users, negative_items = data1
                
                    test_scores = model(test_users, test_items)
                    negative_scores = model(negative_users, negative_items)
                    
                test_scores = test_scores.cpu()
                negative_scores = negative_scores.cpu()

                t_test_users.extend(test_users.detach().numpy())
                t_test_items.extend(test_items.detach().numpy())
                t_negative_users.extend(negative_users.detach().numpy())
                t_negative_items.extend(negative_items.detach().numpy())
                test_score.extend(test_scores.detach().numpy())
                negative_score.extend(negative_scores.detach().numpy())
      

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
            


            