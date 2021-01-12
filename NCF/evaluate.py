import torch
from metrics import MetronAtK

class Engine(object):
    def __init__(self):
        self._metron = MetronAtK(top_k=10)
        
    def evaluate(self, model,evaluate_data, epoch_id,**kwargs):
        #Evaluate model
        model.eval()
        if kwargs["feature"] == True:
            image_tp=[]
            image_tn=[]
            text_tp=[]
            text_tn=[]

        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()

            # Feature O
            if kwargs["feature"] == True:
                if kwargs["data"] == "amazon":
                    user2review = kwargs["text"]
                    item2image = kwargs["image"]
                    image_feature = kwargs["image_feature"]
                    text_feature = kwargs["text_feature"]
                    print("Test User Start")
                    for u in test_users:
                        text_vector = user2review[u.item()]
                        text_tp.append(text_feature.infer_vector([text_vector]))   
                    print("Negative User Start")
                    for nu in negative_users:
                        text_vector = user2review[nu.item()]
                        text_tn.append(text_feature.infer_vector([text_vector]))               
                    print("Test Item Start")
                    for i in test_items:
                        image_vector = item2image[i.item()]
                        image_tp.append(image_feature[image_vector]) 
                    print("Test Negative Start")
                    for ni in negative_items :
                        image_vector = item2image[ni.item()]
                        image_tn.append(image_feature[image_vector]) # CPU error
                
                # Movie
                else: 
                    image_feature = kwargs["image_feature"]
                    text_feature = kwargs["text_feature"]
                    for i in test_items :
                        image_tp.append(image_feature[i.item()])   
                        text_tp.append(text_feature[i.item()]) 
                
                    for j in negative_items :  
                        image_tn.append(image_feature[j.item()])   
                        text_tn.append(text_feature[j.item()])  
            
                
 


                image_tn = torch.FloatTensor(image_tn)
                image_tn = image_tn.cuda()
                image_tp = torch.FloatTensor(image_tp)
                image_tp = image_tp.cuda()
                text_tn = torch.FloatTensor(text_tn)
                text_tn = text_tn.cuda()
                text_tp = torch.FloatTensor(text_tp)
                text_tp = text_tp.cuda()

                test_scores = model(test_users, test_items,image_tp,text_tp)
                negative_scores = model(negative_users, negative_items,image_tn,text_tn)

            # Feature X
            else:
                test_scores = model(test_users, test_items)
                negative_scores = model(negative_users, negative_items)
            #to cpu
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
