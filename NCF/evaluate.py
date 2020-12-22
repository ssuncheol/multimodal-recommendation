import torch
from metrics import MetronAtK

class Engine(object):
    def __init__(self):
        self._metron = MetronAtK(top_k=10)
        
    def evaluate(self, model,evaluate_data,dic_director,one_hot_vector,image,text, epoch_id):
        #Evaluate model
        model.eval()
        director_tp = []
        director_tn = []
        genre_tp=[]
        genre_tn=[]
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
            
            for i in test_items :
                director_tp.append(dic_director[i.item()])
                genre_tp.append(one_hot_vector[i.item()])
                image_tp.append(image[i.item()])   
                text_tp.append(text[i.item()]) 
                
            for j in negative_items :
                director_tn.append(dic_director[j.item()])    
                genre_tn.append(one_hot_vector[j.item()])    
                image_tn.append(image[j.item()])   
                text_tn.append(text[j.item()])  
            
                
            director_tp = torch.LongTensor(director_tp)
            director_tn = torch.LongTensor(director_tn)
            genre_tp = torch.LongTensor(genre_tp)
            genre_tn = torch.LongTensor(genre_tn)
            image_tp = torch.FloatTensor(image_tp)
            image_tn = torch.FloatTensor(image_tn)
            text_tp = torch.FloatTensor(text_tp)
            text_tn = torch.FloatTensor(text_tn)
            
            director_tp , director_tn ,genre_tp,genre_tn,image_tn,image_tp,text_tn,text_tp = director_tp.cuda(),director_tn.cuda(),genre_tp.cuda(),genre_tn.cuda(),image_tn.cuda(),image_tp.cuda(),text_tn.cuda(),text_tp.cuda()
                
            test_scores = model(test_users, test_items,director_tp,genre_tp,image_tp,text_tp)
            negative_scores = model(negative_users, negative_items,director_tn,genre_tn,image_tn,text_tn)
            
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
