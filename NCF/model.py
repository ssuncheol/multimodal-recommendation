import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self,num_users,num_items,embedding_size,num_layers,num_director,num_genre,image,text):
        super(NeuralCF,self).__init__()
    
        self.user_embedding_gmf = nn.Embedding(num_users,embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items,embedding_size)
        self.director_embedding_gmf = nn.Embedding(num_director,embedding_size)
        self.user_embedding_mlp = nn.Embedding(num_users,int(embedding_size*(2**(num_layers-1))))
        self.item_embedding_mlp = nn.Embedding(num_items,int(embedding_size*(2**(num_layers-1))))
        self.director_embedding_mlp = nn.Embedding(num_director,int(embedding_size*(2**(num_layers-1))))

        self.fc1 = nn.Linear(2*embedding_size +num_genre + image + text, ((2*embedding_size +num_genre + image + text)//2))
        self.fc2 = nn.Linear((2*embedding_size +num_genre + image + text)//2, ((2*embedding_size +num_genre + image + text)//4))
        self.fc3 = nn.Linear((2*embedding_size +num_genre + image + text)//4, embedding_size)
        self.fc4 = nn.Linear(int(embedding_size*(2**(num_layers-1))) * 2 +(num_genre) +(image) + (text) , int(embedding_size*(2**(num_layers-1))))
        
        #mlp module 
        MLP_modules = []
        for i in range(num_layers):
            input_size = embedding_size*(2**(num_layers-i))
            MLP_modules.append(nn.Linear(input_size,input_size//2))
            MLP_modules.append(nn.ReLU())

        self.MLP_layers =nn.Sequential(*MLP_modules)
     
        self.predict_layer = nn.Linear(embedding_size*2,1)
   
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.director_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.director_embedding_mlp.weight, std=0.01)
        
    def forward(self,user_indices,item_indices,director,genre,image,text):
        
        x1=self.user_embedding_gmf(user_indices)
        
        x2=self.item_embedding_gmf(item_indices)
                
        x3=self.user_embedding_mlp(user_indices)
        
        x4=self.item_embedding_mlp(item_indices)
        x5=self.director_embedding_gmf(director)
        
        x6=self.director_embedding_mlp(director)
        
        item_gmf = torch.cat([x2, x5,genre,image,text], -1)
        item_mlp = torch.cat([x4, x6,genre,image,text], -1)
        
        
        
        item_feature_gmf = self.fc1(item_gmf)
        item_feature_gmf = self.fc2(item_feature_gmf)
        item_feature_gmf = self.fc3(item_feature_gmf)
        item_feature_mlp = self.fc4(item_mlp)
        element_product=torch.mul(x1,item_feature_gmf)
        element_cat=torch.cat((x3,item_feature_mlp),-1)

        output_MLP = self.MLP_layers(element_cat)
        x=torch.cat((element_product,output_MLP),-1)
        x=self.predict_layer(x)
        return x.view(-1)
    
    