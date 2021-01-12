import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    def __init__(self,num_users,num_items,embedding_size,num_layers,**kwargs):
        super(NeuralCF,self).__init__()
    
        self.user_embedding_gmf = nn.Embedding(num_users,embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items,embedding_size)        
        self.user_embedding_mlp = nn.Embedding(num_users,embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items,embedding_size)

        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            print("IMAGE FEATURE")
            print("TEXT FEATURE")
            self.image_embedding = nn.Linear(kwargs["image"],embedding_size) # Image Embedding
            self.text_embedding = nn.Linear(kwargs["text"],embedding_size) # Text Embedding
 

        else:
            print("NOT FEATURE")
            

        #mlp module 
        MLP_modules = []
        for i in range(num_layers):
            if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
                print("MLP IMAGE TEXT")
                input_size = 4*embedding_size//(2**i)

            else:
                print("MLP NOT FEATURE")
                input_size = 2*embedding_size//(2**i)  
            MLP_modules.append(nn.Linear(input_size,input_size//2))
            MLP_modules.append(nn.ReLU())

        self.MLP_layers =nn.Sequential(*MLP_modules)
        
        # Predict layer
        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            self.predict_layer = nn.Linear(embedding_size + (2*int(embedding_size))//(int(2**(num_layers-1))),1)
        else:
            self.predict_layer = nn.Linear(embedding_size + (int(embedding_size))//(int(2**(num_layers-1))),1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
    

        
    def forward(self,user_indices,item_indices,**kwargs):
        user_gmf=self.user_embedding_gmf(user_indices)
        item_gmf=self.item_embedding_gmf(item_indices)
        
        user_mlp=self.user_embedding_mlp(user_indices)
        item_mlp=self.item_embedding_mlp(item_indices)
        
        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            image=F.relu(self.image_embedding(kwargs["image"]))
            text=F.relu(self.text_embedding(kwargs["text"]))
            item_mlp = torch.cat([item_mlp,image,text], -1)
        
        gmf=torch.mul(user_gmf,item_gmf) 
        mlp=torch.cat([user_mlp,item_mlp],-1)
        mlp = self.MLP_layers(mlp)
        x=torch.cat((gmf,mlp),1)
        x=self.predict_layer(x)
        return x.view(-1)