import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
 


class ACF(nn.Module):
    def __init__(self,num_user,num_item,images,embd_dim):
        super(ACF,self).__init__()
        
        self.dim = embd_dim
        self.user_embedding = nn.Embedding(num_user,embd_dim)
        self.item_j_embedding = nn.Embedding(num_item,embd_dim)
        self.item_k_embedding = nn.Embedding(num_item,embd_dim)
        self.item_p_embedding = nn.Embedding(num_item,embd_dim)
        
        # Pretrained Model
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*(list(self.resnet18.children())[0:6]))

        # Component Conv2d
        self.feature_conv1 = nn.Conv2d(in_channels=128, out_channels=embd_dim,kernel_size=1)
        self.feature_conv2 = nn.Conv2d(in_channels=embd_dim, out_channels=1,kernel_size=1)
        
        # Item Attention Conv2d
        self.feature_conv3 = nn.Conv2d(in_channels=128, out_channels=embd_dim,kernel_size=1)
        self.feature_conv4 = nn.Conv2d(in_channels=embd_dim, out_channels=1,kernel_size=1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_j_embedding.weight, std=0.01)
        nn.init.normal_(self.item_k_embedding.weight, std=0.01)
        nn.init.normal_(self.item_p_embedding.weight, std=0.01)

    
    def forward(self,user_indices,item_j_indices,item_k_indices,positive_indices,imgs,num_sam):    
        user = self.user_embedding(user_indices) 
        user = torch.squeeze(user,1)
        #print("user shape : ",user.shape) # [batch size x dim]
        item_j = self.item_j_embedding(item_j_indices)
        item_j = torch.squeeze(item_j,1)
        #print("item j shape : ", item_j.shape) # [batch size x dim]
        item_k = self.item_k_embedding(item_k_indices)
        item_k = torch.squeeze(item_k,1)
        #print("item k shape : ", item_k.shape) # [batch size x dim]
        item_p = []
        for nn in range(num_sam):
            item_p_subset = positive_indices[:,nn]
            item_p_subset = self.item_p_embedding(item_p_subset)
            item_p.append(item_p_subset)
        item_p = torch.stack(item_p,dim=1)
        #print("item p shape : ", item_p.shape) # [batch size x p x dim] : p = Sampling in Positive
        
        user = user.view(-1,self.dim,1,1)
        item_j = item_j.view(-1,self.dim,1,1)
        
        
        # Component Attention
        xl_bars = []
        for n in range(num_sam):
            img = imgs[:,n,:,:,:] # imgs [batch size x p x channel x height x width]
            img = torch.squeeze(img,1) # [batch size x channel x height x width]
            img = self.features(img) # [batch size x channel x height x width]

            component = self.feature_conv1(img) # [batch size x dim x height x width]
            component += user
            component = F.relu(component)
            component = self.feature_conv2(component) # [batch size x 1 x height x width]
            component = component.view(-1,1,28*28) 
            component_weight = F.softmax(component,dim=-1) # [batch size x 1 x feature]
            
            img = img.view(-1,128,28*28) # [batch size x channel x feature]
            xl_bar = torch.sum(img * component_weight,dim=-1) 
            xl_bar = xl_bar.view(-1,128,1,1) 
            xl_bars.append(xl_bar)

        
        # Item Attention
        items = []
        for ns in range(num_sam):
            xl_bar = xl_bars[ns]
            p_vec = item_p[:,ns,:]
            p_vec = torch.squeeze(p_vec,1)
            p_vec = p_vec.view(-1,self.dim,1,1)
            item = self.feature_conv3(xl_bar)
         
            item += user
            item += item_j
            item += p_vec
        
            item = F.relu(item)
            item = self.feature_conv4(item) 
        
            item = item.view(-1,1)
      
            item_weight = F.softmax(item,dim=1) # [batch size x 1]
  
            items.append(item_weight)
        items = torch.cat(items,1)
        items = items.unsqueeze(-1)
    
        attention = torch.mul(item_p,items) # [batch size x dim]

        attention = torch.sum(attention,1)
    
        user = user.view(-1,self.dim)
        item_j = item_j.view(-1,self.dim)
        new_user = torch.add(user,attention) # [batch size x dim]
    
        score_j = torch.mul(item_j, new_user)
       
        score_j = torch.sum(score_j,-1) # [batch size]
  
        
        score_k = torch.mul(item_k, new_user)
        score_k = torch.sum(score_k,-1) # [batch size]
        
        return score_j, score_k
        