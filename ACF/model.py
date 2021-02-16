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
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*(list(self.resnet18.children())[0:6]))
        
        self.feature_conv1 = nn.Conv2d(in_channels=128, out_channels=embd_dim,kernel_size=1)
        self.feature_conv2 = nn.Conv2d(in_channels=embd_dim, out_channels=1,kernel_size=1)
        self.feature_conv3 = nn.Conv2d(in_channels=128, out_channels=embd_dim,kernel_size=1)
        self.feature_conv4 = nn.Conv2d(in_channels=embd_dim, out_channels=1,kernel_size=1)

    
    def forward(self,user_indices,item_j_indices,item_k_indices,positive_indices,imgs):    
        user = self.user_embedding(user_indices) # [batch size x p x dim]
        #print("user shape : ",user.shape)
        item_j = self.item_j_embedding(item_j_indices) # [batch size x dim]
        #print("item j shape : ", item_j.shape)
        item_k = self.item_k_embedding(item_k_indices) # [batch size x dim]
        #print("item k shape : ", item_k.shape)
        item_p = self.item_p_embedding(positive_indices) # [batch size x p x dim]
        #print("item p shape : ", item_p.shape)
        user = user.view(-1,self.dim,1,1)

        #print("imgs shape : ",imgs.shape)
        imgs = self.features(imgs) # [batch size x channel x H x W]
        #print("imgs shape : ", imgs.shape)
        component = self.feature_conv1(imgs) # [batch size x p x dim]
        #print("component shape : ", component.shape)
        component += user
      
        component = F.relu(component)
        component = self.feature_conv2(component)
        #print("component shape : ", component.shape)
        component = component.view(-1,1,28*28) # [batch size x p x f]
        #print("component shape : ", component.shape)
        component_weight = F.softmax(component,dim=-1)
     
        imgs = imgs.view(-1,128,28*28) # [batch size x p x channel x f]
        xl_bar = torch.sum(imgs * component_weight,dim=-1)
        xl_bar = xl_bar.view(-1,128,1,1) 
        #print("xl bar : ", xl_bar.shape)

        item = self.feature_conv3(xl_bar)
        #print("item shape : ", item.shape)
        item += user
        
        item = F.relu(item)
        item = self.feature_conv4(item) 
        #print("item shape : ", item.shape)
        item_weight = item.view(-1,1)
        #print("item weight shape : ",item_weight.shape)
        #item_weight = F.softmax(item,dim=0) # [batch size x 1]

 
        attention = torch.mul(item_p,item_weight) # [batch size x dim]
    
        user = user.view(-1,self.dim)
        new_user = user+attention # [batch size x dim]

   
        score_j = torch.sum(torch.mul(new_user, item_j),1) # [batch size]
        score_k = torch.sum(torch.mul(new_user, item_k),1) # [batch size]
        
        return score_j, score_k
        