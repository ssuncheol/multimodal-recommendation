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

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_j_embedding.weight, std=0.01)
        nn.init.normal_(self.item_k_embedding.weight, std=0.01)
        nn.init.normal_(self.item_p_embedding.weight, std=0.01)

    
    def forward(self,user_indices,item_j_indices,item_k_indices,positive_indices,imgs,num_sam):    
        user = self.user_embedding(user_indices) 
        #print("user shape : ",user.shape) # [batch size x dim]
        item_j = self.item_j_embedding(item_j_indices)
        #print("item j shape : ", item_j.shape) # [batch size x dim]
        item_k = self.item_k_embedding(item_k_indices)
        #print("item k shape : ", item_k.shape) # [batch size x dim]
        item_p = self.item_p_embedding(positive_indices)
        #print("item p shape : ", item_p.shape) # [batch size x p x dim]
        user = user.view(-1,self.dim,1,1)
        
        xl_bars = []
        for n in range(num_sam):
            img = imgs[:,n,:,:,:]
            img = torch.squeeze(img,1)
            img = self.features(img) 
            #print("img shape : ", img.shape) # [batch size x channel x H x W]
            component = self.feature_conv1(img)
            component += user
            component = F.relu(component)
            component = self.feature_conv2(component)
            component = component.view(-1,1,28*28) 
            component_weight = F.softmax(component,dim=-1)
            
     
            img = img.view(-1,128,28*28) # [batch size x p x channel x f]
            xl_bar = torch.sum(img * component_weight,dim=-1)
            xl_bar = xl_bar.view(-1,128,1,1) 
            xl_bars.append(xl_bar)
 
        items = []
        for ns in range(num_sam):
            xl_bar = xl_bars[ns]
            item = self.feature_conv3(xl_bar)
            #print("item shape : ", item.shape)
            item += user
        
            item = F.relu(item)
            item = self.feature_conv4(item) 
            #print("item shape : ", item.shape)
            item = item.view(-1,1)
            #print("item shape : ",item.shape)
            item_weight = F.softmax(item,dim=1) # [batch size x 1]
            #print("item weight shape : ",item_weight.shape)
            items.append(item_weight)
        items = torch.cat(items,1)
        items = items.unsqueeze(-1)
        #print("items shape",items.shape)
        attention = torch.mul(item_p,items) # [batch size x dim]
        #print("Attention shape",attention.shape)
        attention = torch.sum(attention,1)
        #print("Attention shape",attention.shape)
        user = user.view(-1,self.dim)
        new_user = torch.add(user,attention) # [batch size x dim]
        #print("New User shape : ",new_user.shape)
        score_j = torch.mul(item_j, new_user)
        #print("Score shape : {}".format(score_j.shape))
        score_j = torch.sum(score_j,-1) # [batch size]
        #print("Score shape : {}".format(score_j.shape))
        
        score_k = torch.mul(item_k, new_user)
        score_k = torch.sum(score_k,-1) # [batch size]
        
        return score_j, score_k
        