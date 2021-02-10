import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, dropout, num_layers, feature, **kwargs):
        super(NeuralCF,self).__init__()
        self.feature = feature
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)        
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)
        
        self.feature_extract = resnet_tv.resnet18()
        if ('image' in kwargs.keys()):
            print("IMAGE FEATURE")
            if self.feature == 'raw':
                self.feature_extract.load_state_dict(torch.load(kwargs['extractor_path']))    
            self.feature_extract.eval()
            for param in self.feature_extract.parameters():
                param.requires_grad = False
            self.image_embedding = nn.Linear(kwargs["image"], embedding_size) 
        if ('text' in kwargs.keys()):
            print("TEXT FEATURE")
            self.text_embedding = nn.Linear(kwargs["text"], embedding_size)
 
        #mlp module 
        MLP_modules = []
        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            print("MLP FEATURE 2")
            for i in range(num_layers):
                input_size = 4 * embedding_size // (2 ** i)
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.BatchNorm1d(input_size // 2))
                MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.Dropout(p=dropout))
        elif ('image' in kwargs.keys()) | ('text' in kwargs.keys()):
            print("MLP FEATURE 1")
            for i in range(num_layers):
                input_size = 3 * embedding_size // (2 ** i)
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.BatchNorm1d(input_size // 2))
                MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.Dropout(p=dropout))
        else:
            print("MLP FEATURE 0")
            for i in range(num_layers):
                input_size = 2 * embedding_size // (2 ** i)  
                MLP_modules.append(nn.Linear(input_size, input_size//2))
                MLP_modules.append(nn.BatchNorm1d(input_size // 2))
                MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.Dropout(p=dropout))

        self.MLP_layers =nn.Sequential(*MLP_modules)
        
        # Predict layer
        if ('image' in kwargs.keys()) & ('text' in kwargs.keys()):
            self.predict_layer = nn.Linear(embedding_size + 4 * (2 * int(embedding_size)) // (int(2 ** (num_layers - 1))) // 4, 1)
        elif ('image' in kwargs.keys()) | ('text' in kwargs.keys()):
            self.predict_layer = nn.Linear(embedding_size + 3 * (2 * int(embedding_size)) // (int(2 ** (num_layers - 1))) // 4, 1)
        else:
            self.predict_layer = nn.Linear(embedding_size + (int(embedding_size)) // (int(2 ** (num_layers - 1))), 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
    def forward(self,user_indices,item_indices,**kwargs):
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        
        if kwargs['image'] is not None:
            if self.feature == 'raw':
                _, image = self.feature_extract.feature_list(kwargs['image'])
                # import pdb;pdb.set_trace()
                image = F.relu(self.image_embedding(image[5]))
            else:
                image = F.relu(self.image_embedding(kwargs['image']))
            item_mlp = torch.cat([item_mlp,image], -1)
        if kwargs['text'] is not None:
            text = F.relu(self.text_embedding(kwargs["text"]))
            item_mlp = torch.cat([item_mlp,text], -1)
        
        gmf = torch.mul(user_gmf, item_gmf)
        mlp = torch.cat([user_mlp, item_mlp], -1)
        mlp = self.MLP_layers(mlp)
        x = torch.cat((gmf, mlp), 1)
        x = self.predict_layer(x)
        return x.view(-1)
