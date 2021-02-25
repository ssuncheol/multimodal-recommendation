import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv as resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, dropout, num_layers, feature_data_type, **kwargs):
        super(NeuralCF,self).__init__()
        self.feature_data_type = feature_data_type
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)        
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)
        
        if (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'all'):
            self.feature_extractor = resnet_tv.resnet18()
            print("IMAGE FEATURE")
            if self.feature_data_type == 'raw':
                # map_location = {'cuda:%d' % 0: 'cuda:%d' % kwargs['rank']}
                self.feature_extractor.load_state_dict(torch.load(kwargs['extractor_path'], map_location='cuda:%d' % kwargs['rank']))    
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.image_embedding = nn.Linear(512, embedding_size) 
        if (kwargs['feature_type'] == 'txt') | (kwargs['feature_type'] == 'all'):
            print("TEXT FEATURE")
            self.text_embedding = nn.Linear(kwargs["text"], embedding_size)
 
        # mlp module 
        MLP_modules = []
        if kwargs['feature_type'] == 'all':
            print("MLP FEATURE 2")
            for i in range(num_layers):
                input_size = 4 * embedding_size // (2 ** i)
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.BatchNorm1d(input_size // 2))
                MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.Dropout(p=dropout))
        elif (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'txt'):
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
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.BatchNorm1d(input_size // 2))
                MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.Dropout(p=dropout))

        self.MLP_layers =nn.Sequential(*MLP_modules)
        
        # Predict layer
        if kwargs['feature_type'] == 'all':
            self.predict_layer = nn.Linear(embedding_size + 4 * (2 * int(embedding_size)) // (int(2 ** (num_layers - 1))) // 4, 1)
        elif (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'txt'):
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
        
        if (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'all'):
            if self.feature_data_type == 'raw':
                _, image = self.feature_extractor.feature_list(kwargs['image'])
                # import pdb;pdb.set_trace()
                image = F.relu(self.image_embedding(image[5]))
            else:
                image = F.relu(self.image_embedding(kwargs['image']))
            item_mlp = torch.cat([item_mlp,image], -1)
        if (kwargs['feature_type'] == 'txt') | (kwargs['feature_type'] == 'all'):
            text = F.relu(self.text_embedding(kwargs["text"]))
            item_mlp = torch.cat([item_mlp,text], -1)
        
        gmf = torch.mul(user_gmf, item_gmf)
        mlp = torch.cat([user_mlp, item_mlp], -1)
        mlp = self.MLP_layers(mlp)
        x = torch.cat((gmf, mlp), 1)
        x = self.predict_layer(x)
        return x.view(-1)

class MAML(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, dropout_rate, feature_type, t_feature_dim, v_feature_extractor_path, rank):
        super(MAML, self).__init__()
        self.embed_dim = embed_dim
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.feature_type = feature_type
        self.t_feature_dim = t_feature_dim

        # Embedding Layers
        self.embedding_user = nn.Embedding(n_users, embed_dim, max_norm=1.0)
        self.embedding_item = nn.Embedding(n_items, embed_dim, max_norm=1.0)

        # Image feature extractor module
        self.v_feature_extractor = resnet.resnet18()
        self.v_feature_dim = self.v_feature_extractor.fc.in_features
        
        if v_feature_extractor_path is not None:
            self.v_feature_extractor.load_state_dict(torch.load(v_feature_extractor_path, map_location='cuda:%d' % rank))
        self.v_feature_extractor.eval()
        for param in self.v_feature_extractor.parameters():
            param.requires_grad = False

        # Feature Fusion Layers
        """
        Feature dim -> embed_dim
        """
        if self.feature_type == "rating":
            self.feature_fusion = None
            attention_num = 2

        else :
            attention_num = 3
            if self.feature_type == "all":
                input_dim = self.t_feature_dim + self.v_feature_dim
            elif self.feature_type == "txt":
                input_dim = self.t_feature_dim
            elif self.feature_type == "img":
                input_dim = self.v_feature_dim

            hidden_dim = 256
            modules = []
            for i in range(3):
                if i == 2:
                    hidden_dim *= 2
                modules.append(NormalizeLayer())
                modules.append(nn.Linear(input_dim, hidden_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout_rate))
                input_dim = hidden_dim
                hidden_dim //= 2
            modules.append(nn.Linear(hidden_dim * 2, embed_dim))
            modules.append(NormalizeLayer())

            self.feature_fusion = nn.Sequential(*modules)


        # Attention
        self.attention = nn.Sequential(
            NormalizeLayer(),
            nn.Linear(attention_num * embed_dim, attention_num * embed_dim),
            nn.Tanh(),
            nn.Dropout(0.05),
            NormalizeLayer(),
            nn.Linear(attention_num * embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, user, item, t_feature, image):
        # Embed user, item
        p_u = self.embedding_user(user)
        q_i = self.embedding_item(item)

        # Unsqueeze for negative pairs
        if len(item.size())==2:
            p_u = p_u.unsqueeze(1).expand(-1, q_i.shape[1], -1)
            if self.feature_type == "img" or self.feature_type == "all":
                image = image.view(image.size(0)*image.size(1),*(image.size()[2:]))

        # Extract image feature
        if self.feature_type == "img":
            _, v_feature = self.v_feature_extractor.feature_list(image)
            v_feature = v_feature[5]
            if len(item.size())==2:
                v_feature = v_feature.reshape(q_i.size(0),q_i.size(1),-1)
            item_feature = v_feature
        elif self.feature_type == "all":
            _, v_feature = self.v_feature_extractor.feature_list(image)
            v_feature = v_feature[5]
            if len(item.size())==2:
                v_feature = v_feature.reshape(q_i.size(0),q_i.size(1),-1)
            item_feature = torch.cat((v_feature,t_feature), axis=-1)
        elif self.feature_type == "txt":
            item_feature = t_feature
        else:
            item_feature = None

        # Fused feature
        if item_feature is not None:
            q_i_feature = self.feature_fusion(item_feature)
        else:
            q_i_feature = None

        # Attention
        if self.feature_type != "rating":
            input_cat = torch.cat((p_u, q_i, q_i_feature), axis=-1)
        else:
            input_cat = torch.cat((p_u, q_i), axis=-1)
        attention = self.attention(input_cat)
        # attention = self.embed_dim * modified_softmax(attention)
        attention = self.embed_dim * F.softmax(attention, dim=-1)

        # Distance
        # dist = torch.norm(torch.mul(attention, p_u)-torch.mul(attention,q_i),dim=-1, p=2)
        temp = torch.mul(attention, p_u) - torch.mul(attention, q_i)
        dist = torch.sum(temp ** 2, axis=-1)

        return p_u, q_i, q_i_feature, dist


class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    def forward(self, _input):
        indicator = torch.norm(_input,dim=-1,p=2)>1.0
        norm = F.normalize(_input, dim=-1, p=2)
        indicator = indicator.unsqueeze(-1)
        result = _input * ~indicator + norm * indicator
        return result


def modified_softmax(x):
    exp = torch.exp(x)
    if len(exp.shape) > 2:
        sum = torch.sum(x, axis=-1).reshape(x.shape[0], x.shape[1], -1)
    else:
        sum = torch.sum(x, axis=-1).reshape(x.shape[0], -1)
    return exp / sum
