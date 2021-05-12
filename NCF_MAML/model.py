import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv as resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, dropout, num_layers, **kwargs):
        super(NeuralCF,self).__init__()
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)        
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)

        if (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'all'):
            print("IMAGE FEATURE")
            self.v_feature_extractor = resnet_tv.resnet18()
            self.v_feature_extractor.load_state_dict(torch.load(kwargs['extractor_path'], map_location='cuda:%d' % kwargs['rank']))    
            # attention을 위한 field들
            self.v_feature_dim = self.v_feature_extractor.fc.in_features
            self.v_feature_c1 = self.v_feature_extractor.conv1.out_channels
            self.v_feature_c2 = self.v_feature_extractor.layer1[1].conv2.out_channels
            self.v_feature_c3 = self.v_feature_extractor.layer2[1].conv2.out_channels
            self.v_feature_c4 = self.v_feature_extractor.layer3[1].conv2.out_channels
            self.v_feature_c5 = self.v_feature_extractor.layer4[1].conv2.out_channels

            if kwargs['fine_tuning'] == False:
                self.v_feature_extractor.eval()
                for param in self.v_feature_extractor.parameters():
                    param.requires_grad = False
            
            self.image_embedding = []
            self.image_embedding.append(nn.Linear(self.v_feature_extractor.fc.in_features, 256))
            self.image_embedding.append(nn.BatchNorm1d(256))
            self.image_embedding.append(nn.ReLU())
            self.image_embedding.append(nn.Linear(256, 128))
            self.image_embedding.append(nn.BatchNorm1d(128))
            self.image_embedding.append(nn.ReLU())
            self.image_embedding.append(nn.Linear(128, embedding_size))
            self.image_embedding.append(nn.BatchNorm1d(embedding_size))
            self.image_embedding.append(nn.ReLU())
            self.image_embedding = nn.Sequential(*self.image_embedding)
        if (kwargs['feature_type'] == 'txt') | (kwargs['feature_type'] == 'all'):
            print("TEXT FEATURE")
            self.text_embedding = []
            self.text_embedding.append(nn.Linear(kwargs["text"], 150))
            self.text_embedding.append(nn.BatchNorm1d(150))
            self.text_embedding.append(nn.ReLU())
            self.text_embedding.append(nn.Linear(150, 75))
            self.text_embedding.append(nn.BatchNorm1d(75))
            self.text_embedding.append(nn.ReLU())
            self.text_embedding.append(nn.Linear(75, embedding_size))
            self.text_embedding.append(nn.BatchNorm1d(embedding_size))
            self.text_embedding.append(nn.ReLU())
            self.text_embedding = nn.Sequential(*self.text_embedding)

        # mlp module 
        if kwargs['feature_type'] == 'all':
            input_size = 4 * embedding_size
            print("MLP FEATURE 2")
        elif (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'txt'):
            input_size = 3 * embedding_size
            print("MLP FEATURE 1")
        else:
            input_size = 2 * embedding_size 
            print("MLP FEATURE 0")

        if num_layers == 4:
            MLP_dim = [input_size, input_size, input_size//2, input_size//2, embedding_size]
        else:
            MLP_dim = [input_size, input_size, input_size//2, embedding_size]
        MLP_modules = []
        for i in range(num_layers):
            if i == 0:
                MLP_modules.append(nn.Linear(input_size, int(MLP_dim[i+1])))
            else:
                MLP_modules.append(nn.Linear(int(MLP_dim[i]), int(MLP_dim[i+1])))
            MLP_modules.append(nn.BatchNorm1d(int(MLP_dim[i+1])))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=dropout))
                
        self.MLP_layers =nn.Sequential(*MLP_modules)
        
        # Predict layer
        self.predict_layer = nn.Linear(embedding_size * 2, 1)

        self._init_weight_()

        self.conv_key1 = nn.Conv2d(self.v_feature_c1, embedding_size * 2, 1)
        self.conv_key2 = nn.Conv2d(self.v_feature_c2, embedding_size * 2, 1)
        self.conv_key3 = nn.Conv2d(self.v_feature_c3, embedding_size * 2, 1)
        self.conv_key4 = nn.Conv2d(self.v_feature_c4, embedding_size * 2, 1)
        self.conv_key5 = nn.Conv2d(self.v_feature_c5, embedding_size * 2, 1)

        self.conv_value1 = nn.Conv2d(self.v_feature_c1, self.v_feature_dim, 1)
        self.conv_value2 = nn.Conv2d(self.v_feature_c2, self.v_feature_dim, 1)
        self.conv_value3 = nn.Conv2d(self.v_feature_c3, self.v_feature_dim, 1)
        self.conv_value4 = nn.Conv2d(self.v_feature_c4, self.v_feature_dim, 1)
        self.cnov_value5 = nn.Conv2d(self.v_feature_c5, self.v_feature_dim, 1)

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
    
    def hierarchical_attention(self, v_feature_extractor, key_modules, value_modules, image, user_embedding, item_embedding):
        if len(user_embedding.size()) == 3:
            user_embedding = user_embedding.reshape(user_embedding.size(0) * user_embedding.size(1),
                                                    user_embedding.size(2))
            item_embedding = item_embedding.reshape(item_embedding.size(0) * item_embedding.size(1),
                                                    item_embedding.size(2))
        user_embedding = torch.cat([user_embedding, item_embedding], 1)
        _, feature_map = v_feature_extractor.feature_list(image)
        feature_map = feature_map[:-1]
        score = []
        for i in range(len(feature_map)):
            key = key_modules[i](nn.AvgPool2d(feature_map[i].size(-1))(feature_map[i]))
            score.append(
                torch.diagonal(torch.matmul(key.squeeze(), torch.transpose(user_embedding, 0, 1))).unsqueeze(1))
        score = F.softmax(torch.stack(score).squeeze(), dim=0)

        for i in range(len(feature_map)):
            value = value_modules[i](nn.AvgPool2d(feature_map[i].size(-1))(feature_map[i]))
            if i == 0:
                attention_matrix = torch.mul(score[i].unsqueeze(0).T, value.squeeze())
            else:
                attention_matrix += torch.mul(score[i].unsqueeze(0).T, value.squeeze())

        return attention_matrix
    
    def forward(self, user_indices, item_indices, **kwargs):
        key_modules = [self.conv_key1, self.conv_key2, self.conv_key3, self.conv_key4, self.conv_key5]
        value_modules = [self.conv_value1, self.conv_value2, self.conv_value3, self.conv_value4, self.cnov_value5]

        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)

        if (kwargs['feature_type'] == 'img') | (kwargs['feature_type'] == 'all'):
            if kwargs['hier_attention']:
                image = self.hierarchical_attention(self.v_feature_extractor, key_modules, value_modules, kwargs['image'],
                                                        user_mlp, item_mlp)
                image = self.image_embedding(image)
            else:
                _, image = self.v_feature_extractor.feature_list(kwargs['image'])
                image = self.image_embedding(image[5])
            item_mlp = torch.cat([item_mlp, image], -1)
        if (kwargs['feature_type'] == 'txt') | (kwargs['feature_type'] == 'all'):
            text = self.text_embedding(kwargs["text"])
            item_mlp = torch.cat([item_mlp,text], -1)
        
        gmf = torch.mul(user_gmf, item_gmf)
        mlp = torch.cat([user_mlp, item_mlp], -1)
        mlp = self.MLP_layers(mlp)
        x = torch.cat((gmf, mlp), 1)
        x = self.predict_layer(x)
        return x.view(-1)

class MAML(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, dropout_rate, feature_type, t_feature_dim,
                 v_feature_extractor_path, fine_tuning, rank):
        super(MAML, self).__init__()
        self.embed_dim = embed_dim
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.feature_type = feature_type
        self.t_feature_dim = t_feature_dim
        self.rank = rank

        # Embedding Layers
        self.embedding_user = nn.Embedding(n_users, embed_dim, max_norm=1.0)
        self.embedding_item = nn.Embedding(n_items, embed_dim, max_norm=1.0)

        # Image feature extractor module
        self.v_feature_extractor = resnet.resnet18()
        self.v_feature_dim = self.v_feature_extractor.fc.in_features
        self.v_feature_c1 = self.v_feature_extractor.conv1.out_channels
        self.v_feature_c2 = self.v_feature_extractor.layer1[1].conv2.out_channels
        self.v_feature_c3 = self.v_feature_extractor.layer2[1].conv2.out_channels
        self.v_feature_c4 = self.v_feature_extractor.layer3[1].conv2.out_channels
        self.v_feature_c5 = self.v_feature_extractor.layer4[1].conv2.out_channels

        if v_feature_extractor_path is not None:
            self.v_feature_extractor.load_state_dict(
                torch.load(v_feature_extractor_path, map_location='cuda:%d' % self.rank))

        if fine_tuning is False:
            self.v_feature_extractor.eval()
            for param in self.v_feature_extractor.parameters():
                param.requires_grad = False

        # For attention Layers
        self.conv_key1 = nn.Conv2d(self.v_feature_c1, self.embed_dim*2, 1)
        self.conv_key2 = nn.Conv2d(self.v_feature_c2, self.embed_dim*2, 1)
        self.conv_key3 = nn.Conv2d(self.v_feature_c3, self.embed_dim*2, 1)
        self.conv_key4 = nn.Conv2d(self.v_feature_c4, self.embed_dim*2, 1)
        self.conv_key5 = nn.Conv2d(self.v_feature_c5, self.embed_dim*2, 1)

        self.conv_value1 = nn.Conv2d(self.v_feature_c1, self.v_feature_dim, 1)
        self.conv_value2 = nn.Conv2d(self.v_feature_c2, self.v_feature_dim, 1)
        self.conv_value3 = nn.Conv2d(self.v_feature_c3, self.v_feature_dim, 1)
        self.conv_value4 = nn.Conv2d(self.v_feature_c4, self.v_feature_dim, 1)
        self.cnov_value5 = nn.Conv2d(self.v_feature_c5, self.v_feature_dim, 1)

        # Feature Fusion Layers
        """
        Feature dim -> embed_dim
        """
        if self.feature_type == "rating":
            self.feature_fusion = None
            attention_num = 2

        else:
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
            nn.Linear(attention_num * embed_dim, attention_num * embed_dim),
            nn.Tanh(),
            nn.Linear(attention_num * embed_dim, embed_dim*2),
            nn.Tanh(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.Tanh(),
            nn.Linear(embed_dim*2 , embed_dim)
        )

    def hierarchical_attention(self, v_feature_extractor, key_modules, value_modules, image, user_embedding, item_embedding):
        if len(user_embedding.size()) == 3:
            user_embedding = user_embedding.reshape(user_embedding.size(0) * user_embedding.size(1),
                                                    user_embedding.size(2))
            item_embedding = item_embedding.reshape(item_embedding.size(0) * item_embedding.size(1),
                                                    item_embedding.size(2))
        user_embedding=torch.cat([user_embedding, item_embedding],1)
        _, feature_map = v_feature_extractor.feature_list(image)
        feature_map = feature_map[:-1]
        score = []
        for i in range(len(feature_map)):
            key = key_modules[i](nn.AvgPool2d(feature_map[i].size(-1))(feature_map[i]))
            score.append(
                torch.diagonal(torch.matmul(key.squeeze(), torch.transpose(user_embedding, 0, 1))).unsqueeze(1))
        score = F.softmax(torch.stack(score).squeeze(), dim=0)

        for i in range(len(feature_map)):
            value = value_modules[i](nn.AvgPool2d(feature_map[i].size(-1))(feature_map[i]))
            if i == 0:
                attention_matrix = torch.mul(score[i].unsqueeze(0).T, value.squeeze())
            else:
                attention_matrix += torch.mul(score[i].unsqueeze(0).T, value.squeeze())

        return attention_matrix

    def forward(self, user, item, t_feature, image, hier_attention):
        key_modules = [self.conv_key1, self.conv_key2, self.conv_key3, self.conv_key4, self.conv_key5]
        value_modules = [self.conv_value1, self.conv_value2, self.conv_value3, self.conv_value4, self.cnov_value5]

        # Embed user, item
        p_u = self.embedding_user(user)
        q_i = self.embedding_item(item)
        # Unsqueeze for negative pairs
        if len(item.size()) == 2:
            p_u = p_u.unsqueeze(1).expand(-1, q_i.shape[1], -1)
            if self.feature_type == "img" or self.feature_type == "all":
                image = image.reshape(image.size(0) * image.size(1), *(image.size()[2:]))

        # Extract image feature
        if self.feature_type == "img":
            if hier_attention:
                v_feature = self.hierarchical_attention(self.v_feature_extractor, key_modules, value_modules, image,
                                                        p_u, q_i)
            else:
                _, v_feature = self.v_feature_extractor.feature_list(image)
                v_feature = v_feature[5]
            if len(item.size()) == 2:
                v_feature = v_feature.reshape(q_i.size(0), q_i.size(1), -1)
            item_feature = v_feature

        elif self.feature_type == "all":
            if hier_attention:
                v_feature = self.hierarchical_attention(self.v_feature_extractor, key_modules, value_modules, image,
                                                        p_u, q_i)
            else:
                _, v_feature = self.v_feature_extractor.feature_list(image)
                v_feature = v_feature[5]
            if len(item.size()) == 2:
                v_feature = v_feature.reshape(q_i.size(0), q_i.size(1), -1)
            item_feature = torch.cat((v_feature, t_feature), axis=-1)

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
        attention = self.embed_dim * F.softmax(attention, dim=-1)
        temp = torch.mul(attention, p_u) - torch.mul(attention, q_i)
        dist = torch.sum(temp ** 2, axis=-1)

        return p_u, q_i, q_i_feature, dist


class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, _input):
        indicator = torch.norm(_input, dim=-1, p=2) > 1.0
        norm = F.normalize(_input, dim=-1, p=2)
        indicator = indicator.unsqueeze(-1)
        result = _input * ~indicator + norm * indicator
        return result
