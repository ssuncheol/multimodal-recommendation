import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_tv as resnet



class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


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
        # print()
        # print('after negative: ', attention[0][2])
        # print('after positive: ', attention[0][0])
        # Distance
        # dist = torch.norm(torch.mul(attention, p_u)-torch.mul(attention,q_i),dim=-1, p=2)
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


def modified_softmax(x):
    exp = torch.exp(x)
    if len(exp.shape) > 2:
        sum = torch.sum(x, axis=-1).reshape(x.shape[0], x.shape[1], -1)
    else:
        sum = torch.sum(x, axis=-1).reshape(x.shape[0], -1)
    return exp / sum
