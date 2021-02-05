class UserItemTrainDataset(Dataset):
    def __init__(self, df_train_p, df_train_n, num_neg, **kwargs):
        self.users = np.array(df_train_p['userid'])
        self.items = np.array(df_train_p['train_pos'])
        self.ratings = np.repeat(1, len(df_train_p['userid'])).reshape(-1)
        self.image_dict = None
        self.text_dict = None
        self.df_train_n = df_train_n
        self.df_train_p = df_train_p
        self.num_neg = num_neg
        if 'image' in kwargs.keys() :
            self.image_dict = kwargs['image']
        if 'text' in kwargs.keys():
            self.text_dict = kwargs['text']

    def __getitem__(self, index):
        negative_users = np.array(np.repeat(self.users[index], self.num_neg))
        negative_items = random.sample(list(self.df_train_n[self.df_train_n['userid'] == self.users[index]]["train_negative"].item()), self.num_neg)
        negative_ratings = np.repeat(0, self.num_neg)
        negative_img = []
        negative_txt = []

        if (self.image_dict is not None) & (self.text_dict is not None):
            for i in negative_items:
                negative_img.append(self.image_dict[self.items[i].item()])
                negative_txt.append(self.text_dict[self.items[i].item()])
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), negative_img, np.concatenate((negative_txt, [self.text_dict[self.items[index].item()]]))
        elif self.image_dict is not None:
            for i in negative_items:
                negative_img.append(self.image_dict[self.items[i].item()])
            negative_img.append(self.image_dict[self.items[index].item()])
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]])), negative_img
        elif self.text_dict is not None:
            for i in negative_items:
                negative_txt.append(self.text_dict[self.items[i].item()])
            return torch.cat((torch.LongTensor([self.users[index]]), torch.LongTensor(negative_users))), torch.cat((torch.LongTensor([self.items[index]]), torch.LongTensor(negative_items))), torch.cat((torch.FloatTensor([self.ratings[index]]), torch.FloatTensor(negative_ratings))), torch.cat((torch.FloatTensor(self.text_dict[self.items[index].item()]).unsqueeze(0), torch.FloatTensor(negative_txt)))
        else:
            return np.concatenate((negative_users, [self.users[index]])), np.concatenate((negative_items, [self.items[index]])), np.concatenate((negative_ratings, [self.ratings[index]]))
        
    def __len__(self):
        return len(self.users)
