import torch

def my_collate_tst_2(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    image = [element for items in batch for element in items[2]]
    if type(image[0]) == type(torch.Tensor([])):
        image = torch.stack(image)
    else:
        image = torch.Tensor(image)
    text = [items[3] for items in batch]
    text = torch.Tensor(text)
    label = [items[4] for items in batch]
    label = torch.Tensor(label)
    
    return [user, item, image, text, label]

def my_collate_tst_1(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    feature = [element for items in batch for element in items[2]]
    if type(feature[0]) == type(torch.Tensor([])):
        feature = torch.stack(feature)
    else:
        feature = torch.Tensor(feature)
    label = [items[3] for items in batch]
    label = torch.Tensor(label)

    return [user, item, feature, label]

def my_collate_tst_0(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    label = [items[2] for items in batch]
    label = torch.Tensor(label)
    
    return [user, item, label]

def my_collate_trn_2(batch):
    user = [element for items in batch for element in items[0]]
    user = torch.LongTensor(user)
    item = [element for items in batch for element in items[1]]
    item = torch.LongTensor(item)
    rating = [element for items in batch for element in items[2]]
    rating = torch.FloatTensor(rating)
    image = [element for items in batch for element in items[3]]
    if type(image[0]) == type(torch.Tensor([])):
        image = torch.stack(image)
    else:
        image = torch.Tensor(image)
    text = [element for items in batch for element in items[4]]
    text = torch.Tensor(text)
    return [user, item, rating, image, text]

def my_collate_trn_1(batch):
    user = [element for items in batch for element in items[0]]
    user = torch.LongTensor(user)
    item = [element for items in batch for element in items[1]]
    item = torch.LongTensor(item)
    rating = [element for items in batch for element in items[2]]
    rating = torch.FloatTensor(rating)
    feature = [element for items in batch for element in items[3]]
    if type(feature[0]) == type(torch.Tensor([])):
        feature = torch.stack(feature)
    else:
        feature = torch.Tensor(feature)
    return [user, item, rating, feature]

def my_collate_trn_0(batch):
    user = [element for items in batch for element in items[0]]
    user = torch.LongTensor(user)
    item = [element for items in batch for element in items[1]]
    item = torch.LongTensor(item)
    rating = [element for items in batch for element in items[2]]
    rating = torch.FloatTensor(rating)
    return [user, item, rating]
