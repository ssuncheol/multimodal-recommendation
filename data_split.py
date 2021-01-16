import os
import pandas as pd
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='data_split')
parser.add_argument('--data_path', default='./movielens_dataset/ratings_movielens.csv', type=str,
                    help='datapath')
parser.add_argument('--save_path', default='./movielens_dataset', type=str,
                    help='savepath')
args = parser.parse_args()
random.seed(1)

ratings=pd.read_csv(args.data_path)
type=['leave-one-out','ratio-split']
# indexing user
user_id=ratings[['userid']].drop_duplicates().reset_index(drop=True)
user_id['useridx']=pd.DataFrame(np.arange(len(user_id)))
userdict=dict(zip(user_id['userid'],user_id['useridx']))
ratings=ratings.replace({'userid':userdict})

#indexing item
item_id=ratings[['itemid']].drop_duplicates().reset_index(drop=True)
item_id['itemidx']=pd.DataFrame(np.arange(len(item_id)))
itemdict=dict(zip(item_id['itemid'],item_id['itemidx']))
ratings=ratings.replace({'itemid':itemdict})

# group items according to user index
items_set=set(item_id['itemidx'])
ratings['rate'][ratings['rate']>0]=1.0
rating1=ratings.groupby('userid')['itemid'].apply(list).reset_index()
rating2=ratings.groupby('userid')['timestamp'].apply(list).reset_index()
ratings=pd.concat([rating1[['userid', 'itemid']], rating2['timestamp']], axis=1)

# split
for split_type in type:
    if split_type=='leave-one-out':
        ratings['test_positive_index']=ratings['timestamp'].apply(lambda x: np.array(x).argmax())
        ratings['test_positive']=ratings.apply(lambda x: x['itemid'][x['test_positive_index']], axis=1)
        ratings['negative']=ratings['itemid'].apply(lambda x: list(items_set - set(x)))
        ratings['test_negative']=ratings['negative'].apply(lambda x: random.sample(x, 99))
        ratings['train_negative']=ratings.apply(lambda x: list(items_set - set(x['itemid']) - set(x['test_negative'])), axis=1)
        ratings.apply(lambda x: x['itemid'].remove(x['test_positive']), axis=1)


    elif split_type=='ratio-split':
        ratings['test_positive']=ratings['itemid'].apply(lambda x: random.sample(x, round(len(x)*0.3)))
        ratings['itemid']=ratings.apply(lambda x: list(set(x['itemid'])-set(x['test_positive'])), axis=1)
        ratings['train_negative']=ratings.apply(lambda x: list(items_set - set(x['itemid'])),axis=1)
        ratings['test_negative']=ratings.apply(lambda x: list(items_set - set(x['itemid'])- set(x['test_positive'])), axis=1)

    ratings.rename(columns = {'itemid':'train_positive'}, inplace = True)
    ratings = ratings[['userid','train_positive','test_positive','train_negative', 'test_negative']].reset_index(drop=True)
    train_positive=ratings.join(ratings['train_positive'].apply(lambda x:pd.Series(x)).stack().reset_index(1,name='train_pos').drop('level_1',axis=1))
    test_positive=ratings.join(ratings['test_positive'].apply(lambda x:pd.Series(x)).stack().reset_index(1,name='test_pos').drop('level_1',axis=1))

    # save
    if not os.path.exists(os.path.join(args.save_path, split_type)):
        os.makedirs(os.path.join(args.save_path, split_type))
    train_positive[['userid','train_pos']].to_csv(os.path.join(args.save_path,split_type, 'train_positive.csv'),index=False)
    test_positive[['userid','test_pos']].to_csv(os.path.join(args.save_path,split_type, 'test_positive.csv'),index=False)
    ratings[['userid','train_negative']].to_csv(os.path.join(args.save_path,split_type, 'train_negative.csv'),index=False)
    ratings[['userid','test_negative']].to_csv(os.path.join(args.save_path,split_type, 'test_negative.csv'),index=False)

if not os.path.exists(os.path.join(args.save_path,'index-info')):
    os.makedirs(os.path.join(args.save_path,'index-info'))
user_id.to_csv(os.path.join(args.save_path,'index-info', 'user_index.csv'),index=False)
item_id.to_csv(os.path.join(args.save_path,'index-info', 'item_index.csv'),index=False)
