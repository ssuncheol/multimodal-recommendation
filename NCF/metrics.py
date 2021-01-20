import math
import pandas as pd
import numpy as np

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        
        # rank the items according to the scores for each user 
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return test_in_top_k['user'].nunique() * 1.0 / full['user'].nunique()
    
    def cal_hit_ratio2(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        user_pos_item_num_dict = {}
        for i in full['user'].unique():
            user_pos_item_num_dict[i] = full[full['user']==i]['test_item'].nunique()
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        test_in_top_k = test_in_top_k.groupby(by=['user'], as_index=False).count()
        hit_ratio = []
        for i in test_in_top_k['user']:
            hit_ratio.append(test_in_top_k[test_in_top_k['user'] == i]['item'].item() / user_pos_item_num_dict[i])
        test_in_top_k['hit_ratio'] = hit_ratio
        return test_in_top_k['hit_ratio'].sum() * 1.0 / full['user'].nunique()
 
    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        
        test_in_top_k['dcg'] = test_in_top_k['rank'].apply(lambda x: 1 / math.log2(1 + x))
        count = list(test_in_top_k.groupby(by=['user'], as_index=False).count()['item'])
        idcg_list = []
        for i in count:
            idcg = 0.0
            for j in range(i):
                idcg += np.reciprocal(np.log2(j+2))
            idcg_list.append(idcg)
        test_in_top_k = test_in_top_k.groupby(by=['user'], as_index=False).sum()
        test_in_top_k['idcg'] = idcg_list
        test_in_top_k['ndcg'] = test_in_top_k['dcg'] * 1.0 / test_in_top_k['idcg']

        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
