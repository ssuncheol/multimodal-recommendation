import math
import pandas as pd
import numpy as np

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on
        self._user_pos_item_num_dict = None
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
        self._user_pos_item_num_dict = {}
        for i in test['user'].unique():
            self._user_pos_item_num_dict[i] = test[test['user']==i]['test_item'].nunique()
        
        iteration = 0
        for i in full['user'].unique():
            sub = full[full['user'] == i]
            sub.sort_values(['score'], inplace=True, ascending=False)    
            if iteration == 0:
                real_full = sub.iloc[:10]
            else:
                real_full = real_full.append(sub.iloc[:10])
            iteration += 1
        
        full = pd.merge(real_full, test, on=['user'], how='left')
        # rank the items according to the scores for each user 
        full['rank'] = full.groupby('user')['score'].rank(method='dense', ascending=False)
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
        user_pos_item_num_dict, full, top_k = self._user_pos_item_num_dict, self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        test_in_top_k = test_in_top_k.groupby(by=['user'], as_index=False).count()
        hit_ratio = []
        for i in test_in_top_k['user']:
            hit_ratio.append(test_in_top_k[test_in_top_k['user'] == i]['item'].item() / min(user_pos_item_num_dict[i], 10))
        test_in_top_k['hit_ratio'] = hit_ratio
        return test_in_top_k['hit_ratio'].sum() * 1.0 / full['user'].nunique()
 
    def cal_ndcg(self):
        user_pos_item_num_dict, full, top_k = self._user_pos_item_num_dict, self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        import pdb;pdb.set_trace()
        test_in_top_k['dcg'] = test_in_top_k['rank'].apply(lambda x: 1 / math.log2(1 + x))
        idcg_list = []
        for i in test_in_top_k['user'].unique():
            idcg = 0.0
            for j in range(min(user_pos_item_num_dict[i], 10)):
                idcg += np.reciprocal(np.log2(j+2))
                idcg_list.append(idcg)
        test_in_top_k = test_in_top_k.groupby(by=['user'], as_index=False).sum()
        test_in_top_k['idcg'] = idcg_list
        test_in_top_k['ndcg'] = test_in_top_k['dcg'] * 1.0 / test_in_top_k['idcg']

        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
