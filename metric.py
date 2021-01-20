import numpy as np


def get_performance(gt_item, recommends):
    '''
    gt_item(np.array) : list of ground truth item indices for one user
    -> array[1,3,10]
    recommends(list) : list of top-k recommendation for one user
    -> [1,2,3 ... ,10]
    '''
    # Hit ratio & DCG
    hr = 0
    dcg = 0.0
    for item in gt_item:
        if item in recommends:
            hr += 1
            index = recommends.index(item)
            dcg += np.reciprocal(np.log2(index+2))
    
    if hr>0:
        # if any gt item is in recommends -> hr = 1
        hr1 = 1
        # hr 2 = # of hit item / # of gt item
        hr2 = hr/min(len(gt_item), len(recommends))
    else:
        hr1 = 0
        hr2 = 0

    # nDCG
    idcg = 0.0
    for i in range(min(len(gt_item),len(recommends))):
        idcg += np.reciprocal(np.log2(i+2))
        
    ndcg = dcg / idcg

    return hr1, hr2, ndcg