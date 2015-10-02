__author__ = 'sunxd'
import numpy as np

def NormDiscountCumGain(pred_item_user_rate,actual_item_user_rate):
    #rank is the ranking from the ML algorithm
    #target is the binary with labels annotating
    (num_item, num_user)=pred_item_user_rate.shape
    num_rel4item = np.sum(actual_item_user_rate,axis=1)#count for each item,how many relevant hit(user has visit it)
    mat_item_favorUser_idx= np.argsort(pred_item_user_rate,axis=1)[::-1]
    result = 0;
    denominator=np.append([1],np.log2(np.array(np.arange(2,num_user+1))))
    for i in range(0,num_item):
        DCG=np.sum(np.divide(actual_item_user_rate[i,mat_item_favorUser_idx[i,:]],denominator))
        IDCG=np.sum(np.append([1],np.log2(np.arange(2,num_rel4item[i]))))
        result = result + ((DCG / IDCG) / num_item);


