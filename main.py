__author__ = 'sunxd'
from lce import LCE
import pickle
import sys
from scipy import sparse
import pandas as pd
import numpy as np

def coupon_usecase_test():
    obj=LCE()
    with open('W_Hs_Hu_iter50.pickle', 'rb') as f:
        W,Hs,Hu= pickle.load(f)
    obj.loadStateVar(W,Hs,Hu)
    with open('Xstr_Xutr_Xste.pickle', 'rb') as f:
        _,_,Xs_test=pickle.load(f)
    print Xs_test[:,-1]
    obj.recommendItem2User(Xs_test)


def coupon_usecase_train(): #only output state variable, no prediction
    lce_obj=LCE()
    with open('Xstr_Xutr_Xste.pickle', 'rb') as f:
        Xs_tr,Xu_tr,Xs_te= pickle.load(f)
    lce_obj.run(sparse.coo_matrix(Xs_tr),Xu_tr)

def standardTrain():
    df_Xu_train=pd.read_csv('standardInput/Xu_train.csv')
    df_Xu_test=pd.read_csv('standardInput/Xu_test.csv')
    df_Xs_train=pd.read_csv('standardInput/Xs_train.csv')
    df_Xs_test=pd.read_csv('standardInput/Xs_test.csv')

    lce_obj=LCE()
    lce_obj.run(sparse.coo_matrix(df_Xu_train.values),sparse.coo_matrix(df_Xs_train.values))




if __name__ == '__main__':
    print sys.argv
    #standardTrain()
    import metric
    metric.NormDiscountCumGain(np.array([[1,0],[2,3]]),np.array([[1,0],[0,1]]))
    print 'main-over'
