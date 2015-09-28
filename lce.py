import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy import sparse as sps
from scipy.sparse import coo_matrix
import pickle

eps = 7. / 3 - 4. / 3 - 1

import logging
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('lce.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)
logger.debug('This message should go to the log file')
logger.warning('And this, too')
logger.info('So should this')
'''
logging.basicConfig(filename='example.log', level=logging.DEBUG)
'''
class SP_linalg:

    def matProduct(self,A,B):
        if sparse.issparse(A) or sparse.issparse(B):
            AA=sparse.csc_matrix(A)
            BB=sparse.csc_matrix(B)
            rst=sparse.csc_matrix.dot(AA,BB)
            return rst #in this case, return a
        else:
            return np.dot(A,B)

    def eleProduct(self,A,B):
        if sparse.issparse(A) or sparse.issparse(B):
            AA=sparse.csc_matrix(A)
            BB=sparse.csc_matrix(B)
            rst=sparse.csc_matrix.multiply(AA,BB)
            return rst #in this case, return a
        else:
            return np.multiply(A,B)

    def eleWiseMax(self,A,b):
        if sparse.issparse(A):
            return sparse.csc_matrix.maximum(A,b)
        else:
            return np.maximum(A,b)

    def HamardDivide(self,A,B): # element wise matrix division
         if sparse.issparse(A) or sparse.issparse(B):
            AA=sparse.csc_matrix(A)
            BB=sparse.csc_matrix(B)
            BBinv=sparse.csc_matrix.power(BB,-1)
            rst=sparse.csc_matrix.multiply(AA,BBinv)
            return rst #in this case, return a
         else:
            return np.divide(A,B)
    def trace(self,A):
        if sparse.issparse(A):
            return np.sum(A.diagonal())
        else:
            return np.trace(A)
class LCE():

    def computeMulNorm(self, A, B):  # compute the elementwise sum of A*B
        if sps.issparse(A) or sps.issparse(B):
            x = coo_matrix.multiply(coo_matrix(A), coo_matrix(B))
        else:  # normal matrix
            x = np.multiply(A, B)
        y=(x.sum(axis=0)).sum(axis=1)
        return y  # return a scalar

    def logRst(self,inx):
        import pickle
        with open('W_Hs_Hu_iter'+str(inx)+'.pickle', 'wb') as f:
            pickle.dump((self.W,self.Hs,self.Hu),f)

    def run(self, X_itemProperty_train, X_userRate_train, norm=True):
        A = self.construct_A(X_itemProperty_train, 1, True)
        matRate = self.L2_norm_row(X_userRate_train) if norm == True else X_userRate_train
        try:
            W, Hu, Hs = self.sp_train(X_itemProperty_train, matRate, A);
        except:
            raise

    def L2_norm_row(self, X):  # apply L2 normalization to the row of X
        if sps.issparse(X):
            diag = sparse.spdiags((1. / (np.sqrt(X.multiply(X).sum(1)) + eps)).T, 0, X.shape[0],
                                  X.shape[0])  # in case X is sparse
        else:
            diag = sparse.spdiags(1. / (np.sqrt(np.sum(np.multiply(X, X), axis=1)) + eps).T, 0, X.shape[0], X.shape[0])
        return np.dot(diag, X)
    # main_diag(X)*X
    # len only count for the number of rows
    # X*X is the elementwise multiplication
    # sum(A,num) sums up the column and plus num to the result
    # rst=spdiag(mat,[index array for diagnal],shape) puts the ith(0,-1,+1,+2,-2) diagnal of mat into row vector of rst

    def construct_A(self, X, k, binary=False): #might generate sparse matrix
        nbrs = NearestNeighbors(n_neighbors=1 + k).fit(X)
        if binary:
            return nbrs.kneighbors_graph(X)
        else:
            return nbrs.kneighbors_graph(X, mode='distance')

    def recommendItem2User(self, mat_item_property):
        mat4itemRating=[]
        for vec_item_property in mat_item_property: #iterate all the item property
            qu, _ = self.predict_user_rank(vec_item_property) # qu is the all-user rating for one item
            mat4itemRating.append(qu)

    def getPseudoInv(self,A):
        pass
    def predict_user_rank(self, vec_item_property): #qs=wHs,w is the unknown, which is the map from item to latent
        #solve w from  equation vec_item_property=w*self.latent2Property
        symmetric_inv = np.linalg.inv(np.dot(self.latent2Property, self.latent2Property.T))  # m*n:n*m*m*n=n*n
        print "vec_item_property shape is %s" % vec_item_property.shape
        pseudo_inv = np.dot(self.latent2Property.T,symmetric_inv)
        w = np.dot(vec_item_property,pseudo_inv)  # n*m*m*1= n*1  number of latent variable
        qu = w * self.latent2User  # number of latent* n* numberOfUser= numberOfUser
        return qu, [i[0] for i in sorted(enumerate(qu), key=lambda x: x[1], reverse=True)]

    def __init__(self):
        self.sp_linalg=SP_linalg();

    def loadStateVar(self,W,Hs,Hu):
        self.latent2Property=Hs.todense()
        self.latent2User=Hu.todense()
        self.item2latent=W.todense()
        print type(Hs),type(Hu),type(W)

    def sp_train(self, Xs, Xu, A, k=15, alpha=0.1, beta=0.05, lamb=0.001, epsilon=0.01, maxiter=15000, verbose=True):
        # self.W, Hu, Hs=LCE(X_item_factor,X_item_user,A_item_dist)
        # Xs is item-factor matrix, Xu is the item-user rating matrix, A is the distance between items
        # put attention about the index of items and users: the items index in Xs and Xu should be consistent!!!!
        # Hu is map(latentSpace,user)
        # Hs is map(latentSpace,property)
        n = Xs.shape[0]  # number of items
        v1 = Xs.shape[1]  # number of properties
        v2 = Xu.shape[1]  # number of users

        self.W = abs(np.random.rand(n, k))
        self.Hs = abs(np.random.rand(k, v1))
        self.Hu = abs(np.random.rand(k, v2))
        D = sparse.dia_matrix((A.sum(axis=0), 0), A.shape)

        #constraint
        gamma = 1. - alpha
        trXstXs = self.computeMulNorm(Xs, Xs)  # scalar
        trXutXu = self.computeMulNorm(Xu, Xu)  # scalar

        #values for first iteration
        WtW=self.sp_linalg.matProduct(self.W.T,self.W)
        WtXs=self.sp_linalg.matProduct(self.W.T,Xs)
        WtXu=self.sp_linalg.matProduct(self.W.T,Xu) # [#(latent),#(items)] * [#(items),#(users)]

        WtWHs=self.sp_linalg.matProduct(WtW,self.Hs)
        WtWHu =self.sp_linalg.matProduct(WtW,self.Hu)

        DW=self.sp_linalg.matProduct(D,self.W)
        AW=self.sp_linalg.matProduct(A,self.W)


        itNum = 1
        delta = 2.0 * epsilon

        ObjHist = []


        while itNum < maxiter and delta > epsilon:
            if itNum % 200 ==0:
                self.logRst(itNum)
            # update H
            Hs_1=self.sp_linalg.HamardDivide((alpha * WtXs),self.sp_linalg.eleWiseMax(alpha*WtWHs + lamb * self.Hs,1e-10))
            self.Hs=self.sp_linalg.eleProduct(self.Hs,Hs_1)# equation(10)


            Hu_1=self.sp_linalg.HamardDivide((gamma * WtXu),self.sp_linalg.eleWiseMax(gamma * WtWHu + lamb * self.Hu,1e-10))
            self.Hu=self.sp_linalg.eleProduct(self.Hu,Hu_1) # equation(11)
            #logger.debug("Hu is %s" % Hu)
            #print("Hu is %s" %Hu)

            # update W
            W_t1=alpha*self.sp_linalg.matProduct(Xs,self.Hs.T)+gamma*self.sp_linalg.matProduct(Xu,self.Hu.T)+beta*AW
            W_t2 = alpha *self.sp_linalg.matProduct(self.W,self.sp_linalg.matProduct(self.Hs,self.Hs.T))+gamma * self.sp_linalg.matProduct(self.W,self.sp_linalg.matProduct(self.Hu,self.Hu.T)) + beta * DW + lamb * self.W
            W_t3 = self.sp_linalg.HamardDivide(W_t1, self.sp_linalg.eleWiseMax(W_t2, 1e-10))
            self.W = self.sp_linalg.eleProduct(self.W, W_t3)  # equation (9)

            # calculate objective function

            WtW = self.sp_linalg.matProduct(self.W.T,self.W)
            WtXs= self.sp_linalg.matProduct(self.W.T,Xs)
            WtXu =self.sp_linalg.matProduct(self.W.T,Xu)

            WtWHs = self.sp_linalg.matProduct(WtW,self.Hs)
            WtWHu = self.sp_linalg.matProduct(WtW,self.Hu)
            DW = self.sp_linalg.matProduct(D,self.W)
            AW = self.sp_linalg.matProduct(A,self.W)

            # compute the components for the objective function  equation(1)
            tr1 = alpha * (trXstXs - 2. * self.computeMulNorm(self.Hs, WtXs) + self.computeMulNorm(self.Hs, WtWHs))
            #tr1 seems to be the same all the time
            tr2 = gamma * (trXutXu - 2. * self.computeMulNorm(self.Hu, WtXu) + self.computeMulNorm(self.Hu, WtWHu))
            tr3 = beta * (self.computeMulNorm(self.W, DW) - self.computeMulNorm(self.W, AW))
            tr4_1= lamb * (self.sp_linalg.trace(WtW))
            tr4_2= lamb*(self.computeMulNorm(self.Hs, self.Hs) + self.computeMulNorm(self.Hu, self.Hu))
            tr4=tr4_1+tr4_2

            Obj = tr1 + tr2 + tr3 + tr4  ##### equation(1)
            ObjHist.append(Obj)

            if itNum > 1:
                delta = abs(ObjHist[-1] - ObjHist[-2])
                if verbose:
                    print "Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta
                if itNum > maxiter or delta < epsilon:
                    break
            itNum += 1
        self.item2latent = self.W
        self.latent2User = self.Hu
        self.latent2Property = self.Hs



def testcase_te():
    obj=LCE()
    with open('W_Hs_Hu_iter600.pickle', 'rb') as f:
        W,Hs,Hu= pickle.load(f)
    obj.loadStateVar(W,Hs,Hu)
    with open('Xstr_Xu_Xs_te.pickle', 'rb') as f:
        _,_,Xs_test=pickle.load(f)
    print Xs_test[:,-1]
    obj.recommendItem2User(Xs_test)


def testcase_tr():
    lce_obj=LCE()
    with open('Xstr_Xu_Xs_te.pickle', 'rb') as f:
        Xs_tr,Xu,Xs_te= pickle.load(f)
    lce_obj.run(sparse.coo_matrix(Xs_tr),Xu)
    lce_obj.recommendItem2User(Xs_te)
if __name__ == '__main__':
    testcase_tr()

