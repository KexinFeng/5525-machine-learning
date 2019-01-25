import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from numpy import genfromtxt


class SVMprime():

    def __init__(self, X, t, C):
        t = np.copy(t)
        self.X, self.t = self.init_preprocess(X, t)
        self.C = C
        self.N, self.dim = X.shape

        return

    def dimReduction(self, X):
        rowstd = np.std(X, axis = 0)
        self.col_idx = rowstd > 1e-3
        Xnew = X[:, self.col_idx]

        self.Xmean = np.mean(Xnew, axis = 0)
        self.Xstd = np.std(Xnew, axis = 0, ddof = 1)
        Xnew = (Xnew - self.Xmean) / self.Xstd

        return Xnew

    def init_preprocess(self, X, t):
        X = self.dimReduction(X)
        tnew = np.copy(t).astype(int)
        self.classes  = np.unique(tnew)
        tnew[t == self.classes[0]] = 1
        tnew[t == self.classes[1]] = -1
        return X, tnew

    def preprocess(self, X):
        X = X[:, self.col_idx]
        X = (X - self.Xmean) / self.Xstd
        return X

    def optimization(self, X, t, C):
        # max 1/2 xPx + qx subject to
        # Gx <= h
        # Ax = b

        P = matrix(t[:,None].dot(t[None,:]) * X.dot(X.T))
        q = matrix(- np.ones(self.N))
        A = matrix(t[None,:].astype(float))
        b = matrix([0.0])
        iden = np.identity(self.N)
        G = matrix(np.concatenate((iden, -iden), axis = 0))
        h = matrix(np.concatenate((C * np.ones(self.N), np.zeros(self.N))))
        result = solvers.qp(P, q, G, h, A, b)
        a = np.array(result['x'])

        return a[:,0]

    # @staticmethod
    def SVMparameter(self, a, X, t, C):
        support = np.argwhere (a>1e-8)
        sp = np.reshape(support, (-1,))
        margin = np.argwhere((a < (C-1e-8)) * ( a > 1e-8 ))
        mg = np.reshape(margin, (-1,))

        self.num_sup = len(sp)
        self.num_marg = len(mg)

        mm = X[mg,:].dot(X[sp,:].T)
        nn = a[sp].reshape((-1,1))*t[sp].reshape((-1,1))

        dd = np.sum(mm.dot(nn), axis = 1)
        b = np.mean(t[mg] - dd)

        # b = np.mean(t[mg] - np.sum(X[mg,:].dot(X[sp,:].T).dot(a[sp].reshape((-1,1))*t[sp].reshape(-(1,1))), axis = 1))
        w = (a * t).dot(X)

        return w, b



    def valid(self, Xtest, ttest):
        Xtest = self.preprocess(Xtest)

        prediction = self.prediction(Xtest)
        test_error = 1 - np.mean(prediction == ttest )
        # predicted_indicator = np.array([prediction[i] == ttest[i] for i in range(0, y.size)])
        # test_error = 1 - np.sum(predicted_indicator) / ttest.size
        return test_error

    def prediction(self, X):
        Ntest = X.shape[0]
        y = np.zeros((Ntest, 2))
        y[:, 0] = X.dot(self.w ) + self.b
        y[:, 1] = - y[:, 0]
        pred_idx = np.argmax(y, axis = 1)
        return np.array([self.classes[i] for i in pred_idx])



    def train(self):
        a = self.optimization(self.X, self.t, self.C)
        self.w, self.b = self.SVMparameter(a,self.X, self.t, self.C)
        return

    def scoreFun(self, Xtest):
        Xtest = self.preprocess(Xtest)

        prediction = self.prediction(Xtest)




