import numpy as np
import numpy.linalg as linalg
import math
from module.SGD_miniBatch import SGD_miniB
import time, sys




class svm_SGD_miniB(SGD_miniB):


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
        # t = t.astype(int)
        # t[t == self.classes[0]] = 1
        # t[t ==self.classes[1]] = -1
        return X



    def readData(self, filename ='./MNIST-13.csv'):
        # filename = './MNIST-13.csv'
        readin = np.genfromtxt(filename, delimiter=',')

        x = readin[:, 1:]
        t = readin[:, 0].astype(int)
        return x,t




    def train(self, x, t, lamda=1, Tmax=1e4, k=2000):
        w = self.optimizer(x, t, lamda, Tmax, k)

        return w



    def validation(self, xtest, ttest, w):
        xtest = self.preprocess(xtest)

        pred = self.prediction(xtest, w)
        test_err = 1 - np.mean(pred == ttest)

        return test_err

    def prediction(self, xtest, w):
        assert w.size == xtest.shape[1]
        Ntest = xtest.shape[0]
        y = np.zeros((Ntest, 2))
        y[:, 0] = xtest.dot(w.reshape((-1, ) ) )
        # y[:, 0] = xtest.dot(w.reshape((-1, 1) ) )
        y[:, 1] = -y[:, 0]
        pred_idx = np.argmax(y, axis=1)

        return np.array([self.classes[i] for i in pred_idx])




def main(argv=sys.argv):

    if len(argv) == 4:
        filename = argv[1]
        k = int(argv[2])
        num_runs = int(argv[3])

        start = time.time()
        np.random.seed(int(start))

        svm = svm_SGD_miniB()
        xraw, traw = svm.readData(filename)

        times = np.zeros(num_runs)
        loss_dec_rates = np.zeros(num_runs)

        for i in range(num_runs):
            tic = time.time()
            print('k={}, run {}/{}'.format(k, i + 1, num_runs))
            svm = svm_SGD_miniB()
            x, t = svm.init_preprocess(xraw, traw)

            svm.train(x, t, k=k)

            diff_loss = np.diff(svm.loss_rec)
            loss_dec_rates[i] = np.mean(diff_loss[-2:-1])

            toc = time.time()
            times[i] = toc - tic

            # err = svm.validation(xraw, traw, w)

        print("Average run time (s) = ", np.mean(times))
        print("Std of run time = ", np.std(times))
        print("Final loss decreasing rate = ", np.mean(-loss_dec_rates), ' (per 2k times of gradient)')
        print("Final loss decreasing rate std = ", np.std(loss_dec_rates))
        print("")



    else:
        # print('$ python my_Pegsos <dataFile> <regulatorC>')
        print('Usage: $ python mySoftplus ./MNIST-13.csv 200 5')
        sys.exit(1)



if __name__ == '__main__':
    main()