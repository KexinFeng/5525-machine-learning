import numpy as np
import numpy.linalg as linalg
import math
import time, sys



class Pegasos():

    def divider(self, X, t):
        # divider
        self.index1 = np.argwhere(t == 1).reshape((-1,))
        self.index2 = np.argwhere(t == -1).reshape((-1,))
        self.N = len(self.index1) + len(self.index2)
        self.D = X.shape[1]
        return


    def getSubset(self, k):
        k = int(k)
        if not k == 1:
            percent = k/self.N
            N1 = self.index1.size
            N2 = self.index2.size
            size1 = int(N1*percent)
            size2 = int(N2*percent)

            sizes = np.array([size1, size2])
            if not np.sum(sizes) == k:
                while (np.sum(sizes) > k):
                    sizes[np.argmax(sizes)] -= 1
                while (np.sum(sizes) < k):
                    sizes[np.argmin(sizes)] += 1

            pickIndex1 = np.random.choice(self.index1, sizes[0], replace=False)
            pickIndex2 = np.random.choice(self.index2, sizes[1], replace=False)

            pickIndex = np.append(pickIndex1, pickIndex2)
            assert pickIndex.size == pickIndex1.size + pickIndex2.size
            np.random.shuffle(pickIndex)

        else:
            indices = np.arange(self.N, dtype=int)
            pickIndex = np.random.choice(indices, 1, replace=False)

        return pickIndex


    def projSet(self, x, t, w):
        tnyn = t.reshape((-1,1)) * x.dot(w.reshape((-1, 1)))
        idx = tnyn < 1
        idx = idx.reshape((-1,))
        xp = x[idx, :]
        tp = t[idx]

        return xp, tp


    def optimizer(self, X, t , lamda, Tmax, k):
        # Tmax = int(Tmax)

        self.divider(X,t)

        D = X.shape[1]
        N1 = np.size(self.index1)
        N2 = np.size(self.index2)

        N = N1 + N2
        max_ktot = 100 * N # 2e5
        Tmax = int(max_ktot / k)
        # print("Tmax = ", Tmax)



        w0 = np.zeros(D)
        R = 1/math.sqrt(lamda)
        assert  linalg.norm(w0) <= R
        w = w0
        self.loss_rec = []

        it = 1
        ktot = 0

        starttime = time.time()
        while( it <= Tmax and ktot < max_ktot ):
            ktot += k
            pickIndex = self.getSubset(k)
            xsub = X[pickIndex,:]
            tsub = t[pickIndex]
            assert tsub.shape[0] == k

            xp, tp = self.projSet(xsub, tsub, w)
            yita = 1/(lamda * it)

            # print("size of the projected set ", xp.shape[0])

            gradient = tp.reshape((-1,1)) * xp
            assert gradient.shape[1] == D
            woh = (1 - yita * lamda) * w + yita * np.sum(gradient, axis=0)/k
            w = min(1, R/linalg.norm(woh)) * woh
            assert N == X.shape[0]
            temp_loss = lamda/2*linalg.norm(w)**2 + np.mean(np.max(np.hstack((np.zeros((N, 1)), 1 - t.reshape((-1, 1)) * X.dot(w.reshape((-1, 1)) ) )), axis=1), axis = 0)
            self.loss_rec.append(temp_loss)

            # if linalg.norm(woh)>R:
            #     print("out")

            if self.isNaN(w):
                print("bug!")


            # if it % 1e3 == 0:
            #     # print("wT = ", np.linalg.norm(wT))
            #     # print("yita ", yita)
            #     print("it = ", it / 1e3, "e3")
            #     print("time per iteration: ", time.time() - starttime)
            #     print("")
            #     starttime = time.time()

            it += 1


        return w

    @staticmethod
    def isNaN(v):
        return np.any(v != v)



class svm_Peg(Pegasos):

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





    def readData(self, filename):
        # filename = './MNIST-13.csv'
        readin = np.genfromtxt(filename, delimiter=',')

        x = readin[:, 1:]
        t = readin[:, 0].astype(int)
        return x,t

    def train(self, x, t, lamda=1, Tmax=1e3, k=200):

        # Peg = Pegasos()
        # w = Peg.optimizer(x, t, lamda, Tmax, k)

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

        svm = svm_Peg()
        xraw, traw = svm.readData(filename)

        times = np.zeros(num_runs)
        loss_dec_rates = np.zeros(num_runs)

        for i in range(num_runs):
            tic = time.time()
            print('k={}, run {}/{}'.format(k, i + 1, num_runs))
            svm = svm_Peg()
            x, t = svm.init_preprocess(xraw, traw)

            svm.train(x, t, k=k)
            diff_loss = np.diff(svm.loss_rec)
            loss_dec_rates[i] = np.mean(diff_loss[-2:-1])

            toc = time.time()
            times[i] = toc - tic

            # err = svm.validation(xraw, traw, w)

        print("Average run time = ", np.mean(times))
        print("Std of run time = ", np.std(times))
        print("Final loss decreasing rate = ", np.mean(-loss_dec_rates), ' (per 2k times of gradient)')
        print("Final loss decreasing rate std = ", np.std(loss_dec_rates))
        print("")



    else:
        # print('$ python my_Pegsos <dataFile> <regulatorC>')
        print('Usage: $ python myPegsos ./MNIST-13.csv 200 5')
        sys.exit(1)




if __name__ == "__main__":

    main()