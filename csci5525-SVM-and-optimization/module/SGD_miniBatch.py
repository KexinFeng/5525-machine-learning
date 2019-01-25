import numpy as np
import math
import time

class SGD_miniB():
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
                    # size1 -= 1
                    # print("size1 + size2 > k")
                while (np.sum(sizes) < k):
                    sizes[np.argmin(sizes)] += 1
                    # size2 += 1
                    # print("size1 + size2 < k")

            pickIndex1 = np.random.choice(self.index1, sizes[0], replace=False)
            pickIndex2 = np.random.choice(self.index2, sizes[1], replace=False)

            pickIndex = np.append(pickIndex1, pickIndex2)
            assert pickIndex.size == pickIndex1.size + pickIndex2.size
            np.random.shuffle(pickIndex)

        else:
            indices = np.arange(self.N, dtype=int)
            pickIndex = np.random.choice(indices, 1, replace=False)

        return pickIndex

    def optimizer(self, x, t, lamda, Tmax, k):
        Tmax = int(Tmax)
        N, D = x.shape

        self.divider(x,t)

        beta = .1
        a = 1
        w0 = np.zeros(D)

        wt = w0
        wT = w0
        self.loss_rec = []

        max_ktot = 100 * N  # 2e5
        Tmax = int(max_ktot / k)
        # print("Tmax = ", Tmax)

        # w_array = np.zeros((Tmax, D))

        it = 1
        ktot = 0

        starttime = time.time()

        while( it <= Tmax and ktot < max_ktot):

            ktot += k

            # yita = 1/(lamda * it)

            if it >= 5:
                yita = beta * np.linalg.norm(wT)/math.sqrt(it)
                # print(yita)
            else:
                yita = 1


            grad = self.gradient(x, t, wt, k, a)
            wt =  wt * (1 - yita * lamda) + yita * grad
            # w_array[it - 1] = wt
            # wT = np.mean(w_array[:it, :], axis = 0)
            assert x.shape[0] == t.shape[0]
            # warning: Even though mean is calculated based on array, its time complexity is sitll O(N)
            wT = (wT * (it - 1) / it + wt / it )
            # assert np.allclose(wT, np.mean(w_array[:it, :], axis = 0))

            temp_loss = lamda / 2 * np.linalg.norm(wT)**2 + np.mean(a * np.log(1 + np.exp((1 - t.reshape((-1, 1)) * x.dot(wT.reshape((-1, 1)))) / a)))
            self.loss_rec.append(temp_loss)

            # if it % 1e3 == 0:
            #     # print("wT = ", np.linalg.norm(wT))
            #     # print("yita ", yita)
            #     print("it = ", it / 1e3, "e3")
            #     print("time per iteration: ", time.time() - starttime)
            #     print("")
            #     starttime = time.time()

            it += 1

        return wT

    def gradient(self, x, t, w, k, a):
        # a = 1

        # assert w.shape == x[0].shape
        # N, D = x.shape
        # index = np.arange(N, dtype=int)
        # pick = np.random.choice(index, 1, replace=False)
        # pick = pick[0]
        # grad = t[pick] * x[pick] / (math.exp(-(1 - t[pick] * w.dot(x[pick])))/a + 1)

        pickIndex = self.getSubset(k)
        xsub = x[pickIndex, :]
        tsub = t[pickIndex]
        assert tsub.shape[0] == k

        grad = np.mean(tsub.reshape((-1, 1)) * xsub / (np.exp(-(1 - tsub.reshape((-1, 1)) * xsub.dot(w.reshape(-1, 1)))/a) + 1), axis = 0)

        return grad.reshape((-1,))

