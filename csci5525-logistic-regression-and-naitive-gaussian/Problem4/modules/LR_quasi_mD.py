from sklearn.datasets import load_digits,load_boston
import numpy as np
from numpy import linalg as linalg
import math
import time

class LRmd():
    # Hessian: adaptive studying rate
    # mult-D

    @staticmethod
    def sigmoid(w, x):
        # w: 1*(d+1)
        # x: n*(d+1)

        N, D = x.shape
        # extra_x = np.ones((N, 1))
        # x = np.hstack((x, extra_x))

        a_n = x.dot(w.T)
        p_n = 1/(1 + np.exp(- a_n))

        return p_n     # n*1


    @staticmethod
    def softmax(w, x):
        # w: k*(d+1)
        # x: n*(d+1)
        # p: n*k

        N, D = x.shape

        a_nk = x.dot( w.T )
        a_nk = np.reshape(a_nk, (N, -1))

        max_k = np.max(a_nk, 1)
        exp_nk = np.exp(a_nk - max_k[:, None]) # (n, k) - (n,)
        p = exp_nk / np.sum(exp_nk, 1)[:, None]

        return p


    def loss(self, t, x, w):
        # t: n*k
        # x: n*d
        # w: k*(d+1)
        # p: n*k

        N, D = x.shape
        x = np.hstack((x, np.ones((N, 1))))

        p = self.softmax(w, x)      # p: n*k

        # elem = t * np.log(p) + (1 - t) * np.log((1 - p))
        elem = t * np.log(p)
        E = - np.sum(elem)
        return E


    @staticmethod
    def norm_1d(M, dirction):

        vector = np.sum(M**2, dirction)
        # print('vector.shape', vector.shape)

        return np.sqrt(vector)


    def direction(self, t, x, w):
        # t: n*k
        # x: n*d
        # w: k*(d+1)
        # p: n*k
        # dE = k*(d + 1), dE = vector

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))
        p = self.softmax(w, x)

        vector = (p - t).T.dot(x) # k*(d + 1)
        # vector = vector / self.norm_1d(vector, 1)[:, None]     # cannot broadcast (2,14) (2,)

        vector = vector.flatten()
        return vector/linalg.norm(vector)  # k*(d + 1)


    # def Hessian_inv(self, t, x, w):
    #     # t: n*k
    #     # x: n*d -> n*(d+1)
    #     # w: k*(d+1)
    #     # p: n*k
    #
    #     # H: k,j, d1,d2
    #     # Hinv: k,j, d1,d2
    #
    #     N, D = x.shape
    #     K = t.shape[1]
    #     extra_x = np.ones((N, 1))
    #     x = np.hstack((x, extra_x))
    #
    #     p = self.softmax(w, x)  # p: n*k
    #
    #     R_nkj = np.array(
    #         [ np.diag(p[n]) - (p[n])[:,None].dot(
    #             (p[n])[None,:]
    #         ) for n in range(N)]
    #     )
    #     R_nkj = np.transpose(R_nkj, (1, 2, 0))  # -> R_kjn
    #
    #     Hinv = []
    #     for k in range(K):
    #         for j in range(K):
    #             a_n = R_nkj[k,j]
    #             H_temp =  x.T.dot(
    #                 np.diag(a_n).dot(
    #                     x )
    #             )
    #
    #             # if linalg.matrix_rank(H_temp) < D + 1:
    #                 # print('H_temp not inversible! rank = ',linalg.matrix_rank(H_temp))
    #                 # print('k,j=', (k, j))
    #
    #             Hinv.append( linalg.pinv(H_temp) )      # Particularly, pinv(O) = O
    #
    #     Hinv = (np.array(Hinv)).reshape(K, K, D + 1, D + 1) # k,j*dp,dp
    #
    #     return Hinv # K,K,D+1,D+1


    def Hessian2(self, t, x, w):
        # Slow: for loop on n
        # Is about 7 times slower than the other one

        # t: n*k
        # x: n*d -> n*(d+1)
        # w: k*(d+1)
        # p: n*k

        # H: k,j, d1,d2
        # Hinv: k,j, d1,d2

        N, D = x.shape
        K = t.reshape((N, -1)).shape[1] # reshape to accomand (n,1)->(n,) degenerate.
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))

        p = self.softmax(w, x)  # p: n*k

        H = np.zeros((K*(D + 1), K*(D + 1)))
        for n in range(N):
            R_nkj = np.diag(p[n]) - (p[n])[:, None].dot(
                (p[n])[None, :]
            )
            X_ndd = x[n, :, None].dot(x[n, None, :])
            H = H + np.kron(R_nkj, X_ndd)

        return H # K,K,D+1,D+1 -> K*D, K*D


    def Hessian(self,t, x, w ):
        # t: n*k
        # x: n*d -> n*(d+1)
        # w: k*(d+1)
        # p: n*k

        # H: k,j, d1,d2
        # Hinv: k,j, d1,d2

        N, D = x.shape
        K = t.shape[1]
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))

        p = self.softmax(w, x)  # p: n*k

        R_nkj = np.array(
            [np.diag(p[n]) - (p[n])[:, None].dot(
                (p[n])[None, :]
            ) for n in range(N)]
        )
        R_nkj = np.transpose(R_nkj, (1, 2, 0))  # -> R_kjn

        H = []
        for k in range(K):
            for j in range(K):
                a_n = R_nkj[k, j]
                H_temp = x.T.dot(
                    np.diag(a_n).dot(
                        x)
                )

                # if linalg.matrix_rank(H_temp) < D + 1:
                # print('H_temp not inversible! rank = ',linalg.matrix_rank(H_temp))
                # print('k,j=', (k, j))

                H.append(H_temp)  # Particularly, pinv(O) = O

        H = (np.array(H)).reshape(K, K, D + 1, D + 1)  # k,j * dp,dp
        # _k*j,d,d

        new_D = K*(D + 1)
        H = np.transpose(H, (0, 2, 1, 3))
        H = np.reshape(H, (1, new_D, 1, new_D))
        H = H[0, :, 0, :]

        return H  # K,K,D+1,D+1 -> new_D * new_D the direct product




    # @staticmethod
    # def tensor_multiplier(Hinv, dE, N, K, D):
    #     # Hinv: K,K,D+1,D+1
    #     # dE: k*(D+1)
    #     # dw: k*(d+1)
    #
    #     dw = np.zeros((K, D + 1))
    #
    #     for k in range(K):
    #         for dp in range(D):
    #             dw[k, dp] = np.dot(Hinv[k, :, dp, :].flatten(), np.transpose(dE.flatten()))
    #
    #     return dw


    def IRLS_basic(self, x_train, t_train, yita = 1, max_count = 500, rel_err = 1e-5):
        # t: n*k
        # x: n*d -> n*(d+1)
        # w: k*(d+1)

        N, D = x_train.shape
        K = t_train.shape[1]

        w_init = np.zeros((K, D + 1))

        Hinv = linalg.pinv(self.Hessian(t_train, x_train, w_init))
        # H = self.Hessian(t_train, x_train, w_init)
        # Lin = linalg.inv(linalg.cholesky(H))
        # Hinv = Lin.T.conj().dot(Lin)
        # print('Matrix you are dealing with has D =', Hinv.shape[0])

        dE = self.direction(t_train, x_train, w_init)

        # dw = self.tensor_multiplier(Hinv, dE, N, K, D)
        dw = Hinv.dot(dE)
        w = w_init - yita * dw.reshape((K, -1))


        E = self.loss(t_train, x_train, w)
        # print('E = ', E)
        count = 1

        E_temp = E
        E_buff = []

        start = time.time()

        while 1:

            w_temp = w

            H = self.Hessian(t_train, x_train, w)
            (u, s, vh) = linalg.svd(H)
            Hinv = linalg.pinv(H)
            # H = self.Hessian(t_train, x_train, w)
            # Lin = linalg.inv( linalg.cholesky(H) )
            # Hinv = Lin.T.conj().dot(Lin)

            dE = self.direction(t_train, x_train, w)
            # dw = self.tensor_multiplier(Hinv, dE, N, K, D)
            dw = Hinv.dot(dE)
            w = w - yita * dw.reshape((K, -1))  # reshape and flattern are inverse, and compatible with kron.

            E = self.loss(t_train, x_train, w)
            # print('E = ', E)

            # if np.all(self.norm_1d(dw, 1) <= abs_err):
            if abs(E_temp - E)/E < rel_err and count > 2e1:
            # if np.allclose(w_temp, w) and count > 1e2:
            #     print('Yayyyyyyyy!')
            #     print('count = ', count)
                # print('rel_err of E in adjacent sampling loop = ', abs(E_temp - E)/E )
                break
            elif count >= max_count:
                print("Count is up. I'm out!")
                print('count = ', count)
                break
            else:
                count = count + 1
                if not (count % 1e1):
                    E_temp = E
                    # print('|w_new - w_old| = ', linalg.norm(w - w_temp))
                    # print('yita = ', yita)
                    # if E == NaN:


                    E_buff.insert(0, E)
                    if len(E_buff) >= 10:
                        E_buff.pop()

                    # Adapting the learning_rate
                    yita = self.adapt_learn_rate(E_buff, yita)

                # if not (count % 1e0):
                    # print('E = ', E)

        end = time.time()
        # print('Time consumed:' + str(end - start) + 's')

        return w


    @staticmethod
    def adapt_learn_rate(E_buff, yita_in):
        E = E_buff[0]
        yita = yita_in

        yita_list = [5, 1, 0.5, 0.1]

        # if E < 200:
        #     yita = 100
        if E < 300:
            yita = 10
        if E < 88:
            # maxmin = (E_buff[0] - E_buff[-1])
            # if maxmin < 1 and yita_list[gate_idx]:
            #     yita = yita_list[gate_idx]
            #     gate_idx = gate_idx + 1
            #
            # yita = max(yita, 0.1)
            yita = yita_list[0]
        if E < 85:
            yita = yita_list[1]
        if E < 80:
            yita = yita_list[2]
        if E < 75:
            yita = yita_list[3]

        # if E < 88:
        #     yita = 1
        # if E < 80:
        #     yita = 1

        # if yita_in != yita:
            # print('yita updated:', yita)

        # [E, yita_c] = [85, 10], [87, 1]
        # yita(E): [E, yita] = [85, 1] [90, 10]
        # yita = 2 * (E - 85) + 1

        return yita


    def validation(self, w, t, x):
        # Validation:

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))
        p_nk = self.softmax(w, x)

        result = np.argmax(p_nk, 1)
        test = np.argmax(t, 1)

        err = 1 - np.mean(result == test)
        # print('err_rate_testingSet', err)
        return err, result


def transform_to_hotkey(t_train):
    # if b_or_d == 'b':
    #     N = len(t_train)
    #
    #     t = np.zeros((N, 2))
    #     t[t_train == 0, 0 ] = 1
    #     t[t_train == 1, 1] = 1
    #
    # elif b_or_d == 'd':
    N = len(t_train)
    K = int(max(t_train))
    t = np.zeros((N, K+1))
    for i in range(K):
        t[t_train == i, i] = 1

    return t


def LR(x_train, t_train, x_test, t_test, b_or_d, yita_inp ):
# def main():
#     b_or_d = 'd'
#     x_train, x_test, t_train, t_test = LRmd().extract_data(x_train, t_train, x_test, t_test,b_or_d)

    # truncated svd: the first 400 dim.

    t_train = transform_to_hotkey(t_train)
    t_test = transform_to_hotkey(t_test)


    max_count = int(5e1)

    if b_or_d == 'b':
        yita = 1e3  # 1: drift away, 1000 too big for 'd', 100 too slow?
        relerr = 1e-4
    else:
        yita = yita_inp
        relerr = 1e0

    w = LRmd().IRLS_basic(x_train, t_train, yita = yita, max_count = max_count, rel_err=relerr)
    # print('yita = ', yita)

    err, result = LRmd().validation(w, t_test, x_test)
    # print('err= ', err)
    # print('result = ', result)
    # print('b_or_d:', b_or_d)

    return err



# if __name__ == '__main__':
#     main()