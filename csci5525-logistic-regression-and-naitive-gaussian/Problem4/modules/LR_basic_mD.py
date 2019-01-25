from sklearn.datasets import load_digits,load_boston
import numpy as np
from numpy import linalg as linalg
import math

class LR():
    # basic: constant studying rate
    # multi-D

    @staticmethod
    def extract_data( b_or_d = 'b', test_size = 100):
        digits = load_digits()
        boston = load_boston()

        d_data = digits.data
        b_data = boston.data
        d_target = digits.target
        b_target = boston.target

        t0 = np.median(b_target)

        target = np.zeros(len(b_target), dtype=int)
        target[b_target <= t0] = 1

        b_target = target


        ## divide the data to training set vs testing set
        if b_or_d == 'd':
            divider = 1797 - test_size

            x_train = d_data[:divider, :]  # N*D
            x_test = d_data[divider:, :]

            t_train = d_target[0:divider]
            t_test = d_target[divider:]
        elif b_or_d == 'b':
            divider = 506 - test_size

            x_train = b_data[:divider, :]  # N*D
            x_test = b_data[divider:, :]

            t_train = b_target[0:divider]
            t_test = b_target[divider:]
        else:
            print('Please check input!')

        return x_train, x_test, t_train, t_test

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

        return vector / self.norm_1d(vector, 1)[:, None]     # cannot broadcast (2,14) (2,)


    def Hessian_inv(self, t, x, w):
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
            [ np.diag(p[n]) - (p[n])[:,None].dot(
                (p[n])[None,:]
            ) for n in range(N)]
        )
        R_nkj = np.transpose(R_nkj, (1, 2, 0))  # -> R_kjn

        Hinv = []
        for k in range(K):
            for j in range(K):
                a_n = R_nkj[k,j]
                H_temp = - x.T.dot(
                    np.diag(a_n).dot(
                        x )
                )

                # if linalg.matrix_rank(H_temp) < D + 1:
                    # print('H_temp not inversible! rank = ',linalg.matrix_rank(H_temp))
                    # print('k,j=', (k, j))

                Hinv.append( linalg.pinv(H_temp) )      # Particularly, pinv(O) = O

        Hinv = (np.array(Hinv)).reshape(K, K, D + 1, D + 1) # k,j*dp,dp

        return Hinv # K,K,D+1,D+1


    @staticmethod
    def tensor_multiplier(Hinv, dE, N, K, D):
        # Hinv: K,K,D+1,D+1
        # dE: k*(D+1)
        # dw: k*(d+1)

        dw = np.zeros((K, D + 1))

        for k in range(K):
            for dp in range(D):
                dw[k, dp] = np.dot(Hinv[k, :, dp, :].flatten(), np.transpose(dE.flatten()))

        return dw


    def IRLS_basic(self, x_train, t_train, yita = 1, max_count = 500, rel_err = 1e-7):
        # t: n*k
        # x: n*d -> n*(d+1)
        # w: k*(d+1)
        # dE: k*(D+1)

        N, D = x_train.shape
        K = t_train.shape[1]

        w_init = np.zeros((K, D + 1))

        # Hinv = self.Hessian_inv(t_train, x_train, w_init)
        dE = self.direction(t_train, x_train, w_init)
        # dw = self.tensor_multiplier(Hinv, dE, N, K, D)

        dw = dE
        w = w_init - yita * dw

        E = self.loss(t_train, x_train, w)
        print('E = ', E)
        count = 1

        E_temp = E

        while 1:
            # w_temp = w

            # Hinv = self.Hessian_inv(t_train, x_train, w)
            dE = self.direction(t_train, x_train, w)
            # dw = self.tensor_multiplier(Hinv, dE, N, K, D)
            dw = dE

            w = w - yita * dw

            E = self.loss(t_train, x_train, w)
            # print('E = ', E)

            # if np.all(self.norm_1d(dw, 1) <= abs_err):
            if abs(E_temp - E)/E < rel_err and count > 10:
                print('Yayyyyyyyy!')
                print('count = ', count)
                print('delta E = ', abs(E_temp - E) )
                break
            elif count >= max_count:
                print("Count is up. I'm out!")
                print('count = ', count)
                break
            else:
                count = count + 1
                if not (count % 1e2):
                    print('E = ', E)
                    E_temp = E
                    # print('|w_new - w_old| = ', linalg.norm(w - w_temp))
                    # print('yita = ', yita)
        return w


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
    K = max(t_train)
    t = np.zeros((N, K+1))
    for i in range(K):
        t[t_train == i, i] = 1

    return t


def main():
    b_or_d = 'd'
    x_train, x_test, t_train, t_test = LR().extract_data(b_or_d)
    # For d: rel_err = 1e-5
    # For b: rel_err = 1e-(>5)

    t_train = transform_to_hotkey(t_train)

    yita = 0.05 # 0.01 opt 0.1 max
    max_count = int(1e6)

    w = LR().IRLS_basic(x_train, t_train, yita = yita, max_count = max_count, rel_err=1e-6)
    print('yita = ', yita)

    t_test = transform_to_hotkey(t_test)
    err, result = LR().validation(w, t_test, x_test)
    print('err= ', err)
    print('result', result)
    print('b_or_d ', b_or_d)


if __name__ == '__main__':
    main()