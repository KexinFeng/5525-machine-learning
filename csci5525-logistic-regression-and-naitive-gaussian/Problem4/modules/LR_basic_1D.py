from sklearn.datasets import load_digits,load_boston
import numpy as np
from numpy import linalg as linalg
import math

class LR():
    # basic: linear yita studying rate
    # 1D
    def extract_data(self, b_or_d = 'b', test_size = 100):
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


    def sigmoid(self, w, x):
        # w: 1*(d+1)
        # x: n*(d+1)

        N, D = x.shape
        # extra_x = np.ones((N, 1))
        # x = np.hstack((x, extra_x))

        a_n = x.dot(w.T)
        p_n = 1/(1 + np.exp(- a_n))

        return p_n     # n*1



    def loss(self, t, x, w):
        # t: n*1
        # x: n*d
        # w: 1*(d+1)

        # C1 -> t = 1, p;  C2 -> t = 0, 1-p

        N, D = x.shape
        x = np.hstack((x, np.ones((N, 1))))

        p = self.sigmoid(w, x)

        elem = t * np.log(p) + (1 - t) * np.log((1 - p))
        E = - np.sum(elem)

        return E


    def direction(self, t, x, w):
        # t: n*1
        # x: n*d
        # w: 1*(d+1)

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))

        p = self.sigmoid(w, x)

        # t = t[:,None]
        elem = (p - t)[:, None] * x  # (406,) cannot be broadcast to (406, 14)
                                     # (406,1) - (406,) = (406, 406)
        vector = sum(elem, 1) # 1,D

        return vector/linalg.norm(vector)


    def IRLS_basic(self, x_train, t_train, yita, max_count = 500, abs_err = 0.01):
        # w = [w, w_0]
        # x = [x, 1]

        N, D = x_train.shape

        w_init = np.zeros( D + 1 )
        # w_init = np.hstack((x_train[0,:],0))
        # w_init = w_init/linalg.norm(w_init)

        w = w_init - yita * self.direction(t_train, x_train, w_init)

        E = self.loss(t_train, x_train, w)
        print('E = ', E)
        count = 1

        while 1:
            w_temp = w
            dw = yita * self.direction(t_train, x_train, w)
            w = w - dw

            E = self.loss(t_train, x_train, w)
            # print('E = ', E)

            # if linalg.norm(dw) < abs_err:
            if np.allclose(w_temp, w):
                print('Yayyyyyyyy!')
                print('count = ', count)
                break
            elif count >= max_count:
                print("Count is up. I'm out!")
                print('count = ', count)
                break
            else:
                count = count + 1
                if not (count % 1e3):
                    print('E = ', E)
                    # print('|w_new - w_old| = ', linalg.norm(w - w_temp))
                    # print('yita = ', yita)

        return w


    def validation(self, w, t, x):
        # Validation:

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))
        p_nk = self.sigmoid(w, x)

        result = np.zeros(N)
        result[p_nk >= 0.5] = 1

        err = 1 - np.mean(result == t)
        print('err_rate_testingSet', err)
        return err, result



def main():
    x_train, x_test, t_train, t_test = LR().extract_data('b')

    yita = .004# 0.005 breaks
    max = 1e6

    w = LR().IRLS_basic(x_train, t_train, yita = yita, max_count= max)
    print('yita = ', yita)

    err, result = LR().validation(w, t_test, x_test)
    print('err= ', err)


if __name__ == '__main__':
    main()