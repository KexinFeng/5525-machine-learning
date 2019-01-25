from sklearn.datasets import load_digits,load_boston
import numpy as np
from numpy import linalg as linalg
import math

class LR1d():
    # Hessian: adaptive studying rate
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

        elem = (p - t)[:, None] * x  # (406,) cannot be broadcast to (406, 14)
                                     # (406,1) - (406,) = (406, 406)
        vector = sum(elem, 1) # 1,D

        return vector/linalg.norm(vector)


    def Hessian(self, t, x, w):
        # t: n*1
        # x: n*d
        # w: 1*(d+1)

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))

        p = self.sigmoid(w, x)
        R = np.diag(p * (1 - p))
        H = x.T.dot(
            R.dot(
                x
            )
        )

        r = linalg.matrix_rank(H)
        if r < D:
            print('H not inversible! rank = ', r)

        return H




    def IRLS_basic(self, x_train, t_train, yita, max_count = 500, rel_err = 1e-7):
        # w = [w, w_0]
        # x = [x, 1]

        N, D = x_train.shape

        w_init = np.zeros( D + 1 )
        # w_init = np.hstack((x_train[0,:],0))
        # w_init = w_init/linalg.norm(w_init)

        dw = yita * linalg.pinv(self.Hessian(t_train, x_train, w_init)).dot(
            self.direction(t_train, x_train, w_init)
        )
        w = w_init - dw

        E = self.loss(t_train, x_train, w)
        print('E = ', E)
        count = 1

        E_temp = E
        while 1:
            w_temp = w

            # w = w - yita * self.direction(t_train, x_train, w)
            dw = yita * linalg.pinv(self.Hessian(t_train, x_train, w)).dot(
                self.direction(t_train, x_train, w)
            )
            w = w - dw
            E = self.loss(t_train, x_train, w)


            # if linalg.norm(dw) < rel_err * linalg.norm(w):
            # if np.allclose(w, w_temp) and count > 10:
            if abs(E_temp - E)/E < rel_err and count > 100:
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
                    E_temp = E
                    # print('norm dw = ', linalg.norm(dw))
                    # print('|w_new - w_old| = ', linalg.norm(w - w_temp))
        return w


    def validation(self, w, t, x):
        # Validation:

        N, D = x.shape
        extra_x = np.ones((N, 1))
        x = np.hstack((x, extra_x))
        p_nk = self.sigmoid(w, x)

        result = np.zeros(N, dtype= int)
        result[p_nk >= 0.5] = int(1)

        err = 1 - np.mean(result == t)
        # print('err_rate_testingSet', err)
        return err, result


def main():
    b_or_d = 'b' # Fixed !
    x_train, x_test, t_train, t_test = LR1d().extract_data(b_or_d)

    yita = 10 # 0.005 breaks
    max_count = 1e5

    w = LR1d().IRLS_basic(x_train, t_train, yita = yita, max_count= max_count, rel_err = 1e-5)
    print('yita = ', yita)

    err, result = LR1d().validation(w, t_test, x_test)
    print('err= ', err)
    print('result = ', result)
    print('b_or_d:', b_or_d)



if __name__ == '__main__':
    main()