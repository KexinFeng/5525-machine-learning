from sklearn.datasets import load_digits,load_boston

import numpy as np
from numpy import linalg as linalg
import matplotlib.pyplot as plt
import sys, os, re

from modules import GNB



class PrePro():

    # @staticmethod
    # def extract_data(b_or_d = 'b'):
    #     digits = load_digits()
    #     boston = load_boston()
    #
    #     d_data = digits.data
    #     b_data = boston.data
    #     d_target = digits.target
    #     b_target = boston.target
    #
    #
    #     t0 = np.median(b_target)
    #     target = np.zeros(len(b_target), dtype=int)
    #     target[b_target <= t0] = 1
    #
    #     b_target = target
    #
    #     if b_or_d == 'b':
    #         data = b_data
    #         target = b_target
    #     elif b_or_d == 'd':
    #         data = d_data
    #         target = d_target
    #     else:
    #         print('b for Boston and d for Digits')
    #         data = np.array([])
    #         target = np.array([])
    #
    #     return data, target

    @staticmethod
    def split(data, size_testS):
        N = data.shape[0]
        size = size_testS

        idxs = np.random.permutation(np.arange(N))
        test = data[idxs[:size], :]
        train = data[idxs[size:], :]

        return test, train


def batch_computation(x, t, num_splits , train_percent):

    num_splits = int(num_splits)
    num_percent = len(train_percent)
    N,D = x.shape
    size_trainS = int(N * 0.8)
    size_testS = N - size_trainS

    errs = np.zeros((num_splits, num_percent))

    x_t = np.hstack((t.reshape((-1, 1)), x))



    for idx_spl in range(num_splits):

        test, train = PrePro().split(x_t, size_testS)

        for idx_per in range(num_percent):

            percent = train_percent[idx_per]
            use_n = int(N * percent)

            x_train = train[:use_n, 1:]
            t_train = train[:use_n, 0]
            x_test = test[:, 1:]
            t_test = test[:, 0]

            errs[idx_spl, idx_per] = GNB.GNB(x_train, t_train, x_test, t_test)

    return errs



def naiveBayesGaussian(filename, num_splits, train_percent_str):

    str = train_percent_str.strip('[]\n ')

    train_percent_temp= re.split(r'[\s,]+', str)

    train_percent = []
    for i in train_percent_temp:
        train_percent.append(int(i))

    # train_percent = [10, 25, 50, 75, 100]
    train_percent = np.array(train_percent) * 0.01

    print(train_percent)

    num_splits = int(num_splits)
    # num_percent = len(train_percent)


    data = np.genfromtxt(filename, delimiter=',')

    x = data[:, :-1]
    y = data[:, -1]


    if os.path.basename(filename) == 'boston.csv':
        t0 = np.median(y)
        target = np.zeros(len(y), dtype=int)
        target[y <= t0] = 1
        y = target
        b_or_d = 'b'
        print('Boston')

    elif os.path.basename(filename) == 'digits.csv':
        b_or_d = 'd'
        print('Digits')

    else:
        print('Please check the input.')



    errs = batch_computation(x, y,  num_splits, train_percent )

    print('errs', errs)

    means = np.mean(errs, 0)
    er_errs = np.std(errs, 0)

    print('Mean of test error rate:', means)
    print('Std of test error rate:', er_errs)


    output = np.concatenate(
        [means.reshape(1, len(train_percent)), er_errs.reshape(1, len(train_percent))])
    output_file = 'naiveBayesGauss_' + os.path.basename(filename)
    np.savetxt(output_file, output, delimiter=",")



if __name__ == "__main__":
    # filename = sys.argv[1]
    # num_crossval = sys.argv[2]
    # print(filename)
    # print(num_crossval)



    # if os.path.basename(filename) == 'boston.csv':
    #     print('b')
    # if os.path.isabs()

    # filename =  './data/boston.csv'
    # # filename = './data/digits.csv'
    # num_splits = 10
    # train_percent = [10, 25, 50, 75, 100]
    #
    # naiveBayesGaussian(filename, num_splits, train_percent)


    naiveBayesGaussian(sys.argv[1], sys.argv[2], sys.argv[3])





