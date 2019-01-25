from sklearn.datasets import load_digits,load_boston

import numpy as np
from numpy import linalg as linalg
import matplotlib.pyplot as plt
import sys, os, re

from modules import LR_quasi_mD



class PrePro():

 
    @staticmethod
    def split(data, size_testS):
        N = data.shape[0]
        size = size_testS

        idxs = np.random.permutation(np.arange(N))
        test = data[idxs[:size], :]
        train = data[idxs[size:], :]

        return test, train


def batch_computation(x, t, num_splits , train_percent, b_or_d):

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

            if idx_per <= 1:
                yita = 10
            else:
                yita = 100

            errs[idx_spl, idx_per] = LR_quasi_mD.LR(x_train, t_train, x_test, t_test, b_or_d, yita)

    return errs



def logisticRegression(filename, num_splits, train_percent_str):


    # train_percent_temp= re.split(r'[\s,]+' ,train_percent_str.strip('[]\n\s'))
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



    errs = batch_computation(x, y,  num_splits, train_percent, b_or_d = b_or_d )

    print('errs', errs)

    means = np.mean(errs, 0)
    er_errs = np.std(errs, 0)

    print('Mean of test error rate:', means)
    print('Std of test error rate:', er_errs)

    output = np.concatenate(
        [means.reshape(1, len(train_percent)), er_errs.reshape(1, len(train_percent))])
    output_file = 'logisticRegression_' + os.path.basename(filename)
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

    # filename = './data/digits.csv'
    # num_splits = 10
    # train_percent = [10, 25, 50, 75, 100]
    #
    # logisticRegression(filename, num_splits, train_percent)

    logisticRegression(sys.argv[1], sys.argv[2], sys.argv[3])





