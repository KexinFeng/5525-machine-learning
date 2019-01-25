from sklearn.datasets import load_digits,load_boston

import numpy as np
from numpy import linalg as linalg
import matplotlib.pyplot as plt
import sys, os

from module_1 import q3a as B_classifier
from module_1 import q3c as D_classifier



class PrePro():

    @staticmethod
    def extract_data(b_or_d = 'b'):
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

        if b_or_d == 'b':
            data = b_data
            target = b_target
        elif b_or_d == 'd':
            data = d_data
            target = d_target
        else:
            print('b for Boston and d for Digits')
            data = np.array([])
            target = np.array([])

        return data, target

    @staticmethod
    def k_fold_val(data, fold):
        N = data.shape[0]
        size = int(N / fold)
        # N = size*fold
        fold_list = []
        # idx = np.linspace(0, N-1, N)
        # np.random.shuffle(idx)
        # idx = idx.reshape((fold, -1))

        for i in range(fold):
            # fold_list.append(data[idx[i], :])
            fold_list.append(data[np.random.choice(N, size, replace=False), :])
        return fold_list


def cross_val(x, t, num_crossval, b_or_d):
    if b_or_d == 'b':

        # x, t = PrePro().extract_data(b_or_d)
        target_data = np.hstack((t.reshape((-1, 1)), x))

        folder_list = PrePro().k_fold_val(target_data, fold = num_crossval)

        errs = []
        errs_train = []
        for i in range(len(folder_list)):
            test = folder_list[i]
            train = []
            for j in range(len(folder_list)):
                if j == i:
                    continue
                else:
                    train.append(folder_list[i])
                # print(j, 'th round ends')

            train = np.concatenate(train)


            if i == len(folder_list) - 1:
                plot_gate = True
            else:
                plot_gate = False

            err, err_train, y_test, result_test, y_train, result_train\
                = B_classifier.hw1p3a(train[:, 1:], train[:,0], test[:, 1:], test[:, 0], plot_gate)
            errs.append(err)
            errs_train.append(err_train)




    elif b_or_d == 'd':

        # data, target = PrePro().extract_data(b_or_d)
        target_data = np.hstack((t.reshape((-1, 1)), x))

        folder_list = PrePro().k_fold_val(target_data, fold=num_crossval)

        errs = []
        errs_train = []
        for i in range(len(folder_list)):
            test = folder_list[i]
            train = []
            for j in range(len(folder_list)):
                if j == i:
                    continue
                else:
                    train.append(folder_list[i])
                # print(j, 'th round ends')

            train = np.concatenate(train)

            if i == len(folder_list) - 1:
                plot_gate = True
            else:
                plot_gate = False

            err, err_train, y_test, result_test, y_train, result_train\
                = D_classifier.hw1p3c(train[:, 1:], train[:, 0], test[:, 1:], test[:, 0],plot_gate)
            errs.append(err)
            errs_train.append(err_train)


    else:
        print('Error in cross_val')
        return 0


    return errs, errs_train





def LDA1dProjection(filename, num_crossval):

    num_crossval = int(num_crossval)
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


    errs, errs_train = cross_val(x, y, num_crossval, b_or_d = b_or_d)


    print("Testing set err rate:\n", errs)
    print('standard deviation of testing set error rate:\n', np.std(errs))
    print('Mean of testing set error rate:\n', np.mean(errs))

    print("Training set err rate:\n", errs_train)
    print('standard deviation of training set error rate:\n', np.std(errs_train))
    print('Mean of training set error rate:\n', np.mean(errs_train))







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
    # num_crossval = 10
    #
    # LDA1dProjection(filename, num_crossval)

    LDA1dProjection(sys.argv[1], sys.argv[2])



    plt.show()