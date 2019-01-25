
import numpy as np
from numpy import linalg as linalg
import matplotlib.pyplot as plt
import sys, os, re
from module.svm_prime import SVMprime


def cross_val(filename, C):
    readin = np.genfromtxt(filename, delimiter = ',')

    data = readin[:, 1:]
    target = readin[:, 0].astype(int)
    num_split = 10

    err_mean_test, err_std_test, w_modes, w_std, num_sup, sup_std, num_marg, marg_std = n_cross_val(data, target, C, num_split)

    print(" ")
    print('Mean test error: ', err_mean_test)
    print('Std of test error: ', err_std_test)

    print(" ")


    return err_mean_test, err_std_test, w_modes, w_std, num_sup, sup_std, num_marg, marg_std





def n_cross_val(X, t, C, num_split):
    num_split = int(num_split)
    N, D = X.shape
    size_testS = int(N / num_split)
    size_trainS = N - size_testS

    err_rates = np.zeros(num_split)
    w_modes = np.zeros(num_split)
    num_sups = np.zeros(num_split)
    num_margs = np.zeros(num_split)


    idx = np.arange(0, N, dtype=int)
    np.random.shuffle(idx)

    for split in range(0, num_split):
        Xtrain = X[np.remainder(idx, num_split) != split, :]
        ttrain = t[np.remainder(idx, num_split) != split]
        Xtest = X[np.remainder(idx, num_split) == split, :]
        ttest = t[np.remainder(idx, num_split) == split]

        model = SVMprime(Xtrain, ttrain, C)
        model.train()

        err_rates[split] = model.valid(Xtest, ttest)
        w_modes[split] = np.linalg.norm( model.w)
        num_sups[split] = model.num_sup
        num_margs[split] = model.num_marg


    return np.mean(err_rates), np.std(err_rates), np.mean(w_modes), np.std(w_modes), np.mean(num_sups), np.std(num_sups), np.mean(num_margs), np.std(num_margs)




def main(argv=sys.argv):
    if len(argv) == 3:
        filename = argv[1]
        C = float(argv[2])
        cross_val(filename, C)
    else:
        print("Usage:"+'$ python myDualSVM.py <dataFile> <regulatorC>')
        sys.exit(1)

if __name__ == "__main__":

    main()


