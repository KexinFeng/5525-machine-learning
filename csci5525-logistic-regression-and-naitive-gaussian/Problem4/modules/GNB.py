# from sklearn.datasets import load_digits,load_boston
import numpy as np
from numpy import linalg as linalg
import math

# digits = load_digits()
# boston = load_boston()
# d_data = digits.data
# b_data = boston.data
# d_target = digits.target
# b_target = boston.target
#
# t0 = np.median(b_target)
# target = np.zeros(len(b_target))
# target[b_target <= t0 ] = 1
# b_target = target
#
# divider = 1797-100
#
#
#
# x_train = d_data[:divider,:] # N*D
# x_test = d_data[divider:,:]
#
# t_train = d_target[0:divider]
# t_test = d_target[divider:]
###############################

# b_divider =  406
#
# x_train = b_data[:b_divider,:] # N*D
# x_test = b_data[b_divider:,:]
#
# t_train = b_target[0:b_divider]
# t_test = b_target[b_divider:]

def GNB(train_data, train_target, test_data, test_target):
    #### Function begin:
    # x_train = x_train[None,:,:] # 1*N*D
    x_train = train_data
    t_train = train_target
    x_test = test_data
    t_test = test_target

    max_t = int(max(t_train))
    min_t = int(min(t_train))
    n_class = max_t-min_t+1
    N, D = (x_train.shape)
    # K = n_class


    classes = t_train ==0
    for idx in range(min_t+1,max_t+1,1):
        class_i = t_train==idx
        classes = np.vstack((classes,class_i))
    n_elem = np.sum(classes, 1) # classes: K*N*1

    # x_train_parted = x_train[classes[0]] # N*D
    # x_train_parted = x_train_parted[None,:,:] # K*N*D
    x_train_parted = dict()
    for idx in range(min_t, max_t+1, 1):
        x_train_parted[idx] = x_train[classes[idx]]
    # print('partitioned shape:', x_train_parted.shape)

    pi = n_elem/np.sum(n_elem)
    # pi = pi[:,None,None]

    mu = np.zeros((n_class,D))
    for cl in x_train_parted:
        mu[cl] = np.mean(x_train_parted[cl],0)
    # mu = np.mean(x_train_parted,1)
    mu.shape
    mu = mu[:,None,:] # K*N*D

    S = np.zeros((D,D ))
    for idx,number in enumerate(n_elem):
        Sk = np.dot( (x_train_parted[idx]-mu[idx]).transpose(),  (x_train_parted[idx]-mu[idx]) )
        S = S + Sk
    S = S/sum(n_elem)
    ## training ends



    ## begin testing:

    # y = np.zeros((n_class,N))
    # for idx,number in enumerate(n_elem):
    #     y[idx] = math.log(pi[idx]) - 0.5 * np.diag((x_train_parted[idx]-mu[idx]).dot(
    #         linalg.pinv(S).dot((x_train_parted[idx]-mu[idx]).transpose()))
    #     ) # -0.5*math.log(linalg.det(S))

    # x_test: N*D

    mu = mu[:,0,:] # K*D
    y = np.zeros((x_test.shape[0], n_class))
    logpi = np.zeros(len(pi))
    for i in range(len(pi)):
        logpi[i] = math.log(pi[i])
    for n, x in enumerate(x_test):
        # x : N=1*D (D,)
        y[n] = logpi - 0.5 * np.diag((x[None,:] - mu).dot(
            linalg.pinv(S).dot(
                (x[None, :] - mu).transpose()
            )
        ))
    result = np.argmax(y, 1)

    # for n,x in enumerate(x_test):
    #     for k in range(n_class):
    #         y[n,k] = logpi[k] - 0.5 * np.dot((x - mu[k]), np.dot(linalg.pinv(S),(x - mu[k]).transpose()))
    # result = np.argmax(y,1)

    err = 1 - np.mean( result == t_test )
    # print('err_rate',err)
    # print('result', result)

    return err

