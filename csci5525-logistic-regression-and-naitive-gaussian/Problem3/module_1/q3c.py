from sklearn.datasets import load_digits
import numpy as np
from numpy import linalg as linalg
import math
import matplotlib.pyplot as plt


def hw1p3c(train_data, train_target, test_data, test_target, plot_gate):
    # digits = load_digits()
    # d_data = digits.data
    # d_target = digits.target
    #
    # divider = 1797-100
    #
    # x_train = d_data[:divider,:] # N*D
    # x_test = d_data[divider:,:]
    #
    # t_train = d_target[0:divider]
    # t_test = d_target[divider:]

    x_train = train_data # N*D
    x_test = test_data

    t_train = train_target
    t_test = test_target


    classes = t_train==0
    for idx in range(1,10,1):
        class_i = t_train==idx
        classes = np.vstack((classes,class_i))

    n_elem = np.sum(classes,1)

    # cla = classes[0]
    # x_train[cla].shape
    # classes.shape

    means = np.zeros((10,64))
    for idx,cla in enumerate(classes):
        # print(x_train[cla].shape)
        means[idx] = np.mean(x_train[cla], 0)

    mu = np.mean(x_train,0)

    Sw = np.zeros((64,64))
    for idx in range(0,10,1):
        Sw_i = np.dot((x_train[classes[idx]] - means[idx]).transpose(), (x_train[classes[idx]] - means[idx]))
        Sw = Sw + Sw_i

    Sb = np.zeros((64,64))
    for idx in range(0,10,1):
        Sb = Sb + n_elem[idx]*np.dot( (x_train[classes[idx]] - mu).transpose(),x_train[classes[idx]]-mu )

    A = np.dot(linalg.pinv(Sw),Sb)
    lam,vec = linalg.eig( A )
    w = vec[:,lam>0.0000001]
    w = w[:,0:2] # D*K=2
    w = w/linalg.norm(w)




    # Now project it onto 2D and histogram the projected points:

    proj_data = dict()

    if plot_gate:
        plt.subplot(1,2,1)
    for labName,cl in enumerate(classes):
        proj_data_i= np.dot(x_train[cl],w) # N,D * D,K = N,K
        x,y = np.transpose(proj_data_i) # k1,k2 = K*N
        if plot_gate:
            plt.scatter(x,y,label = ("class"+str(labName)))
        proj_data[labName] = proj_data_i   # proj_data: C * N*K
    if plot_gate:
        plt.legend()
        plt.title("Training set's projected points in the last run of validation")


    if plot_gate:
        plt.subplot(1,2,2)
    for labName,cl in enumerate(classes):
        projdata = np.dot(x_test[np.where(t_test == int(labName))],w) # N,D * D,K = N,K
        x,y = np.transpose(projdata ) # k1,k2 = K*N
        if plot_gate:
            plt.scatter(x,y,label = ("class"+str(labName)))
        # proj_data[labName] = projdata    # proj_data: C * N*K
    if plot_gate:
        plt.legend()
        plt.title("Testing set's projected points in the last run of validation")





    # Gaussian generative modeling:
    def gaus_param(X):
        # X is of N*K
        N = len(X)
        mu = np.mean(X,0) # 1*K
        Sigma = np.dot( np.transpose(X-mu), X-mu) / (N-1) # K*K
        return mu,Sigma


    mu = dict() # N=1 * K=2
    Sigma = dict()
    prior = dict() # prior probability estimator
    for idx in proj_data:
        mu[idx],Sigma[idx] = gaus_param(proj_data[idx])
        prior[idx] = n_elem[idx]/sum(n_elem)



    def disc(mu,Sigma,prior,test):
        # This is a gasussian discriminant on 2D
        # test: N*K
        # mu: 1*K
        # Sigma: K*K
        # return: int class 0-9 N*1
        if test.ndim <2:
            test = test[None,:]

        n_class = len(mu)
        result = np.zeros(len(test))
        for ord,x in enumerate(test): # x: 1*K
            logp = np.zeros(n_class)
            for idx in range(0,n_class,1):
                # distance = 0.5 * np.dot(    (x-mu[idx])     )
                logp[idx] = math.log(prior[idx]) - 0.5*math.log( linalg.det(Sigma[idx]) ) - 0.5*  np.dot(  ( x-mu[idx]),np.dot( linalg.inv(Sigma[idx]), (x-mu[idx]).transpose() ))
            result[ord] = np.argmax(logp)
        return result


    proj_test = np.dot(x_test[:],w) # N*D. D*K = N*K
    result_test = disc(mu,Sigma,prior,proj_test)

    err = 1-np.mean(result_test==t_test[:])
    y_test = t_test



    proj_test = np.dot(x_train[:],w) # N*D. D*K = N*K
    result_train = disc(mu,Sigma,prior,proj_test)

    err_train = 1-np.mean(result_train==t_train[:])
    y_train = t_train


    return err, err_train, y_test, result_test, y_train, result_train