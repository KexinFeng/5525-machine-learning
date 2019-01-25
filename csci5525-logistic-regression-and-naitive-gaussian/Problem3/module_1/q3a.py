import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def hw1p3a(train_data, train_target, test_data, test_target, plot_gate):
    # from sklearn.datasets import load_boston
    # import numpy as np
    # from numpy import linalg as linalg

    # boston = load_boston()
    # b_data = boston.data
    # b_target = boston.target
    #
    # t0 = np.median(b_target)
    # divider = 406
    #
    # bDataTrainS = b_data[:divider,:]
    # bDataTestS = b_data[divider:,:]
    #
    # bTargetTrainS = b_target[:divider]
    # bTargetTestS = b_target[divider:]


    bDataTrainS = train_data
    bTargetTrainS = train_target

    bDataTestS = test_data
    bTargetTestS = test_target

    # class1 -> 0, > t0 ; class2 -> 1, <= t0;  result > 0-> 1-> class2
    class1 = bTargetTrainS == 0
    class2 = bTargetTrainS == 1

    N1 = sum(class1)
    N2 = sum(class2)


    m1 = np.mean(bDataTrainS[class1],0)[None,:] # 1*D
    m2 = np.mean(bDataTrainS[class2,:],0)[None,:]
    m = np.mean(bDataTrainS,0)[None,:]


    x1 = bDataTrainS[class1] # N * D=13
    x2 = bDataTrainS[class2]

    Sw = (x1-m1).transpose().dot((x1-m1)) + (x2-m2).transpose().dot(x2-m2)

    # w = linalg.inv(Sw).dot((m2-m1).transpose())
    w = linalg.pinv(Sw).dot((m2-m1).transpose())
    w = w/linalg.norm(w) # D*1
    # print(w)

    m = m.transpose() # D*1

    def classifier(x,win,mid):
        # x is N*D data matrix
        # x = x.reshape((D,1))

        x = x.transpose()
        D = len(win)
        win = win.reshape((D,1))
        mid = mid.reshape((D,1))
        # return np.inner(w,x)>0
        return (win.transpose().dot(x-mid))[0]>0 # D contracted, inner product on D, feature space.


    # test classifier:
    # a = np.array([np.ones([1,13]),np.ones([1,13])*2])[:,0,:]
    # print(a.shape)
    # print(classifier(a,w))

    # Histogram the projected points:
    # %matplotlib inline # jupyter notebook magical command
    # import matplotlib.pyplot as plt

    # x = [x1,x2,x1[:50]+x2[:50],x1[:50]*x2[:50]]
    x = [x1,x2]
    y = []

    if plot_gate:
        plt.figure()

        plt.subplot(2,1,1)
        for num,labName in zip([0,1],"01"):
            y_i = w.transpose().dot(x[num].transpose()-m)  # 1*N
            # print(y.shape)
            plt.scatter(y_i, np.ones(y_i.shape)*.5, label = ("class"+labName), cmap = plt.get_cmap('gist_rainbow'))
            y.append(y_i[0])
        plt.vlines(0,0,1)
        plt.legend()
        plt.title('Projected points of Training set in the last run of validation')


        plt.subplot(2,1,2)
        n_bins = 20
        plt.hist(y, n_bins, histtype= 'bar')
        # plt.title('Hist of projected points')
        # plt.legend()


    x11 = bDataTestS[np.where(bTargetTestS == 0)] # N * D=13
    x22 = bDataTestS[np.where(bTargetTestS == 1)]
    xx = [x11, x22]
    if plot_gate:
        plt.figure()

        plt.subplot(2,1,1)
        for num,labName in zip([0,1],"01"):
            y_i = w.transpose().dot(xx[num].transpose()-m)  # 1*N
            # print(y.shape)
            plt.scatter(y_i, np.ones(y_i.shape)*.5, label = ("class"+labName), cmap = plt.get_cmap('gist_rainbow'))
            # y.append(y_i[0])
        plt.vlines(0,0,1)
        plt.legend()
        plt.title('Projected points of Testing set in the last run of validation')


        plt.subplot(2,1,2)
        n_bins = 20
        plt.hist(y, n_bins, histtype= 'bar')
        # plt.title('Hist of projected points')
        # plt.legend()



    # Error rate:
    x_test = bDataTestS
    y_test = bTargetTestS
    result_test = classifier(x_test,w,m)

    err = np.mean(y_test != result_test)


    x_train = bDataTrainS
    y_train = bTargetTrainS
    result_train = classifier(x_train,w,m)

    err_train = np.mean(y_train != result_train)




    return err, err_train, y_test, result_test, y_train, result_train


