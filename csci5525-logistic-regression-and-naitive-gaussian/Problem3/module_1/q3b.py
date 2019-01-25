def hw1p3b():
    from sklearn.datasets import load_boston
    import numpy as np
    from numpy import linalg as linalg

    boston = load_boston()
    b_data = boston.data
    b_target = boston.target

    t0 = np.median(b_target)
    divider = 406

    x_train = b_data[:divider,:] # N*D
    bDataTestS = b_data[divider:,:]

    target = b_target[0:divider]
    bTargetTestS = b_target[divider:]

    class1 = target <= t0
    class2 = target > t0
    N1 = sum(class1)
    N2 = sum(class2)

    m1 = np.mean(x_train[class1],0)[None,:] #1*D
    m2 = np.mean(x_train[class2],0)[None,:]
    m = np.mean(x_train,0)[None,:]

    #################################################

    Sw = (x_train[class1]-m1).transpose().dot( (x_train[class1]-m1) ) + (x_train[class2]-m2).transpose().dot( (x_train[class2]-m2) )
    # Sb1 = N1*(m1-m).transpose().dot( (m1-m) ) + N2*(m2-m).transpose().dot( (m2-m) )
    Sb = N1*N2/(N1+N2) * np.dot(np.transpose(m1-m2),(m1-m2))
    # print('allclose',np.allclose(Sb,Sb1))


    print('Sw.shape = ',Sw.shape)
    print('Sb.shape = ', Sb.shape)

    A = np.dot(linalg.inv(Sw),Sb)
    lam,vec = linalg.eig( A )
    w = vec[:,0:2] # D*2

    # test eigen vector vs Sw^-1(m1-m2):
    w1 = np.dot(linalg.inv(Sw),(m1-m2).transpose())
    w1 = w1/linalg.norm(w1)

    print('check w ~ w1: ',np.allclose(w1,w[:,:1]))
    print('w2:',w[:,1:])
    print('eigen_values:',lam.real)
    print('rank(Sw^-1.Sb', linalg.matrix_rank(A))


    # Histogram the projected points:
    import matplotlib.pyplot as plt

    classes = [class1,class2]

    f1 = plt.figure()
    for cl,labName in zip(classes,"01"):
        proj_data = np.dot(x_train[cl],w)
        x,y = np.transpose(proj_data)
        plt.scatter(x, y, label=("class" + labName), cmap=plt.get_cmap('gist_rainbow'))
    plt.legend()


    # Test that when w is 1D, the projected points restore to direct 1D result,ie Sw^-1*(m2-m1):
    print("Test whether or not restore to direct 1D result,ie Sw^-1*(m2-m1):")

    w = -vec[:,:1]  # w = vec[:,:1]; w.shape = (13,1); w = vec[:,0]; w.shape = (13,)
    # print('vec.shape',vec.shape)

    # f2 = plt.figure()
    # for cl,labName in zip(classes,"01"):
    #     proj_data = np.dot(x_train[cl],w)
    #     # x,y = np.transpose(proj_data)
    #     x = np.transpose(proj_data)
    #     y = np.zeros(x.shape)
    #     plt.scatter(x-np.dot(m,w),y, label = ("class"+labName), cmap = plt.get_cmap('gist_rainbow'))
    # plt.legend()
    # plt.vlines(0,0,1)
    # # plt.show()




    def classifier(x,win,mid):
        # x is N*D data matrix
        # x = x.reshape((D,1))

        x = x.transpose()
        D = len(win)
        win = win.reshape((D,1))
        mid = mid.reshape((D,1))
        # return np.inner(w,x)>0
        return (win.transpose().dot(x-mid))[0]>0 # D contracted, inner product on D, feature space.


    x = bDataTestS
    y = bTargetTestS
    result = classifier(x,w,m)

    err = np.mean((y>t0)!= result)

    print('err rate is: ',err)
    print('Linear dividable? ', np.mean(classifier(x_train,w,m) == (target > t0 )))

    # print('wb:',w.real)

    # plt.show()

if __name__ == '__main__':
    hw1p3b()













