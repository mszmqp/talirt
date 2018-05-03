# coding=utf-8

'''
Created on 2018年1月27日
@author: mengshuai@100tal.com
function: matrix_factorization_accu_pred
'''

import numpy as np
from numpy import dot
from numpy import mean, mat
import matplotlib.pyplot as plt
from reco_accu_pred.evaluate.evaluate_ import get_mse
from reco_accu_pred.utils.data_process import modify


# 确认矩阵分解的维数
def nmf(train, test):
    '''
    :param tarin:
    :param test:
    :return  the best key
    '''
    import seaborn as sns
    import numpy as np
    from sklearn.decomposition import NMF
    pal = sns.color_palette("Set2", 2)
    # X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    # print train[:4, :4]
    n_com = [i for i in range(1, train.shape[1], 1)]
    mse_li = []
    for k in n_com:
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(train)
        H = model.components_
        matr_mult = dot(W, H)

        # transform to int
        for i in range(matr_mult.shape[0]):
            for j in range(matr_mult.shape[1]):
                matr_mult[i, j] = modify(matr_mult[i, j])

        mse_ = get_mse(matr_mult, test)
        mse_li.append(mse_)

    a = min(mse_li)
    for item in range(len(mse_li)):
        if mse_li[item] == a:
            print(item)
            print(mse_li[item])
            break

    return item

    '''
    plt.plot(n_com, mse_li, c=pal[0], label='item-based test', linewidth=1)
    plt.xlabel('num of components')
    plt.ylabel('mse')
    plt.show()
    '''

def multy_mf(i, j, k, alpha, e, beta):

    P[i][k] = P[i][k] + alpha * (2 * e * Q[k][j] - beta * P[i][k])
    Q[k][j] = Q[k][j] + alpha * (2 * e * P[i][k] - beta * Q[k][j])

def matrix_factoriz(train, test, K, steps=100, alpha=0.006, beta=0.06):

    '''
    :param R:
    :param K:
    :param steps:
    :param alpha:
    :param beta:
    :return: P*Q
    '''

    # nmf_test
    N = len(train)
    M = len(train[0])
    global P, Q
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    mean_all = mean(train)
    b_stu = []
    b_item = []
    for i in range(train.shape[0]):
        b_stu.append(mean(train[i, :]) - mean_all)
    for i in range(train.shape[1]):
        b_item.append(mean(train[:, i]) - mean_all)
    Q = Q.T
    for step in range(steps):
        print(step)
        for i in range(len(train)):
            for j in range(len(train[i])):
                if train[i][j] != 0:
                    pre_val = np.dot(P[i, :], Q[:, j]) + mean_all + b_stu[i] + b_item[j]
                    eij = train[i][j] - pre_val
                    b_stu[i] = b_stu[i] + alpha * (2 * eij - beta * b_stu[i])
                    b_item[j] = b_item[j] + alpha * (2 * eij - beta * b_item[j])
                    # eij = train[i][j] - np.dot(P[i, :], Q[:, j])
                    # print cpu_count()
                    for k in range(K):
                        try:
                            '''
                            print i,j,k
                            t = threading.Thread(target=multy_mf, args=(i, j, k, alpha, eij, beta))
                            t.start()
                            t.join()
                            '''
                            multy_mf(i, j, k, alpha, eij, beta)
                        except:
                            print("Error: unable to start thread")

    matr_mult = dot(P, Q)

    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            matr_mult[i, j] = matr_mult[i, j] + mean_all + b_stu[i] + b_item[j]


    for j in range(matr_mult.shape[1]):
        for i in range(matr_mult.shape[0]):
            matr_mult[i, j] = modify(matr_mult[i, j])

    mse_ = get_mse(matr_mult, test)
    print("the MSE of the iterator is %s" % mse_)
    return matr_mult, mse_, P, Q.T
    # return P, Q, b_stu, b_item

# 修正matrix factorization,批量操作
def modify_mf(R, test, K, steps=1000, alpha=0.001, beta=0.02):

    '''
    :param R:
    :param K:
    :param steps:
    :param alpha:
    :param beta:
    :return: P, Q.T
    '''

    N = len(R)
    M = len(R[0])
    global P, Q
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    Q = Q.T
    # 保存行列为零的两组index
    row = []
    col = []

    for i in range(len(R)):
        temp = []
        for j in range(len(R[i])):
            if R[i, j] != 0:
                temp.append(j)
        row.append(temp)

    for j in range(len(R[0])):
        temp = []
        for i in range(len(R)):
            if R[i, j] != 0:
                temp.append(i)
        col.append(temp)


    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                ei = mat(R[i, row[i]] - np.dot(P[i, :], Q[:, row[i]]))
                ej = mat(R[col[j], j] - np.dot(P[col[j], :], Q[:, j]))
                for k in range(K):
                    a = mat(Q[k, row[i]])
                    b = mat(P[col[j], k])
                    print(ei * a.transpose())
                    P[i][k] = P[i][k] + alpha * (2 * ei * a.transpose() - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * ej * b.transpose() - beta * Q[k][j])

        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    matr_mult = dot(P, Q)

    for j in range(matr_mult.shape[1]):
        for i in range(matr_mult.shape[0]):
            matr_mult[i, j] = modify(matr_mult[i, j])

    mse_ = get_mse(matr_mult, test)
    print("the MSE of the iterator is %s" % mse_)
    return P, Q
