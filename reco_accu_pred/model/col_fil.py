# coding=utf-8

'''
Created on 2018年1月27日
@author: mengshuai@100tal.com
function: matrix_factorization_accu_pred
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
from scipy.cluster import hierarchy as sch
from reco_accu_pred.evaluate.evaluate_ import get_mse, multy_key_in
from reco_accu_pred.utils.data_process import modify


def pear_simi():
    '''
    :return: user_simi, item_simi
    '''

    dir_input1 = 'P.csv'
    dir_input2 = 'Q.csv'
    path_data = './data/'
    dir_input1 = path_data + dir_input1
    dir_input2 = path_data + dir_input2
    user = pd.read_csv(dir_input1, sep=',')
    item = pd.read_csv(dir_input2, sep=',')
    user_simi = np.corrcoef(user)
    item_simi = np.corrcoef(item)
    return user_simi, item_simi


def eulidSim(stu_ans, kind, epsilon = 1e-9):

    user_similarity = np.zeros((stu_ans.shape[0], stu_ans.shape[0]))
    item_similarity = np.zeros((stu_ans.shape[1], stu_ans.shape[1]))

    if kind == 'user':
        for i in range(stu_ans.shape[0]):
            print('user: ', i)
            for j in range(stu_ans.shape[0]):
                all = 0
                diff = 0
                for m in range(stu_ans.shape[1]):
                    if stu_ans[i, m]+stu_ans[j, m] != 0:
                        all += 1
                        if stu_ans[i, m] != stu_ans[j, m]:
                            diff += 1
                        else:
                            pass
                    else:
                        pass
                user_similarity[i, j] = 1-float(diff)/(all+epsilon)
        return user_similarity

    elif kind == 'item':
        for i in range(stu_ans.shape[1]):
            print('item: ', i)
            for j in range(stu_ans.shape[1]):
                all = 0
                diff = 0
                for m in range(stu_ans.shape[0]):
                    if stu_ans[m, i]+stu_ans[m, j] != 0:
                        all += 1
                        if stu_ans[m, i] != stu_ans[m, j]:
                            diff += 1
                        else:
                            pass
                    else:
                        pass
                item_similarity[i, j] = 1-float(diff)/(all+epsilon)

        return item_similarity


def fast_similarity(stu_ans, kind, epsilon=1e-9):

    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = stu_ans.dot(stu_ans.T) + epsilon
    elif kind == 'item':
        sim = stu_ans.T.dot(stu_ans) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def hier_algo(Stu_ans):

    disMat = sch.distance.pdist(Stu_ans, 'cosine')
    # 进行层次聚类:
    Z = sch.linkage(disMat, method='average')
    # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # P = sch.dendrogram(Z)
    # plt.savefig('plot_dendrogram.png')
    # 根据linkage matrix Z得到聚类结果:
    cluster = sch.fcluster(Z, t=1, criterion='inconsistent')

    for i in cluster:
        print(i)
    print("Original cluster by hierarchy clustering:\n", cluster)


def predict_fast_simple(stu_ans, similarity, kind):
    global slice
    if kind == 'user':
        user_pre = similarity.dot(stu_ans) / np.array([np.abs(similarity).sum(axis=1)]).T
        # transform to int
        for i in range(user_pre.shape[0]):
            for j in range(user_pre.shape[1]):
                user_pre[i, j] = modify(user_pre[i, j])
        return user_pre
    elif kind == 'item':
        item_pre = stu_ans.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        # transform to int
        for i in range(item_pre.shape[0]):
            for j in range(item_pre.shape[1]):
                item_pre[i, j] = modify(item_pre[i, j])

        return item_pre

def predict_topk(ratings, similarity, kind, k):
    global slice
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            # 返回前k个学生序号
            top_k_users = [np.argsort(abs(similarity[:, i]))[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))

                # transform to int
                pred[i, j] = modify(pred[i, j])

    if kind == 'item':
        for j in range(ratings.shape[1]):
            # 返回前k个题目序号
            top_k_items = [np.argsort(abs(similarity[:, j]))[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
                # transform to int
                pred[i, j] = modify(pred[i, j])

    return pred


def predict_nobias(ratings, similarity, kind):

    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]


    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = modify(pred[i, j])

    return pred


def predict_topk_nobias(ratings, similarity, kind, k):

    # global slice

    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(abs(similarity[:, i]))[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]

    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(abs(similarity[:, j]))[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
        pred += item_bias[np.newaxis, :]

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = modify(pred[i, j])
    return pred

def mse_plo(item_similarity, user_similarity, train, test):

    '''
    search for the best threshold of the top-k of nobias-col_fil
    :param item_similarity:
    :param user_similarity:
    :return:
    '''
    import seaborn as sns
    user_test_mse = []
    user_test_acc = []
    item_test_mse = []
    item_test_acc = []

    print('user')
    for k in range(1, user_similarity.shape[0], int(user_similarity.shape[0]/10)):
        # print(k)
        # user_pred = predict_topk(train, user_similarity, kind='user', k=k)
        user_pred = predict_topk_nobias(train, user_similarity, kind='user', k=k)
        user_test_mse.append(get_mse(user_pred, test))
        user_test_acc.append(multy_key_in(user_pred, test))

    print('item')
    for k in range(1, item_similarity.shape[0], int(item_similarity.shape[0]/10)):
        # print(k)
        # item_pred = predict_topk(train, item_similarity, kind='item', k=k)
        item_pred = predict_topk_nobias(train, item_similarity, kind='item', k=k)
        item_test_mse.append(get_mse(item_pred, test))
        item_test_acc.append(multy_key_in(user_pred, test))

    sns.set()

    pal = sns.color_palette("Set2", 2)

    item_array = [i for i in range(1,  int(item_similarity.shape[0]/10))]
    user_array = [i for i in range(1, int(user_similarity.shape[0]/10))]

    a = min(item_test_mse)
    b = min(user_test_mse)

    i = 0
    j = 0
    print("the best thresholds of user-based: ")
    for item in range(len(user_test_mse)):
        if user_test_mse[item] == b:
            print(user_array[item])
            i = user_array[item]
            break
    print("the min of mse of user-based is ", b)
    print("the max of accuracy is ", user_test_acc[i])

    print("the best thresholds of item-based: ")
    for item in range(len(item_test_mse)):
        if item_test_mse[item] == a:
            print(item_array[item])
            j = item_array[item]
            break
    print("the min of mse of item-based is ", a)
    print("the max of accuracy is ", item_test_acc[j])

    return user_array[item], item_array[item]
    '''
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.plot(user_array, user_test_mse, c=pal[1], label='user-based test', linewidth=1)
    plt.ylabel('mse')
    plt.xlabel('num of top users')

    plt.subplot(122)
    plt.plot(item_array, item_test_mse, c=pal[0], label='item-based test', linewidth=1)
    plt.ylabel('mse')
    plt.xlabel('num of top items')
    plt.show()
    '''


if __name__ == '__main__':

    pear_simi()
