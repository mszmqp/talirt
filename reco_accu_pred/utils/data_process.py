# -*-coding:utf-8-*-

'''
Created on 2018年1月27日
@author:mengshuai@100tal.com
'''

import numpy as np
from pandas import DataFrame


def per_posi(stu_ans):

    '''
    :param stu_ans:
    :print: the percentage of positive sample
    '''
    len_1 = len(np.where(stu_ans == 1)[0])
    len_0 = len(np.where(stu_ans == -1)[0])

    tf = float(len_1) / (len_1 + len_0)
    tf *= 100
    print('the percentage of positive is {:4.2f}%'.format(tf))

def modify(score):

    return score

    if score >= 0:
        score = 1
    else:
        score = -1
    return score

    '''
    if score > 1.5:
        score = 2
    else:
        score = 1
    return score
    '''

def train_test_split(stu_ans):

    '''
    random split
    :param stu_ans:
    :return: train, test
    '''

    test = np.zeros(stu_ans.shape)
    train = stu_ans.copy()
    num = 0
    stu_lis = [i for i in range(stu_ans.shape[0])]
    stu_lis = np.random.choice(np.array(stu_lis).nonzero()[0], size=120, replace=False)
    lq_lis = [i for i in range(stu_ans.shape[1])]
    lq_lis = np.random.choice(np.array(lq_lis).nonzero()[0], size=1, replace=False)

    all_lis = [i for i in range(stu_ans.shape[0]*stu_ans.shape[1])]
    all_lis = np.random.choice(np.array(all_lis).nonzero()[0], size=1400, replace=False)
    reverse = [[int(x/stu_ans.shape[1]), x % stu_ans.shape[1]] for x in all_lis]


    count = 0
    for i in reverse:
        if stu_ans[i[0], i[1]] == -1 and count <= 250:
            count += 1
            continue
        train[i[0], i[1]] = 0.
        test[i[0], i[1]] = stu_ans[i[0], i[1]]
    print(np.all((train * test) == 0))
    # return train, test

    for user in range(stu_ans.shape[0]):
        len_ = len(stu_ans[user, :].nonzero()[0])
        len_ = len_/10
        # len_ = 10
        test_ratings = np.random.choice(stu_ans[user, :].nonzero()[0], size=int(len_), replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = stu_ans[user, test_ratings]
    # Test and training are truly disjoint
    # print(np.all((train * test) == 0))
    return train, test

def train_test(stu_ans):

    '''
    simulate split
    :param stu_ans:
    :return: train, test
    '''

    test = np.zeros(stu_ans.shape)
    train = stu_ans.copy()
    lis_sam = [i for i in range(stu_ans.shape[0])]
    slice = np.random.choice(np.array(lis_sam).nonzero()[0], size=200, replace=False)
    for user in range(stu_ans.shape[0]):
        if user in slice:
            continue
        test_ratings = stu_ans[user, :].nonzero()[0][-5:]
        train[user, test_ratings] = 0.
        test[user, test_ratings] = stu_ans[user, test_ratings]
    # Test and training are truly disjoint
    # print(np.all((train * test) == 0))
    return train, test

def save2csv(result, name):

    '''
    result to csv
    :param result, name:
    '''
    path_dir = './data/'
    result_ = DataFrame(result)
    result_.to_csv(path_or_buf=path_dir+name)

if __name__ == '__main__':
    pass