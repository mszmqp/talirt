# -*-coding:utf-8-*-

'''
Created on 2018年1月27日
@author:mengshuai@100tal.com
run cf_algorithm
'''

import os
import time
import csv
import math
import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import ibis
import pandas as pd
import random
from math import sqrt
from pandas import DataFrame
from numpy import linalg as la
from reco_accu_pred.model.matrix_factorization import *
from reco_accu_pred.evaluate.evaluate_ import *
from reco_accu_pred.model.col_fil import *
from reco_accu_pred.utils.data_process import *

sys.path.append("../")

path_data = './data/'

class CF:
    pass


class IRT:
    pass



class RecommendABC(object):

    def __init__(self,city_id,grade_id,subject_id,level_id,):
        pass

    def _xxx(self):
        pass


    def fectch_data(self):
        """
        模型包括候选题目数据，以及班型所有学生
        Returns
        -------
            格式？
        """
        pass

    def fit(self):
        """

        训练
        模型包括候选题目数据，以及班型所有学生
        Returns
        -------

        """

        pass
    def seri(self):
        # cPickle
        # sklearn joblib
        # 二进制，或者json
        pass

    def save_redis(self):
        # 序列化的结果保存到一个位置（redis）
        pass
    def save_file(self):

        # 序列化的结果保存到一个位置（redis）
        pass
    @classmethod
    def load_from_redis(self,model):

        # return self object
        pass

    @classmethod
    def load_from_file(self, model):
        # return self object
        pass

    def predict(self,stu_id,items=None):
        """

        Parameters
        ----------
        stu_id
        items : 可以为空，已经作答过的题目不用返回

        Returns
        -------

        """





class RecommendCF(RecommendABC):

    def __init__(self):
        pass
        self.model =  CF()


    def _xxx(self):
        pass


def obtain_data(from_cache):

    '''
    :param cache_file: 缓存位置
    :param from_cache: 是否从缓存读取
    :return: 学生答题画像
    '''

    cache_file = "hz_stu_top.pickle"
    cache_file = path_data + cache_file
    sql1 = """
            select 
            sa.sa_stu_id as stu_id,
            sa.sa_lq_id as lq_id,
            sa.sa_create_time as crea_time,
            sa.sa_answer_status as status
            from
            (
                --选择答题数据
                select 
                s.*
                from
                (
                    select 
                    sa_stu_id,
                    sa_lq_id,
                    sa_answer_status,
                    sa_c_id, 
                    sa_cl_id,
                    sa_create_time,
                    sa_city_code,
                    sa_grd_id
                    from odata.ods_ips_tb_stu_answer sa
                    where 
                    sa.sa_year='2018' 
                    and sa.sa_subj_id ='ff80808127d77caa0127d7e10f1c00c4' -- 限定学科
                    and sa.sa_grd_id = '7' -- 限定年级
                    and sa.sa_lev_id = 'ff8080812fc298b5012fd3d3becb1248'  -- 限定班型
                    and sa.sa_city_code = '0571' -- 限定城市
                    and sa.sa_term_id = '1'
                    and sa.sa_create_time is not null
                ) s
            ) sa
            order by sa.sa_stu_id,
            sa.sa_create_time
    """

    if from_cache:
        # print >> sys.stderr, "从缓存读取题目画像数据"
        print("从缓存读取题目画像数据", file=sys.stderr)
        return pd.read_pickle(cache_file)
    print("从impala读取题目画像数据", file=sys.stderr)
    ibis.options.sql.default_limit = None
    impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')
    df_question = impala_client.sql(sql1).execute()
    df_question.to_pickle(cache_file)
    impala_client.close()
    print("count:", len(df_question), file=sys.stderr)
    return df_question

def obtain_hive():
    '''
    读取hive暂时没有用到
    :return:
    '''

    from pyhive import hive
    conn = hive.Connection(host='192.168.23.236', port=21050, username='app_bi')

    sql = """
    select
    sa.fk_student as stu_id, 
    sa.fk_question as lq_id, 
    case when sa.asw_first_status='错误' then 0 else 1 end as status
    from dwdb.dwd_stdy_ips_level_answ sa 
    where sa.qst_type_status='客观题' 
    and sa.asw_first_status in ('正确','错误') 
    and sa.fk_year='2017' 
    and sa.city_name = '成都' 
    and sa.grd_name='小学六年级' 
    and sa.term_name='暑期班' 
    and sa.subj_name='数学' 
    and sa.lev_name='尖子班'
    and sa.cl_name='课后测'
    """

    conn.execute(sql)
    for result in conn.fetchall():
        print(result)


def data_from_csv():
    '''
    read data from csv
    '''
    dir_input = 'data_hive.csv'
    dir_input = path_data + dir_input
    names = ['stu_id', 'lq_id', 'status']
    data = pd.read_csv(dir_input, sep=',', names=names)
    return data


def pre_data(data):

    '''
    :param data:
    :return: 与处理之后的学生答题画像
    '''

    users = pd.DataFrame({
        'stu_id': data['stu_id'],
        'lq_id': data['lq_id'],
        'status': data['status']
    })

    print('the count of student is ' + str(len(users['stu_id'].unique())))
    print('the count of item is ' + str(len(users['lq_id'].unique())))

    # 剔除含有答题数小于设定要求的同学答题数据
    # 保留删除的序号，用于后续保留数据的删选
    del_lis = []
    del_group = users.groupby('stu_id').size()
    del_num = 0
    for item in del_group:
        if item < 10:
            del_lis.append(del_num)
        del_num += 1

    dic_user = {}
    del_stu = 0
    res_stu = 0

    # 创建学生字典、保留的最后序号
    for user in users['stu_id'].unique():
        if not del_stu in del_lis:
            dic_user[user] = res_stu
            res_stu += 1
        del_stu += 1

    # 创建试题字典
    dic_lq = {}
    init = 0
    # 创建学生字典
    for lq in users['lq_id'].unique():
        dic_lq[lq] = init
        init += 1
    # print dic_lq

    # 替换2为0
    # users.status = users.status.replace(2, 0)

    n_users = len(dic_user)
    n_lq_id = users.lq_id.unique().shape[0]
    stu_ans = np.zeros((n_users, n_lq_id))
    # print(stu_ans.shape)

    for row in users.itertuples():
        # print(row[1], row[2], row[3])
        if row[3] in dic_user:
            # stu_ans[dic_user[row[3]], dic_lq[row[1]]] = row[2]
            stu_ans[dic_user[row[3]], dic_lq[row[1]]] = 1 if row[2] == 1 else -1
    sparsity = float(len(stu_ans.nonzero()[0]))
    sparsity /= (stu_ans.shape[0] * stu_ans.shape[1])
    sparsity *= 100
    print('Sparsity: {:4.2f}%'.format(sparsity))

    per_posi(stu_ans)
    return stu_ans

def ran_mse():
    ran_test = test.copy()
    mse_list = []

    for step in range(100):
        for i in range(test.shape[0]):
            for j in range(test.shape[1]):
                if test[i, j] != 0:
                    if np.random.rand(1) >= 0.5:
                        ran_test[i, j] = 1
                    else:
                        ran_test[i, j] = -1
        temp_ = get_mse(ran_test, test)
        mse_list.append(temp_)

    print('MSE of random guess is %s' % np.array(mse_list).mean(axis=0))

    '''
    plt.hist(temp_, bins=7)
    plt.title('mse distribution')
    plt.xlabel('mse')
    plt.ylabel('count')
    plt.show()
    '''

if __name__ == '__main__':

    global train, test
    # 获得答题数据，默认是impala。
    data = obtain_data(from_cache=False)
    # 从csv获取数据
    # data = data_from_csv()
    # 数据预处理：数据字典、数据删除以及数据转化
    stu_ans = pre_data(data)
    # 数据拆分为训练集与测试集
    train, test = train_test_split(stu_ans)
    hier_algo(train)


    # 仿真实验下的数据拆分
    # train, test = train_test(stu_ans)
    per_posi(train)
    per_posi(test)

    # 随机猜测所得MSE
    ran_mse()

    # nmf and get the min of mse
    train_copy = train.copy()
    test_copy = test.copy()

    # train_copy[train_copy[:] == -1] = 2
    # test_copy[test_copy[:] == -1] = 2
    # k_nmf = nmf(train_copy, test_copy)

    '''
    k_nmf = 20
    result_nmf, mes_nmf, P, Q = matrix_factoriz(train_copy, test_copy, k_nmf)
    multy_key_in(result_nmf, test_copy)
    # 保存算法输出结果
    save2csv(result_nmf, name='result.csv')
    save2csv(P, name='P.csv')
    save2csv(Q, name='Q.csv')
    '''

    grid_thres = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thres_grid = [1-i for i in grid_thres]

    for i in range(9):
        print('\n\n')
        print(i)
        if i == 0:
            continue
            # 计算矩阵分解的用户试题相似性
            user_similarity, item_similarity = pear_simi()
            user_simi1 = user_similarity
            item_simi1 = item_similarity

        elif i == 1:
            # c-f algorithm
            user_similarity = fast_similarity(train, kind='user')
            item_similarity = fast_similarity(train, kind='item')
            user_simi2 = user_similarity
            item_simi2 = item_similarity
            # print(user_similarity.shape)
            # print(item_similarity.shape)
        else:
            break
            user_similarity = grid_thres[i-2]*user_simi1 + thres_grid[i-2]*user_simi2
            item_similarity = grid_thres[i-2]*item_simi1 + thres_grid[i-2]*item_simi2


        # choose the best threshold of k in user-based and item-based.
        k_stu, k_lq = mse_plo(item_similarity, user_similarity, train, test)
        print('\n')

        '''
        # 学生偏见控制
        pred_usr = predict_topk_nobias(train, user_similarity, kind='user', k=k_stu)
        mse_usr = get_mse(pred_usr, test)
        print('nobias User-based CF MSE: ' + str(mse_usr))
        # four_key
        multy_key_in(pred_usr, test)

        # 试题偏见控制
        pred_item = predict_topk_nobias(train, item_similarity, kind='item', k=k_lq)
        mse_item = get_mse(pred_item, test)
        print('nobias item-based CF MSE: ' + str(mse_item))
        # four_key
        multy_key_in(pred_item, test)
        '''


        '''
        # the integrated algorithm
        pred_final = (mes_nmf*result_nmf + mse_usr*pred_usr + mse_item*pred_item) / (mes_nmf + mse_usr + mse_item)
        # pred_final = (result_nmf + pred_usr + pred_item)/3
        mse_final = get_mse(pred_final, test)
        print('integrated algorithm  MSE: ' + str(mse_final))
        multy_key_in(pred_final, test)
        '''


    '''
    #user_prediction = predict_fast_simple(train, user_similarity, kind='user')
    user_prediction = predict_topk(train, user_similarity, kind='user', k=301)

    #item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    item_prediction = predict_topk(train, item_similarity, kind='item', k=36)

    print('User-based CF MSE: ' + str(get_mse(user_prediction, test)))
    multy_key_in(user_prediction, test)
    print('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))
    multy_key_in(item_prediction, test)
    '''


    '''
    # note_1
    user_similarity = eulidSim(train, kind='user')
    user_similarity = DataFrame(user_similarity)
    user_similarity.to_csv('./12_user_similarity.csv')

    item_similarity = eulidSim(train, kind='item')
    item_similarity = DataFrame(item_similarity)
    item_similarity.to_csv('./12_item_similarity.csv')

    user_similarity = pd.read_csv('./14_user_similarity.csv')
    item_similarity = pd.read_csv('./14_item_similarity.csv')

    user_similarity = user_similarity.as_matrix(columns=None)
    item_similarity = item_similarity.as_matrix(columns=None)
    '''
