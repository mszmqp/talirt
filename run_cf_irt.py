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
from numpy import exp, shape, mat
from numpy import linalg as la
# from model.matrix_factorization import *
# from evaluate.evaluate_ import *
# from model.col_fil import *
# from utils.data_process import *

sys.path.append("../")

path_data = './data/'

class CF(object):

    def __init__(self, data):
        self.stu_ans = pd.DataFrame({
                    'stu_id': data['stu_id'],
                    'lq_id': data['lq_id'],
                    'status': data['status']
                    })

    def pre_data(self):
        # print('the count of student is ' + str(len(self.stu_ans['stu_id'].unique())))
        # print('the count of item is ' + str(len(self.stu_ans['lq_id'].unique())))
        # 剔除含有答题数小于设定要求的同学答题数据
        # 保留删除的序号，用于后续保留数据的删选
        del_lis = []
        del_group = self.stu_ans.groupby('stu_id').size()
        del_num = 0
        for item in del_group:
            if item < 10:
                del_lis.append(del_num)
            del_num += 1

        dic_user = {}
        del_stu = 0
        res_stu = 0
        # 创建学生字典、保留的最后序号
        for user in self.stu_ans['stu_id'].unique():
            if del_stu not in del_lis:
                dic_user[user] = res_stu
                res_stu += 1
            del_stu += 1
        # 创建试题字典
        dic_lq = {}
        init = 0
        # 创建学生字典
        for lq in self.stu_ans['lq_id'].unique():
            dic_lq[lq] = init
            init += 1
        # 替换2为0
        # users.status = users.status.replace(2, 0)
        n_users = len(dic_user)
        n_lq_id = self.stu_ans.lq_id.unique().shape[0]
        stu_lq = np.zeros((n_users, n_lq_id))
        for row in self.stu_ans.itertuples():
            # print(row[1], row[2], row[3])
            if row[3] in dic_user.keys():
                stu_lq[dic_user[row[3]], dic_lq[row[1]]] = 1 if row[2] == 1 else -1
        sparsity = float(len(stu_lq.nonzero()[0]))
        sparsity /= (stu_lq.shape[0] * stu_lq.shape[1])
        sparsity *= 100
        # print('Sparsity: {:4.2f}%'.format(sparsity))
        return dic_user, dic_lq, stu_lq

    def fast_similarity(self, stu_ans, kind='user', epsilon=1e-9):

        # epsilon -> small number for handling dived-by-zero errors
        if kind == 'user':
            sim = stu_ans.dot(stu_ans.T) + epsilon
        elif kind == 'item':
            sim = stu_ans.T.dot(stu_ans) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    def modify(self, _pred):

        return _pred

    def predict_topk_nobias(self, ratings, similarity, kind='user', k=30):

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
                pred[i, j] = self.modify(pred[i, j])
        return pred

class IRT(object):

    def __init__(self, data):
        self.users = pd.DataFrame({
            'stu_id': data['stu_id'],
            'lq_id': data['lq_id'],
            'diff': data['diff_ori'],
            'status': data['status']
        })

    def pre_data(self):

        '''
        :param data:
        与处理之后的学生答题画像
        '''

        # print('the count of student is ' + str(len(self.users['stu_id'].unique())))
        # print('the count of item is ' + str(len(self.users['lq_id'].unique())))
        diff_group = self.users.groupby('diff').size()
        # print(diff_group)

        # 保存题目难度
        lq_id = self.users['lq_id'].unique()
        lq = []
        # 题目id
        for item in lq_id:
            lq.append(item)
        # 题目难度
        diff_lq = self.users.groupby('lq_id').max()['diff']
        diff = []
        for item in diff_lq:
            diff.append(item)
        # 题目难度字典
        lq_diff = {}
        for i in range(len(lq)):
            lq_diff[lq[i]] = diff[i]

        max_diff, min_diff = max(self.users['diff']), min(self.users['diff'])
        # print(max_diff, min_diff)

        '''
        # 剔除含有答题数小于设定要求的同学答题数据
        # 保留删除的序号，用于后续保留数据的删选
        del_lis = []
        del_group = users.groupby('stu_id').size()
        del_num = 0
        for item in del_group:
            if item < 5:
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
        '''

        # 用于存储用户试题二级字典
        stu_ans = {}

        dic_user = []
        # 创建试题字典
        dic_lq = {}
        init = 0
        for lq in self.users['lq_id'].unique():
            dic_lq[lq] = init
            init += 1

        for row in self.users.itertuples():
            if row[4] not in dic_user:
                temp = []
                # diffic
                temp.append(row[1])
                # status
                temp.append(0 if row[3] == 2 else 1)
                # 如果学生已经存在，直接append
                if row[4] in stu_ans.keys():
                    stu_ans[row[4]].append(temp)
                # 新建列表
                else:
                    stu_ans[row[4]] = []
                    stu_ans[row[4]].append(temp)
        return lq_diff, stu_ans

    def loadDataSet(self, stu_ans, stu_key):

        '''
        load the data to gradAscent
        :param stu_key:
        :return:
        '''

        data_a = []
        data_b = []
        data_ab = []
        labelMat = []
        num = 0
        for lineArr in stu_ans[stu_key]:
            # lineArr[0]为难度，lineArr[1]为区分度，lineArr[2]为对错标签。
            # Da(θ-b)
            # print(lineArr[0], lineArr[1])
            data_b.append([float(lineArr[0])])
            data_a.append([float(0.3)])
            labelMat.append(float(lineArr[1]))
        return data_a, data_b, labelMat

    def sigmoid(self, inX):
        return 1.0 / (1 + exp(-inX))

    def gradAscent(self, data_a, data_b, classLabels, alpha, maxCycles):

        '''
        calculate the gradAscent
        :param data_a:
        :param data_b:
        :param classLabels:
        :param alpha:
        :param maxCycles:
        :return:
        '''
        D = 1.7
        # 常数项
        data_a = mat(data_a)
        data_b = mat(data_b)
        # 将数组转为矩阵
        labelMat = mat(classLabels).transpose()
        m, n = shape(data_b)

        # 返回矩阵的行和列  
        weights = 1
        # 初始化最佳回归系数  
        error_pr = [0.0 for i in range(m)]
        error_pr = mat(error_pr).transpose()
        for i in range(0, maxCycles):
            h = self.sigmoid(D * data_a * weights - D * float(data_a[0][0]) * data_b)
            error = labelMat - h
            weights += D * alpha * data_a.transpose() * error
            if sum(abs(error - error_pr)) < 0.0001:
                break
            error_pr = error
        # print(weights)
        return float(weights)

    def theta_stu(self, stu_ans):

        '''
        :return: the ability of student
        '''
        alpha = 0.05
        maxCycles = 20
        weights = {}

        for stu in stu_ans.keys():
            try:
                data_a, data_b, label = self.loadDataSet(stu_ans, stu)
            except:
                continue
            weights[stu] = self.gradAscent(data_a, data_b, label, alpha, maxCycles)
        return weights

class RecommendABC(object):

    def __init__(self, year, city_id, grade_id, subject_id, level_id, term):
        self.year = year
        self.city_id = city_id
        self.grade_id = grade_id
        self.subject_id = subject_id
        self.level_id = level_id
        self.term = term

    def fetch_data(self):
        """
        :param from_cache:
        :return: df_question
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
    def load_from_redis(self, model):
        # return self object
        pass

    @classmethod
    def load_from_file(self, model):
        # return self object
        pass

    def predict(self, stu_id, items=None):
        """
        Parameters
        ----------
        stu_id
        items : 可以为空，已经作答过的题目不用返回
        Returns
        -------
        """
        raise NotImplemented

class RecommendCF(RecommendABC):

    pred_file = 'pred_cf.pickle'
    pred_file = path_data + pred_file

    def fetch_data(self, from_cache=False):
        param = {}
        param['year'] = self.year
        param['city_id'] = self.city_id
        param['grade_id'] = self.grade_id
        param['subject_id'] = self.subject_id
        param['level_id'] = self.level_id
        param['term'] = self.term

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
                                   sa.sa_year = '%(year)s' 
                                   and sa.sa_subj_id ='%(subject_id)s' -- 限定学科
                                   and sa.sa_grd_id = '%(grade_id)s' -- 限定年级
                                   and sa.sa_lev_id = '%(level_id)s'  -- 限定班型
                                   and sa.sa_city_code = '%(city_id)s' -- 限定城市
                                   and sa.sa_term_id = '%(term)s'
                                   and sa.sa_create_time is not null
                               ) s
                           ) sa
                           order by sa.sa_stu_id,
                           sa.sa_create_time
                     """ % param

        if from_cache:
            # print >> sys.stderr, "从缓存读取题目画像数据"
            # print("从缓存读取题目画像数据", file=sys.stderr)
            return pd.read_pickle(cache_file)
        # print("从impala读取题目画像数据", file=sys.stderr)
        ibis.options.sql.default_limit = None
        impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')
        df_question = impala_client.sql(sql1).execute()
        df_question.to_pickle(cache_file)
        impala_client.close()
        # print("count:", len(df_question), file=sys.stderr)
        return df_question

    def fit(self, df_question):

        cf = CF(df_question)
        dic_user, dic_lq, stu_ans = cf.pre_data()
        # print(stu_ans[0:5, ])
        stu_simi = cf.fast_similarity(stu_ans)
        # print(stu_simi[0:5, 0:5])
        pred = cf.predict_topk_nobias(stu_ans, stu_simi)
        # print(pred[0:5, ])
        return dic_user, dic_lq, pred

    def seri(self, dic_user, dic_lq, pred):

        items = dic_user.items()
        back_items = [[v[1], v[0]] for v in items]
        back_items.sort()
        back_user = [back_items[i][1] for i in range(0, len(back_items))]

        items = dic_lq.items()
        back_items = [[v[1], v[0]] for v in items]
        back_items.sort()
        back_lq = [back_items[i][1] for i in range(0, len(back_items))]
        # print(pred.shape[0], pred.shape[1])
        # print(len(back_user), len(back_lq))
        pred = DataFrame(pred, index=back_user, columns=back_lq)
        pred.to_pickle(RecommendCF.pred_file)
        return pred

    def predict(self, stu_id, items=None, pickle_file=False):

        if pickle_file:
            pred = pd.read_pickle(RecommendCF.pred_file)
        else:
            pass
        return pred.loc[stu_id, items]

class RecommendIRT(RecommendABC):

    '''
    # 示例程序为从csv读取文件
    '''

    pred_file = 'theta_irt.pickle'
    pred_file = path_data + pred_file

    def fetch_data(self, from_cache=False):
        param = {}
        param['year'] = self.year
        param['city_id'] = self.city_id
        param['grade_id'] = self.grade_id
        param['subject_id'] = self.subject_id
        param['level_id'] = self.level_id
        param['term'] = self.term

        cache_file = "hz_stu_top.pickle"
        cache_file = path_data + cache_file

        # sql可以更改已满足irt所需参数
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
                                   sa.sa_year = '%(year)s' 
                                   and sa.sa_subj_id ='%(subject_id)s' -- 限定学科
                                   and sa.sa_grd_id = '%(grade_id)s' -- 限定年级
                                   and sa.sa_lev_id = '%(level_id)s'  -- 限定班型
                                   and sa.sa_city_code = '%(city_id)s' -- 限定城市
                                   and sa.sa_term_id = '%(term)s'
                                   and sa.sa_create_time is not null
                               ) s
                           ) sa
                           order by sa.sa_stu_id,
                           sa.sa_create_time
                     """ % param

        if from_cache:
            # print >> sys.stderr, "从缓存读取题目画像数据"
            # print("从缓存读取题目画像数据", file=sys.stderr)
            return pd.read_pickle(cache_file)
        # print("从impala读取题目画像数据", file=sys.stderr)
        ibis.options.sql.default_limit = None
        impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')
        df_question = impala_client.sql(sql1).execute()
        df_question.to_pickle(cache_file)
        impala_client.close()
        # print("count:", len(df_question), file=sys.stderr)
        return df_question

    def obtain_from_csv(self):
        '''
        read data from csv
        '''
        dir_input = 'hive_stu_ans.csv'
        dir_input = path_data + dir_input
        # 可根据需要进行修改
        names = ['stu_id', 'lq_id', 'diff_aft', 'discrimination', 'time', 'cl_id', 'status', 'diff_ori']
        data = pd.read_csv(dir_input, sep=',', names=names)
        return data

    def fit(self, df_question):

        irt = IRT(df_question)
        lq_diff, stu_ans = irt.pre_data()
        weights = irt.theta_stu(stu_ans)
        return lq_diff, weights

    def seri(self, weights):

        items = weights.items()
        back_items = [[v[0], v[1]] for v in items]
        back_user = [back_items[i][0] for i in range(0, len(back_items))]
        theta_stu = [back_items[i][1] for i in range(0, len(back_items))]
        # print(pred.shape[0], pred.shape[1])
        # print(len(back_user), len(back_lq))
        columns = ['theta']
        theta_stu = DataFrame(theta_stu, index=back_user, columns=columns)
        theta_stu.to_pickle(RecommendIRT.pred_file)
        return theta_stu

    def predict(self, stu_id, lq_diff, items=None, pickle_file=False):

        if pickle_file:
            theta_stu = pd.read_pickle(RecommendIRT.pred_file)
        else:
            theta_stu = self.seri()
        theta = theta_stu.loc[stu_id, 'theta']
        inX = 1.7 * 0.3 * (theta - float(lq_diff[items]))
        return 1.0 / (1 + exp(-inX))

if __name__ == '__main__':

    # test cf_algorithm
    re_cf = RecommendCF(year='2018', city_id='0571', grade_id='7', \
                 subject_id='ff80808127d77caa0127d7e10f1c00c4', \
                 level_id='ff8080812fc298b5012fd3d3becb1248', \
                 term='1')

    stu = 'ff80808146248e430146271765bb0baa'
    lq = '8a53ce07d5ff409bbe276a354708f677'
    df_question = re_cf.fetch_data()
    dic_user, dic_lq, pred = re_cf.fit(df_question)
    pred = re_cf.seri(dic_user, dic_lq, pred)
    print("CF_ALGORITHM: the score of student: " + stu + " at the item: " + lq + " is " + str(re_cf.predict(stu, lq, True)))

    # test irt_algorithm
    re_irt = RecommendIRT(year='2018', city_id='0571', grade_id='7', \
                        subject_id='ff80808127d77caa0127d7e10f1c00c4', \
                        level_id='ff8080812fc298b5012fd3d3becb1248', \
                        term='1')
    stu = '009b1e101aa54843ba61a188941de4b6'
    lq = '03e61a4dc33b451294c2e3d79f4ec468'
    df_question = re_irt.obtain_from_csv()
    lq_diff, weights = re_irt.fit(df_question)
    theta_stu = re_irt.seri(weights)
    print("IRT_ALGORITHM: the probability of student: " + stu + " at the item: " + lq + " is " + str(re_irt.predict(stu, lq_diff, lq, True)))
