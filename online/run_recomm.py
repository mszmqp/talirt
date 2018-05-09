# -*-coding:utf-8-*-

'''
Created on 2018年1月27日
@author:mengshuai@100tal.com
run cf_algorithm
'''

import os
import time
# import csv
# import math
# import copy
import sys
import json
# import matplotlib.pyplot as plt
import numpy as np
# import ibis
import pandas as pd
import random
# from math import sqrt
# from pandas import DataFrame
# from numpy import exp, shape, mat
# from numpy import linalg as la
# import shutil
# from model.matrix_factorization import *
# from evaluate.evaluate_ import *
# from model.col_fil import *
# from utils.data_process import *
from scipy.special import expit as sigmod
from scipy.optimize import minimize
# from tqdm import tqdm
import tempfile
import abc
import argparse

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("recommend")
logger_ch = logging.StreamHandler(stream=sys.stderr)
logger.addHandler(logger_ch)
_sim_threshold = 0.0


def log(*args):
    print(' '.join(args), file=sys.stderr)


class DiskDB:

    def __init__(self, path='./cache_learn'):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save_json(self, table, key, value):
        path = os.path.join(self.path, table)
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, key)
        with open(file_name, 'w') as fh:
            json.dump(value, fh)

    def save_bin(self, table, key, value):
        path = os.path.join(self.path, table)
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, key)
        with open(file_name, 'wb') as fh:
            fh.write(value)

    def load_json(self, table, key):
        file_name = os.path.join(self.path, table, key)
        with open(file_name, 'r') as fh:
            return json.load(fh)

    def load_bin(self, table, key):
        file_name = os.path.join(self.path, table, key)
        with open(file_name, 'rb') as fh:
            return fh.read()


class SimpleCF:
    default_value = np.nan

    def fit(self, response: pd.DataFrame, sequential=True):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        assert response is not None
        self.response_matrix = None

        if sequential:
            self.response_matrix = response.pivot(index="user_id", columns="item_id", values='answer')

        else:
            self.response_matrix = response
            self.response_matrix.index.name = 'user_id'

            # 答错用-1表示，答对用1表述，未作答用0
        self.response_matrix[self.response_matrix == 0] = -1
        self.response_matrix.fillna(0, inplace=True)
        # 矩阵中应该不包含全是0的行，否则不能求模
        return True

    def predict(self, stu_id, items):
        """

        Parameters
        ----------
        stu_id 目标学生id
        items 候选题目id集合

        Returns
        -------

        """

        assert self.response_matrix is not None
        if isinstance(items, pd.DataFrame):
            items = items.index
        elif isinstance(items, pd.Series) or isinstance(items, list):
            pass
        else:
            raise ValueError('items 类型错误')
        # 矩阵中没有目标学生的记录
        if not self.response_matrix.index.contains(stu_id):
            log('CF', 'stu_not_in_matrix')
            prob = pd.Series(data=[self.default_value] * len(items), index=items)
            return prob, None

        # 目标学生的向量
        stu_vector_df = self.response_matrix.loc[stu_id, :]

        # 候选题目集合没出现在矩阵中
        if self.response_matrix.columns.intersection(items).empty:
            log('CF', 'items_not_in_matrix')
            # 返回的预测概率都是0.5
            prob = pd.Series(data=[self.default_value] * len(items), index=items)
            return prob, stu_vector_df

        stu_vector = stu_vector_df.values.flatten()
        stu_vector = stu_vector.reshape(len(stu_vector), 1)
        # 目标学生在矩阵中的位置
        stu_iloc = self.response_matrix.index.get_loc(stu_id)

        # 全部学生的向量矩阵
        all_vector = self.response_matrix.values

        # cosine 相似度
        sim_score = np.dot(all_vector, stu_vector) / np.linalg.norm(all_vector, ord=2, axis=1).reshape(len(all_vector),
                                                                                                       1) / np.linalg.norm(
            stu_vector, ord=2)

        # 相似度>0.8以上的学生记录,只保留候选题目集合
        # todo 不按相似度过滤可以扩大召回，反正后面有把相似度作为权重相乘
        global _sim_threshold
        selected = sim_score.flatten() >= _sim_threshold
        # 从相似学生list中去掉目标学生自己
        selected[stu_iloc] = False

        if not any(selected):
            # 没有与其相似的用户
            log('CF', 'stu_no_sim_stu')
            # 返回的预测概率都是0.5
            prob = pd.Series(data=[self.default_value] * len(items), index=items)
            return prob, stu_vector_df

        sim_vector = self.response_matrix.loc[selected, items]

        # 相似度作为权重，求其相似学生答题结果的平均值
        weights = sim_score[selected]
        # prob 是pandas的series对象
        prob = sim_vector.mul(weights.flatten(), axis=0).mean(axis=0, skipna=True)
        # 这个值的区间是[-1,1],需要转换成[0,1]，转换方法为 x-min/interval,
        prob = (prob + 1) / 2

        # prob 是pandas的series对象,其index是题目id，value是作答概率，
        # 其中有空值存在,空值就是没有预测出来结果，我们设置为0.5的概率值
        if self.default_value is not None:
            prob.fillna(self.default_value, inplace=True)
        return prob, stu_vector_df

    def to_pickle(self):
        fh = tempfile.TemporaryFile(mode='w+b')
        self.response_matrix.to_pickle(path=fh)
        fh.seek(0)
        data = fh.read()
        fh.close()
        return data

    @classmethod
    def from_pickle(cls, data):
        fh = tempfile.TemporaryFile(mode='w+b')
        fh.write(data)
        fh.seek(0)
        response_matrix = pd.read_pickle(fh)
        fh.close()
        obj = cls()
        # cf构造函数 需要把0-1作答结果转成-1，1的形式，
        # 这里不能用构造函数传入数据
        obj.response_matrix = response_matrix
        return obj

    def mse(self, user=None):
        from sklearn.metrics import mean_squared_error
        if user is None:
            user_list = list(self.response_matrix.index)
        elif isinstance(user, str):
            user_list = [user]
        else:
            user_list = user

        item_list = list(self.response_matrix.columns)
        error_list = []

        for user_id in user_list:
            prob, vector = self.predict(user_id, item_list)
            answered = vector != 0
            vector = vector[answered]
            prob = prob[answered]
            vector[vector == -1] = 0
            error = mean_squared_error(prob, vector)
            error_list.append(error)
        return pd.Series(error_list, index=user_list, name='mse')


class UIrt2PL:
    def __init__(self, D=1.702):
        self.D = D
        self.k = 1
        self.user_vector = None
        self.item_vector = None

    def fit(self, response: pd.DataFrame, sequential=True):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        assert response is not None

        if sequential:
            self.response_sequence = response[['user_id', 'item_id', 'answer', 'a', 'b']]
            self.response_matrix = self.response_sequence.pivot(index="user_id", columns="item_id", values='answer')

        else:
            self.response_matrix = response.copy()
            self.response_matrix.index.name = 'user_id'
            # 矩阵形式生成序列数据
            self.response_sequence = pd.melt(self.response_matrix.reset_index(), id_vars=['user_id'],
                                             var_name="item_id",
                                             value_name='answer')
            # 去掉空值
            self.response_sequence.dropna(inplace=True)

        # 
        self._init_model()
        labels = set(response.columns).intersection(set(['a', 'b', 'c']))
        if sequential and labels:
            item_info = response[['item_id'] + list(labels)].drop_duplicates(subset=['item_id'])
            item_info.set_index('item_id', inplace=True)
            self.set_abc(item_info, columns=list(labels))

        return self.estimate_theta()

    def _init_model(self):
        assert self.response_sequence is not None
        user_ids = list(self.response_matrix.index)
        user_count = len(user_ids)
        item_ids = list(self.response_matrix.columns)
        item_count = len(item_ids)
        self.user_vector = pd.DataFrame({
            'iloc': np.arange(user_count),
            'user_id': user_ids,
            'theta': np.zeros(user_count)},
            index=user_ids)
        self.item_vector = pd.DataFrame(
            {'iloc': np.arange(item_count),
             'item_id': item_ids,
             'a': np.ones(item_count),
             'b': np.zeros(item_count),
             'c': np.zeros(item_count)}, index=item_ids)

        self.response_sequence = self.response_sequence.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id',
                                                             how='left')
        self.response_sequence = self.response_sequence.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id',
                                                             how='left')
        # 统计每个应试者的作答情况
        # user_stat = self.response_sequence.groupby('user_id')['answer'].aggregate(['count', 'sum']).rename(
        #     columns={'sum': 'right'})

        x = self.response_sequence.groupby(['user_id', 'b']).aggregate({'answer': ['count', 'sum']})
        y = x.unstack()
        y.columns = ['_'.join([str(x) for x in col[1:]]).strip().replace('sum', 'right') for col in y.columns.values]

        for i in range(1, 6):
            if 'right_%d' % i in y.columns:
                y['accuracy_%d' % i] = y['right_%d' % i] / y['count_%d' % i]

        y['count_all'] = y.filter(regex='^count_', axis=1).sum(axis=1)
        y['right_all'] = y.filter(regex='^right_', axis=1).sum(axis=1)
        y['accuracy_all'] = y['right_all'] / y['count_all']

        self.user_vector = self.user_vector.join(y, how='left')
        # self.user_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        # self.user_vector['accuracy'] = self.user_vector['right'] / self.user_vector['count']

        # 统计每个项目的作答情况
        item_stat = self.response_sequence.groupby('item_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})
        self.item_vector = self.item_vector.join(item_stat, how='left')
        self.item_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.item_vector['accuracy'] = self.item_vector['right'] / self.item_vector['count']

    def set_theta(self, values):
        """

        Parameters
        ----------
        values

        Returns
        -------

        """
        assert isinstance(values, pd.DataFrame) or isinstance(values,
                                                              np.ndarray), "values的类型必须是pandas.DataFrame或numpy.ndarray"

        if self.user_vector is None:
            assert isinstance(values, pd.DataFrame), "values的类型必须是pandas.DataFrame"
            user_count = len(values)
            user_ids = list(values.index)

            self.user_vector = pd.DataFrame({
                'iloc': np.arange(user_count),
                'user_id': user_ids,
                'theta': values.loc[:, 'theta'].values.flatten(),
            },
                index=user_ids)

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.user_vector.loc[values.index, 'theta'] = values.loc[:, 'theta'].values.flatten()

            elif isinstance(values, np.ndarray):
                self.user_vector.loc[:, 'theta'] = values.flatten()

            else:
                raise TypeError("values的类型必须是pandas.DataFrame 或numpy.ndarray")

    def set_abc(self, values, columns=None):
        """
        values 可以是pandas.DataFrame 或者 numpy.ndarray
        当values:pandas.DataFrame,,shape=(n,len(columns))，一行一个item,
        pandas.DataFrame.index是item_id,columns包括a,b,c。

        当values:numpy.ndarray,shape=(n,len(columns)),一行一个item,列对应着columns参数。
        Parameters
        ----------
        values
        columns 要设置的列

        Returns
        -------

        """

        assert isinstance(values, pd.DataFrame) or isinstance(values,
                                                              np.ndarray), "values的类型必须是pandas.DataFrame或numpy.ndarray"
        if columns is None:
            if isinstance(values, pd.DataFrame):
                columns = [x for x in ['a', 'b', 'c'] if x in values.columns]
            else:
                raise ValueError("需要指定columns")

        if self.item_vector is None:
            assert isinstance(values, pd.DataFrame), "values的类型必须是pandas.DataFrame"
            item_count = len(values)
            item_ids = list(values.index)

            self.item_vector = pd.DataFrame({
                'iloc': np.arange(item_count),
                'item_id': item_ids,
                'a': np.ones(item_count),
                'b': np.zeros(item_count),
                'c': np.zeros(item_count),

            },
                index=item_ids)

            self.item_vector.loc[:, columns] = values.loc[:, columns].values

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.item_vector.loc[values.index, columns] = values.loc[:, columns].values

            elif isinstance(values, np.ndarray):
                self.item_vector.loc[:, columns] = values

            else:
                raise TypeError("values的类型必须是pandas.DataFrame或numpy.ndarray")

    def set_items(self, items: pd.DataFrame):
        self.item_vector = items

    def set_users(self, users: pd.DataFrame):
        self.user_vector = users

    def predict_s(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"

        user_v = self.user_vector.loc[users, ['theta']]

        if isinstance(items, pd.DataFrame) and set(items.columns).intersection(set(['a', 'b'])):
            item_v = items.loc[:, ['a', 'b']]
        else:
            item_v = self.item_vector.loc[items, ['a', 'b']]

        z = item_v['a'].values * (user_v['theta'].values - item_v['b'].values)
        # z = alpha * (theta - beta)
        e = np.exp(z)
        s = e / (1.0 + e)
        return s

    def predict_x(self, users, items):
        if isinstance(items, pd.DataFrame):
            self.set_items(items)
        if isinstance(users, pd.DataFrame):
            self.set_theta(users)

        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, 'theta'].values.reshape((user_count, 1))
        a = self.item_vector.loc[items, 'a'].values.reshape((1, item_count))
        b = self.item_vector.loc[items, 'b'].values.reshape((1, item_count))
        # c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))
        # c = c.repeat(user_count, axis=0)
        z = a.repeat(user_count, axis=0) * (
                theta.repeat(item_count, axis=1) - b.repeat(user_count, axis=0))
        prob_matrix = sigmod(z)
        # e = np.exp(z)
        # s =   e / (1.0 + e)
        return prob_matrix

    def predict_simple(self, stu_id, items: pd.DataFrame):
        """

        Parameters
        ----------
        theta
        items

        Returns
        -------

        """
        theta = self.user_vector.loc[stu_id, 'theta']
        b = items.loc[:, ['b']].values
        z = self.D * (theta - b)
        prob = sigmod(z)
        # items['irt'] = prob
        return pd.Series(data=prob.flatten(), index=items.index), theta

    def _prob(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray = None):
        """

        Parameters
        ----------
        theta shape=(n,1) n是学生数量
        a  shape=(1,m) m是题目数量
        b  shape=(1,m) m是题目数量
        c  shape=(1,m) m是题目数量

        Returns
        -------

        """

        z = self.D * a * (theta.reshape(len(theta), 1) - b)
        # print(type(z))
        if c is None:
            return sigmod(z)
        return c + (1 - c) * sigmod(z)

    def _object_func(self, theta: np.ndarray, y: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                     c: np.ndarray = None):
        """
        .. math::
            Object function  = - \ln L(x;\theta)=-(\sum_{i=0}^n ({y^{(i)}} \ln P + (1-y^{(i)}) \ln (1-P)))
        Parameters
        ----------
        theta
        a
        b
        c

        Returns
        -------
        res : OptimizeResult

        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer
        exited successfully and message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
        """

        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b)
        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        obj = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        # 用where处理不了空值，如果是空值，where认为是真
        # obj = - np.sum(np.where(y, np.log(y_hat), np.log(1 - y_hat)))
        # print('obj', obj)
        # 目标函数没有求平均
        return - np.sum(np.nan_to_num(obj, copy=False))

    def _jac_theta(self, theta: np.ndarray, y: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                   c: np.ndarray = None):
        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b)
        # 一阶导数
        # 每一列是一个样本，求所有样本的平均值
        all = self.D * a * (y_hat - y)

        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        grd = np.sum(np.nan_to_num(all, copy=False), axis=1)
        # grd = grd.reshape(len(grd), 1)
        # print(grd.shape, file=sys.stderr)
        return grd

    def estimate_theta(self, tol=None, options=None, bounds=None):
        """
        已知题目参数的情况下，估计学生的能力值。
        优化算法说明参考 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        Parameters
        ----------
        method 优化算法，可选 CG、Newton-CG、L-BFGS-B
        tol
        options
        bounds
        join join=True，所有学生一起估计；反之，每个学生独立估计

        Returns
        -------

        """
        item_count = len(self.item_vector)
        if 'a' in self.item_vector.columns:
            a = self.item_vector.loc[:, 'a'].values.reshape(1, item_count)
        else:
            a = None

        b = self.item_vector.loc[:, 'b'].values.reshape(1, item_count)
        # if 'c' in self.item_vector.columns:
        #     c = self.item_vector.loc[:, 'c'].values.reshape(1, item_count)
        # else:
        #     c = None

        success = []

        # self._es_res_theta = []

        # 每个人独立估计
        for index, row in self.response_matrix.iterrows():
            # 注意y可能有缺失值
            yy = row.dropna()
            # len(y) == len(y.dropna())
            # 全对的情况
            if yy.sum() == len(yy):
                theta = self.response_sequence.loc[self.response_sequence['user_id'] == index, 'b'].max() + 0.5
                success.append(True)
                # self._es_res_theta.append(res)
            else:
                y = row.values.reshape(1, len(row))
                theta = self.user_vector.loc[index, 'theta']

                res = minimize(self._object_func, x0=[theta], args=(y, a, b), jac=self._jac_theta,
                               bounds=bounds, options=options, tol=tol)
                theta = res.x[0]
                success.append(res.success)

                # self._es_res_theta.append(res)
            # 全错估计值会小于0
            theta = 0 if theta < 0 else theta

            self.user_vector.loc[index, 'theta'] = theta

        return all(success)

    def to_dict(self):
        return self.user_vector['theta'].to_dict()

    @classmethod
    def from_dict(cls, serialize_data):
        obj = cls()
        index = []
        theta = []
        for key, value in serialize_data.items():
            index.append(key)
            theta.append(value)
        obj.set_theta(pd.DataFrame({'theta': theta}, index=index))
        return obj

    def to_pickle(self):
        fh = tempfile.TemporaryFile(mode='w+b')
        self.user_vector.to_pickle(path=fh)
        fh.seek(0)
        data = fh.read()
        fh.close()
        return data

    @classmethod
    def from_pickle(cls, data):
        fh = tempfile.TemporaryFile(mode='w+b')
        fh.write(data)
        fh.seek(0)
        user_vector = pd.read_pickle(fh)
        fh.close()
        obj = cls()
        # cf构造函数 需要把0-1作答结果转成-1，1的形式，
        # 这里不能用构造函数传入数据
        obj.user_vector = user_vector
        return obj


_candidate_items = None
_stu_response_items = None
_level_response = None


def load_candidate_items(**kwargs):
    """
    获取候选题目信息
    Returns
    -------

    """
    global _candidate_items
    if _candidate_items is not None:
        return _candidate_items

    # _candidate_items = pd.DataFrame({'item_id': np.arange(5),
    #                                  'a': [1] * 5,
    #                                  'b': np.arange(start=1, stop=6, step=1),
    #                                  }).set_index('item_id')  # 题目区分度1，难度1

    import ibis

    ibis.options.sql.default_limit = None
    impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')

    _sql = """
                select
                    que_id as item_id,
                    difficulty as b,
                    1 as a
                from dmdb.dm_qstportrait_qst
                where
                    parent_id='0'
                    and	subject_id=2
                    and	grade_group_id=2
                    and qt_id in ('1','2','3','4','5')
                    and state=0
                    and kh_ids like '%%%(knowledge_id)s%%'
                    -- and subject_name = ''

        """ % kwargs

    _candidate_items = impala_client.sql(_sql).execute().set_index('item_id').iloc[:100, :]

    impala_client.close()
    return _candidate_items


def load_level_response(**kwargs):
    """
    拉取当前班型下的所有答题记录
    Parameters
    ----------
    stu_id

    Returns
    -------

    """
    global _level_response
    if _level_response is not None:
        return _level_response

    import ibis

    ibis.options.sql.default_limit = None
    impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')

    _sql = """
                select
                    sa.c_sortorder,
                    fk_student as stu_id,
                    stu_name as user_id,
                    fk_question as item_id,
                    difficulty_id as b,
                    1 as a,
                    case when asw_first_status_id =1 then 1 else 0  end   answer
                from dwdb.dwd_stdy_ips_level_answ sa
                join dimdb.dim_grade on sa.fk_grade=dim_grade.pk_grade
                join dimdb.dim_term on sa.fk_term=dim_term.pk_term
                join dimdb.dim_classlevel on sa.fk_classlevel=dim_classlevel.pk_classlevel
                join dimdb.dim_subject on sa.fk_subject=dim_subject.pk_subject

                where
                        fk_year='%(year)s'
                        and fk_city = '%(city_id)s'
                        and dim_grade.grd_id='%(grade_id)s'
                        and dim_term.term_id='%(term_id)s'
                        and dim_subject.old_subj_id='%(subject_id)s'
                        and sa.is_deleted='否'
                        and dim_classlevel.old_lev_id='%(level_id)s'
                        


        """ % kwargs
    # print(_sql, file=sys.stderr)
    _level_response = impala_client.sql(_sql).execute()
    impala_client.close()
    return _level_response


def load_stu_response(stu_id, level_response=None):
    """
    获取当前学生的答题记录
    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    global _stu_response_items, _level_response
    # if _stu_response_items is not None:
    #     return _stu_response_items
    # stu_id = kwargs['stu_id']
    if level_response is None:
        level_response = _level_response
    _stu_response_items = level_response.loc[level_response['user_id'] == stu_id, ['item_id', 'b', 'answer']]
    return _stu_response_items.drop_duplicates(subset=['item_id']).set_index('item_id')


class Recommend(object):
    model_irt = None
    model_cf = None
    probs = {}

    def __init__(self, db, param):
        self.db = db
        self.param = param
        self.year = param['year']
        self.city_id = param['city_id']
        self.grade_id = param['grade_id']
        self.subject_id = param['subject_id']
        self.level_id = param['level_id']
        self.term_id = param['term_id']
        self.knowledge_id = param['knowledge_id']

    def train_model(self, response: pd.DataFrame, sequential=True):
        self.model_irt = UIrt2PL()
        self.model_cf = SimpleCF()
        return all([self.model_cf.fit(response=response, sequential=sequential),
                    self.model_irt.fit(response=response, sequential=sequential)])

    def load_model(self):
        key = '_'.join(
            [str(self.year),
             str(self.city_id),
             str(self.grade_id),
             str(self.subject_id),
             str(self.term_id),
             str(self.level_id),
             ])

        # self.model_irt = UIrt2PL.from_dict(self.db.load_json('irt', key=key))
        self.model_irt = UIrt2PL.from_pickle(self.db.load_bin('irt', key=key))
        self.model_cf = SimpleCF.from_pickle(self.db.load_bin('cf', key=key))
        # return True

    def save_model(self):
        key = '_'.join(
            [str(self.year),
             str(self.city_id),
             str(self.grade_id),
             str(self.subject_id),
             str(self.term_id),
             str(self.level_id),
             ])

        # self.db.save_json('irt', key, self.model_irt.to_dict())
        self.db.save_bin('irt', key, self.model_irt.to_pickle())
        self.db.save_bin('cf', key, self.model_cf.to_pickle())

    def select(self, stu_cur_acc, candidate_items):

        if stu_cur_acc > 0.95:
            result = candidate_items[candidate_items['prob'] < 0.5].sort_values('prob', ascending=False)
        elif stu_cur_acc < 0.6:
            result = candidate_items.sort_values('prob', ascending=False)
        else:
            result = candidate_items[(candidate_items['prob'] >= 0.5) & (candidate_items['prob'] <= 0.9)].sort_values(
                'prob',
                ascending=False)
        if len(result) == 0:
            result = candidate_items.sort_values('prob', ascending=False)

        return result

    def get_rec(self, stu_id: str, stu_acc, candidate_items: pd.DataFrame):
        """

        Parameters
        ----------
        stu_id
        candidate_items 要求item_id为index

        Returns
        -------

        """
        prob_irt, stu_theta = self.model_irt.predict_simple(stu_id, candidate_items)
        prob_cf, stu_vector = self.model_cf.predict(stu_id, candidate_items)
        candidate_items['irt'] = prob_irt
        candidate_items['cf'] = prob_cf
        merge_prob = []

        self.probs['irt'] = prob_irt
        self.probs['cf'] = prob_cf
        result = []
        prob_irt.fillna(-1, inplace=True)
        prob_cf.fillna(-1, inplace=True)
        for item1, item2 in zip(prob_cf.iteritems(), prob_irt.iteritems()):
            index_cf, value_cf = item1
            index_irt, value_irt = item2
            value_cf = float(value_cf)
            value_irt = float(value_irt)
            weight_irt = 0.5
            weight_cf = 0.5
            assert index_irt == index_cf
            # 两个数据都是空
            if value_irt == -1 and value_cf == -1:
                merge_prob.append(np.nan)
                continue
            if value_irt == -1:
                weight_cf = 1
                value_irt = 0
            if value_cf == -1:
                weight_irt = 1
                value_cf = 0

            value = weight_irt * value_irt + weight_cf * value_cf
            merge_prob.append(value)

            # result.append((index_cf, value))
            # print(index_cf, value)
        candidate_items['prob'] = merge_prob
        result = self.select(stu_acc, candidate_items)
        return result


def online(param):
    # global _candidate_items, _stu_response_items, _level_response
    # _level_response = pd.read_pickle('level_response.bin')
    # candidate_items = load_candidate_items(**param)
    # candidate_items.to_pickle('candidate_items.bin')
    # candidate_items = pd.read_pickle('candidate_items.bin')
    # stu_response = load_stu_response(param['stu_id'])
    # stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)
    # 从候选集合中剔除已作答过的题目
    # candidate_items.drop(stu_response.index, inplace=True, errors='ignore')

    candidate_items = pd.DataFrame(param['candidate_items'])
    candidate_items['a'] = 1
    stu_response = pd.DataFrame(param['stu_response'])
    stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)

    rec_obj = Recommend(db=DiskDB(), param=param)
    rec_obj.load_model()
    result = rec_obj.get_rec(stu_id=param['stu_id'], stu_acc=stu_acc, candidate_items=candidate_items)
    print(result.to_json())


def metric(rec_obj, train_data, test_data):
    from sklearn import metrics

    y_prob = rec_obj.model_irt.predict_s(test_data.loc[:, 'user_id'], test_data.loc[:, ['a', 'b']])

    selected = np.isfinite(y_prob)
    y_prob = y_prob[selected]
    y_true = test_data.loc[:, 'answer'][selected]

    print("irt", 'mse', metrics.mean_squared_error(y_true, y_prob), file=sys.stderr)
    threshold = 0.5
    y_pred = y_prob.copy()
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0

    print("irt", 'acc', metrics.accuracy_score(y_true, y_pred), file=sys.stderr)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    print("irt", 'auc', metrics.auc(fpr, tpr), file=sys.stderr)


def main(options):
    # param = {'year': '2018',
    #          'city_id': '0571',
    #          'grade_id': '7',
    #          'subject_id': 'ff80808127d77caa0127d7e10f1c00c4',
    #          'level_id': 'ff8080812fc298b5012fd3d3becb1248',
    #          'term_id': '1',
    #          'knowledge_id': "cb1471bd830c49c2b5ff8b833e3057bd",
    #          'stu_id': '殷烨嵘',
    #           'stu_response':{'user_id':[],'item_id':[],'answer':[],'b':[]},
    #           'candidate_items':{'item_id':[],'b':[]},
    #          }

    # pd.DataFrame.from_records([(3,'a'),(4,'h')])
    # pd.DataFrame.from_records([{'id':3,'xx':'a'},{'id':4,'xx':'h'}])
    # pd.DataFrame({'a':[1,4],'b':[3,6]})
    if options.log == 'info':
        logger_ch.setLevel(logging.INFO)
    elif options.log == 'warning':
        logger_ch.setLevel(logging.WARNING)
    elif options.log == 'debug':
        logger_ch.setLevel(logging.DEBUG)
    elif options.log == 'error':
        logger_ch.setLevel(logging.ERROR)

    if options.run == 'online':
        run_func = online
    elif options.run == 'test_one':
        run_func = test_one
    elif options.run == 'test_level':
        run_func = test_level

    for line in options.input:
        param = json.loads(line)
        log_msg_prefix = "%(city_id)s %(subject_id)s %(grade_id)s %(level_id)s %(stu_id)s %(knowledge_id)s" % param
        _format = '%(asctime)s - %(levelname)s -' + log_msg_prefix + ' %(message)s '
        formatter = logging.Formatter(fmt=_format, datefmt=None)
        logger_ch.setFormatter(formatter)
        run_func(param)


def test_one(**param):
    # 这两份数据是所有策略都要用的，所以单独进行
    global _candidate_items, _stu_response_items, _level_response

    load_level_response(**param)
    # _level_response.to_pickle('level_response.bin')
    _level_response = pd.read_pickle('level_response.bin')

    # candidate_items = pd.DataFrame(param['candidate_items'])
    # candidate_items['a'] = 1
    # stu_response = pd.DataFrame(param['stu_response'])
    # stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)

    train_data = _level_response.loc[_level_response['c_sortorder'] < 6, :]
    test_data = _level_response.loc[_level_response['c_sortorder'] >= 6, :]

    stu_response = load_stu_response(param['stu_id'], train_data)
    stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)
    candidate_items = load_candidate_items(**param)
    # 从候选集合中剔除已作答过的题目
    candidate_items.drop(stu_response.index, inplace=True, errors='ignore')

    rec_obj = Recommend(db=DiskDB(), param=param)
    # print('-' * 10, 'train', '-' * 10, file=sys.stderr)

    ok = rec_obj.train_model(train_data)
    print('train_model', ok, file=sys.stderr)

    # print(rec_obj.model_irt.user_vector.loc[param['stu_id'], :], file=sys.stderr)

    print('-' * 10, 'save', '-' * 10, file=sys.stderr)

    rec_obj.save_model()

    print('-' * 10, 'load', '-' * 10, file=sys.stderr)

    rec_obj.load_model()
    print(rec_obj.model_irt.user_vector.loc[param['stu_id'], :], file=sys.stderr)

    # print('-' * 10, 'predict', '-' * 10, file=sys.stderr)
    print('-' * 10, 'recommend', '-' * 10, file=sys.stderr)

    result = rec_obj.get_rec(param['stu_id'], stu_acc=stu_acc, candidate_items=candidate_items)

    # print(candidate_items.sort_values('prob'), file=sys.stderr)
    print(result, file=sys.stderr)

    # print(json.dumps(result))
    print('-' * 10, 'metric', '-' * 10, file=sys.stderr)

    metric(rec_obj, train_data=train_data, test_data=test_data)
    return


def test_level(**param):
    # 这两份数据是所有策略都要用的，所以单独进行
    global _candidate_items, _stu_response_items, _level_response

    # load_level_response(**param)
    # _level_response.to_pickle('level_response.bin')
    _level_response = pd.read_pickle('level_response.bin')

    candidate_items = load_candidate_items(**param)

    train_data = _level_response.loc[_level_response['c_sortorder'] < 6, :]
    test_data = _level_response.loc[_level_response['c_sortorder'] >= 6, :]

    rec_obj = Recommend(db=DiskDB(), param=param)

    ok = rec_obj.train_model(train_data)

    metric(rec_obj, train_data=train_data, test_data=test_data)

    rec_difficulty = []
    for stu_id in list(rec_obj.model_irt.user_vector.index):
        param['stu_id'] = stu_id
        # user_info = rec_obj.model_irt.user_vector.loc[stu_id, :]
        stu_response = load_stu_response(param['stu_id'], train_data)
        # 从候选集合中剔除已作答过的题目
        stu_candidate_items = candidate_items.drop(stu_response.index, errors='ignore')

        stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)

        result = rec_obj.get_rec(stu_id, stu_acc=stu_acc, candidate_items=stu_candidate_items)
        if len(result) == 0:
            rec_d = -1
        else:
            rec_d = result.iloc[0]['b']
        # user_info['rec_difficulty'] = rec_difficulty
        rec_difficulty.append(rec_d)

    user_info = rec_obj.model_irt.user_vector
    user_info['rec_b'] = rec_difficulty
    file_name = '推荐结果测试.xlsx'
    writer = pd.ExcelWriter(file_name)
    user_info.to_excel(writer, encoding="UTF-8")
    writer.save()

    # print(user_info, file=sys.stderr)

    # rec_obj.save_model()

    # rec_obj.load_model()

    # print(candidate_items.sort_values('prob'), file=sys.stderr)

    # print(result, file=sys.stderr)

    # print(json.dumps(result))
    # print('-' * 10, 'metrix', '-' * 10, file=sys.stderr)

    return


# c_sortorder


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"输出文件；默认标准输出设备")
    parser.add_argument("-r", "--run", dest="run", choices=['online', 'test_one', 'test_level'], default='online',
                        help=u"运行模式")
    parser.add_argument("-l", "--log", dest="log", choices=['info', 'warning', 'debug', 'error'], default='info',
                        help=u"运行模式")

    return parser


if __name__ == '__main__':

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
