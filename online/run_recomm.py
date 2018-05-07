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

sys.path.append("../")

path_data = './data/'
"""
模型输入数据进行训练


"""

_sim_threshold = 0.0


def log(*args):
    print(' '.join(args), file=sys.stderr)


class DBABC(object):
    def save(self, data, year, city_id, grade_id, subject_id, level_id, term_id):
        pass

    def load(self, year, city_id, grade_id, subject_id, level_id, term_id):
        pass


class MemoryDB(DBABC):

    def __init__(self, table):
        self.path = os.path.join('/tmp/cache_learn', table)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save_json(self, data, year, city_id, grade_id, subject_id, level_id, term_id):
        file_name = '_'.join([str(year), str(city_id), str(grade_id), str(subject_id), str(level_id), str(term_id)])
        with open(os.path.join(self.path, file_name), 'w') as fh:
            json.dump(data, fh)

    def save_bin(self, data, year, city_id, grade_id, subject_id, level_id, term_id):
        file_name = '_'.join([str(year), str(city_id), str(grade_id), str(subject_id), str(level_id), str(term_id)])

        with open(os.path.join(self.path, file_name), 'wb') as fh:
            fh.write(data)

    def load_json(self, year, city_id, grade_id, subject_id, level_id, term_id):
        file_name = '_'.join([str(year), str(city_id), str(grade_id), str(subject_id), str(level_id), str(term_id)])
        with open(os.path.join(self.path, file_name), 'r') as fh:
            return json.load(fh)

    def load_bin(self, year, city_id, grade_id, subject_id, level_id, term_id):
        file_name = '_'.join([str(year), str(city_id), str(grade_id), str(subject_id), str(level_id), str(term_id)])
        with open(os.path.join(self.path, file_name), 'rb') as fh:
            return fh.read()

    def file_path(self, year, city_id, grade_id, subject_id, level_id, term_id):
        file_name = '_'.join([str(year), str(city_id), str(grade_id), str(subject_id), str(level_id), str(term_id)])

        return os.path.join(self.path, file_name)


class DiskDB(MemoryDB):
    def __init__(self, table, path='./cache_learn/'):
        self.path = os.path.join(path, table)
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)


class RedisDB(DBABC):
    pass


class SimpleCF:
    default_value = None

    def __init__(self, response: pd.DataFrame = None, sequential=True):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        self.response_matrix = None
        if response is not None:
            if sequential:
                self.response_matrix = response.pivot(index="user_id", columns="item_id", values='answer')

            else:
                self.response_matrix = response
                self.response_matrix.index.name = 'user_id'

            # 答错用-1表示，答对用1表述，未作答用0
            self.response_matrix[self.response_matrix == 0] = -1
            self.response_matrix.fillna(0, inplace=True)
            # 矩阵中应该不包含全是0的行，否则不能求模

            # 被试者id
            # self._user_ids = list(self.response_matrix.index)
            # self.user_count = len(self._user_ids)
            # 项目id
            # self._item_ids = list(self.response_matrix.columns)
            # self.item_count = len(self._item_ids)

    def fit(self):
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

        # if 'item_id' not in self.response_matrix.columns:
        #     return None

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

    def serialize(self):
        fh = tempfile.TemporaryFile(mode='w+b')
        self.response_matrix.to_pickle(path=fh)
        fh.seek(0)
        data = fh.read()
        fh.close()
        return data

    @classmethod
    def unserialize(cls, data):
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

    def __init__(self, response: pd.DataFrame = None, sequential=True, k=1, D=1.702):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        if response is not None:
            if sequential:
                if 'user_id' not in response.columns or 'item_id' not in response.columns or 'answer' not in response.columns:
                    raise ValueError("input dataframe have no user_id or item_id  or answer")

                self.response_sequence = response[['user_id', 'item_id', 'answer']]
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

            # 被试者id
            self._user_ids = list(self.response_matrix.index)
            self.user_count = len(self._user_ids)
            # 项目id
            self._item_ids = list(self.response_matrix.columns)
            self.item_count = len(self._item_ids)
            self._init_model()
            labels = set(response.columns).intersection(set(['a', 'b', 'c']))
            if sequential and labels:
                item_info = response[['item_id'] + list(labels)].drop_duplicates(subset=['item_id'])
                item_info.set_index('item_id', inplace=True)
                self.set_abc(item_info, columns=list(labels))

        else:

            self.user_vector = None
            self.item_vector = None
        self.trace = None
        self.D = D
        self.k = k

    def _init_model(self):
        assert self.response_sequence is not None
        self.user_vector = pd.DataFrame({
            'iloc': np.arange(self.user_count),
            'user_id': self._user_ids,
            'theta': np.zeros(self.user_count)},
            index=self._user_ids)
        self.item_vector = pd.DataFrame(
            {'iloc': np.arange(self.item_count),
             'item_id': self._item_ids,
             'a': np.ones(self.item_count),
             'b': np.zeros(self.item_count),
             'c': np.zeros(self.item_count)}, index=self._item_ids)

        self.response_sequence = self.response_sequence.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id',
                                                             how='left')
        self.response_sequence = self.response_sequence.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id',
                                                             how='left')
        # 统计每个应试者的作答情况
        user_stat = self.response_sequence.groupby('user_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})
        self.user_vector = self.user_vector.join(user_stat, how='left')
        self.user_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.user_vector['accuracy'] = self.user_vector['right'] / self.user_vector['count']
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
            self.user_count = len(values)
            self._user_ids = list(values.index)

            self.user_vector = pd.DataFrame({
                'iloc': np.arange(self.user_count),
                'user_id': self._user_ids,
                'theta': values.loc[:, 'theta'].values.flatten(),
            },
                index=self._user_ids)

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
            self.item_count = len(values)
            self._item_ids = list(values.index)

            self.item_vector = pd.DataFrame({
                'iloc': np.arange(self.item_count),
                'item_id': self._item_ids,
                'a': np.ones(self.item_count),
                'b': np.zeros(self.item_count),
                'c': np.zeros(self.item_count),

            },
                index=self._item_ids)

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

    def predict_proba(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"

        user_v = self.user_vector.loc[users, ['theta']]
        item_v = self.item_vector.loc[items, ['a', 'b', 'c']]
        z = item_v['a'].values * (user_v['theta'].values - item_v['b'].values)
        # z = alpha * (theta - beta)
        e = np.exp(z)
        s = (1 - item_v['c'].values) * e / (1.0 + e) + item_v['c'].values
        return s

    def predict_proba_x(self, users, items):
        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, 'theta'].values.reshape((user_count, 1))
        a = self.item_vector.loc[items, 'a'].values.reshape((1, item_count))
        b = self.item_vector.loc[items, 'b'].values.reshape((1, item_count))
        c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))
        c = c.repeat(user_count, axis=0)
        z = a.repeat(user_count, axis=0) * (
                theta.repeat(item_count, axis=1) - b.repeat(user_count, axis=0))

        e = np.exp(z)
        s = c + (1 - c) * e / (1.0 + e)
        return s

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

        if 'a' in self.item_vector.columns:
            a = self.item_vector.loc[:, 'a'].values.reshape(1, self.item_count)
        else:
            a = None

        b = self.item_vector.loc[:, 'b'].values.reshape(1, self.item_count)
        # if 'c' in self.item_vector.columns:
        #     c = self.item_vector.loc[:, 'c'].values.reshape(1, self.item_count)
        # else:
        #     c = None

        success = []

        self._es_res_theta = []

        # 每个人独立估计
        for index, row in self.response_matrix.iterrows():
            # 注意y可能有缺失值
            y = row.values.reshape(1, len(row))
            theta = self.user_vector.loc[index, 'theta']

            res = minimize(self._object_func, x0=[theta], args=(y, a, b), jac=self._jac_theta,
                           bounds=bounds, options=options, tol=tol)
            self.user_vector.loc[index, 'theta'] = res.x[0]
            success.append(res.success)
            self._es_res_theta.append(res)

        return all(success)

    def fit(self):
        return self.estimate_theta()

    def serialize(self):
        return self.user_vector['theta'].to_dict()

    def unserialize(self, serialize_data):

        index = []
        theta = []
        for key, value in serialize_data.items():
            index.append(key)
            theta.append(value)
        self.set_theta(pd.DataFrame({'theta': theta}, index=index))


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

    _candidate_items = impala_client.sql(_sql).execute().set_index('item_id')
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

                    fk_student as stu_id,

                    stu_name as user_id,
                    fk_question as item_id,
                    difficulty_id as b,
                    1 as a,
                    case when asw_first_status='错误' then 0 else 1  end   answer
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
                        and asw_first_status_id=1


        """ % kwargs

    _level_response = impala_client.sql(_sql).execute()
    impala_client.close()
    return _level_response


def load_stu_response(**kwargs):
    """
    获取当前学生的答题记录
    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    global _stu_response_items
    if _stu_response_items is not None:
        return _stu_response_items
    stu_id = kwargs['stu_id']
    level_response = load_level_response(**kwargs)
    _stu_response_items = level_response.loc[level_response['user_id'] == stu_id, 'item_id'].unique()
    return _stu_response_items


class RecommendABC(object):
    model = None
    db = None

    def __init__(self, **kwargs):
        self.param = kwargs
        self.year = kwargs['year']
        self.city_id = kwargs['city_id']
        self.grade_id = kwargs['grade_id']
        self.subject_id = kwargs['subject_id']
        self.level_id = kwargs['level_id']
        self.term_id = kwargs['term_id']
        self.knowledge_id = kwargs['knowledge_id']

    @abc.abstractmethod
    def train_model(self, response: pd.DataFrame):
        raise NotImplemented

    @abc.abstractmethod
    def load_model(self):
        raise NotImplemented

    @abc.abstractmethod
    def save_model(self):
        """
        模型数据持久化保存
        Returns
        -------

        """
        raise NotImplemented

    @abc.abstractmethod
    def get_rec(self, stu_id: str, candidate_items: pd.DataFrame):
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
    db = DiskDB(table='cf')

    def train_model(self, response: pd.DataFrame):
        if response is None:
            response = load_level_response(**self.param)
        self.model = SimpleCF(response=response, sequential=True)
        ok = self.model.fit()
        return ok

    def load_model(self):
        data = self.db.load_bin(year=self.year,
                                city_id=self.city_id,
                                grade_id=self.grade_id,
                                subject_id=self.subject_id,
                                level_id=self.level_id,
                                term_id=self.term_id)
        self.model = SimpleCF.unserialize(data)

        # return True

    def save_model(self):
        data = self.model.serialize()
        self.db.save_bin(data=data,
                         year=self.year,
                         city_id=self.city_id,
                         grade_id=self.grade_id,
                         subject_id=self.subject_id,
                         level_id=self.level_id,
                         term_id=self.term_id
                         )

    def get_rec(self, stu_id: str, candidate_items: pd.DataFrame):
        """

        Parameters
        ----------
        stu_id
        candidate_items 要求item_id为index

        Returns
        -------

        """
        if self.model is None:
            self.load_model()
        if self.model is None:
            return (None, None)
        return self.model.predict(stu_id=stu_id, items=candidate_items.index)


class RecommendIRT(RecommendABC):
    '''
    # 示例程序为从csv读取文件
    '''
    db = DiskDB(table='irt')
    D = 1.702

    def train_model(self, response: pd.DataFrame):
        self.model = UIrt2PL(response, sequential=True, D=self.D)
        ok = self.model.fit()
        # print("train", ok, file=sys.stderr)
        return ok

    def info(self):
        print(self.model.user_vector.describe(), file=sys.stderr)

    def load_model(self):
        stu_theta = self.db.load_json(year=self.year,
                                      city_id=self.city_id,
                                      grade_id=self.grade_id,
                                      subject_id=self.subject_id,
                                      level_id=self.level_id,
                                      term_id=self.term_id)

        return stu_theta

    def save_model(self):
        data = self.model.serialize()
        self.db.save_json(data=data, year=self.year,
                          city_id=self.city_id,
                          grade_id=self.grade_id,
                          subject_id=self.subject_id,
                          level_id=self.level_id,
                          term_id=self.term_id)

    def get_rec(self, stu_id: str, candidate_items: pd.DataFrame):
        stu_thetas = self.load_model()
        theta = stu_thetas.get(stu_id, None)
        if theta is None:
            return (None, None)

        probs = self._predict(theta, candidate_items)
        return probs, theta

    def _predict(self, theta, items: pd.DataFrame):
        """
        把_predict直接写在推荐类里，这样在线上就不用再创建model对象了，应该能节省点时间
        Parameters
        ----------
        theta
        items

        Returns
        -------

        """
        b = items.loc[:, ['b']].values
        z = self.D * (theta - b)
        prob = sigmod(z)
        items['irt'] = prob
        return items


def recommend(**param):
    global _candidate_items, _stu_response_items, _level_response

    candidate_items = load_candidate_items(**param)
    stu_response = load_stu_response(**param)

    # 从候选集合中剔除已作答过的题目
    candidate_items.drop(stu_response, inplace=True, errors='ignore')

    # IRT 推荐策略
    rec_irt = RecommendIRT(**param)
    rec_irt.load_model()
    prob_irt, stu_theta = rec_irt.get_rec(stu_id=param['stu_id'], candidate_items=_candidate_items)

    # CF 推荐策略
    rec_cf = RecommendCF(**param)
    rec_cf.load_model()
    prob_cf, stu_vector = rec_cf.get_rec(stu_id=param['stu_id'], candidate_items=_candidate_items)

    irt_weight = 0.5
    cf_weight = 0.5
    # prob_cf 会有空值
    prob_irt * irt_weight + prob_cf * cf_weight


def main():
    param = {'year': '2018',
             'city_id': '0571',
             'grade_id': '7',
             'subject_id': 'ff80808127d77caa0127d7e10f1c00c4',
             'level_id': 'ff8080812fc298b5012fd3d3becb1248',
             'term_id': '1',
             'knowledge_id': "cb1471bd830c49c2b5ff8b833e3057bd",
             'stu_id': '黄白杰',
             }
    # 这两份数据是所有策略都要用的，所以单独进行
    global _candidate_items, _stu_response_items, _level_response

    candidate_items = load_candidate_items(**param)
    stu_response = load_stu_response(**param)
    # 从候选集合中剔除已作答过的题目
    candidate_items.drop(stu_response, inplace=True, errors='ignore')

    re_irt = RecommendIRT(**param)
    # 训练模型
    re_irt.train_model(response=_level_response)
    # 保存模型
    re_irt.save_model()
    # debug 打印训练结果概述
    re_irt.info()
    # 获取作答概率
    probs, theta = re_irt.get_rec(stu_id='黄白杰', candidate_items=_candidate_items)

    print(theta)
    print(probs)

    print('-' * 10, "cf", '-' * 10)
    re_cf = RecommendCF(**param)
    re_cf.train_model(response=_level_response)
    re_cf.save_model()
    re_cf.load_model()
    cf_probs, stu_vector = re_cf.get_rec(stu_id='黄白杰', candidate_items=_candidate_items)
    print(cf_probs)
    return


if __name__ == '__main__':
    main()
    quit(0)
    # test cf_algorithm
    re_cf = RecommendCF(year='2018', city_id='0571', grade_id='7',
                        subject_id='ff80808127d77caa0127d7e10f1c00c4',
                        level_id='ff8080812fc298b5012fd3d3becb1248',
                        term='1')

    stu = 'ff80808146248e430146271765bb0baa'
    lq = '8a53ce07d5ff409bbe276a354708f677'
    df_question = re_cf.fetch_data()
    dic_user, dic_lq, pred = re_cf.fit(df_question)
    pred = re_cf.serialize(dic_user, dic_lq, pred)
    print("CF_ALGORITHM: the score of student: " + stu + " at the item: " + lq + " is " + str(
        re_cf.predict(stu, lq, True)))

    # test irt_algorithm
    re_irt = RecommendIRT(year='2018', city_id='0571', grade_id='7',
                          subject_id='ff80808127d77caa0127d7e10f1c00c4',
                          level_id='ff8080812fc298b5012fd3d3becb1248',
                          term='1')
    stu = '009b1e101aa54843ba61a188941de4b6'
    lq = '03e61a4dc33b451294c2e3d79f4ec468'
    df_question = re_irt.obtain_from_csv()
    lq_diff, weights = re_irt.fit(df_question)
    theta_stu = re_irt.serialize(weights)
    print("IRT_ALGORITHM: the probability of student: " + stu + " at the item: " + lq + " is " + str(
        re_irt.predict(stu, lq_diff, lq, True)))
