# coding=utf-8
from __future__ import print_function
import warnings
from itertools import combinations
import numpy as np
import pymc3 as pm
import pandas as pd
# import theano.tensor as tt
from theano.tensor.basic import as_tensor_variable
import os
import sys
import abc
import shutil
from pymc3.backends.base import MultiTrace
from scipy.special import expit as sigmod
from talirt.utils.pymc import TextTrace, SQLiteTrace
from pymc3.sampling import _cpu_count
from scipy.optimize import minimize
from tqdm import tqdm
import logging
from talirt.estimator.uirt import MLE, BockAitkinEM
from talirt.utils import uirt_lib

"""
多维IRT模型中，能力值theta的先验分布是
（参考论文http://www.pacificmetrics.com/wp-content/uploads/2015/07/AERA2008_Duong_Subedi_Lee.pdf）

Examinee ability
parameters were generated from multivariate normal distribution with zero mean vector 0 and
identity covariance matrix I (i.e., non-correlated abilities). 


scipy中的多元正态分布
scipy.stats.multivariate_normal.rvs(mu_actual, cov_actual, size=N)

pymc3中的多元正态分布
https://docs.pymc.io/api/distributions/multivariate.html
cov = np.array([[1., 0.5], [0.5, 2]])
mu = np.zeros(2)
vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=(5, 2))
"""


class UIRT(object):

    def __init__(self, model="2PL", D=1.702):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        self.D = D
        self.user_vector = None
        self.item_vector = None
        self.item_count = 0
        self.user_count = 0
        self.item_ids = None
        self.user_ids = None
        self.response_matrix = None
        self.response_sequence = None
        self.logger = logging.getLogger()
        self.model = model
        self.estimator = None

    def fit(self, response: pd.DataFrame = None, orient="records", **kwargs):
        """

        Parameters
        ----------
        response  作答数据，至少包含三列 user_id item_id answer
        orient
        estimate
        kwargs

        Returns
        -------

        """

        assert response is not None
        if orient == 'records':
            assert len({'user_id', 'item_id', 'answer'}.intersection(set(response.columns))) == 3
            if 'difficulty' in response.columns and 'b' not in response.columns:
                response.rename(columns={'difficulty': 'b'}, inplace=True)
            if 'a' not in response.columns:
                response.loc[:, 'a'] = 1
            _columns = list(
                {'user_id', 'item_id', 'answer', 'a', 'b', 'c', 'theta'}.intersection(set(response.columns)))
            self.response_sequence = response.loc[:, _columns]
            self.response_matrix = self.response_sequence.pivot(index="user_id", columns="item_id", values='answer')
        elif orient == 'matrix':
            self.response_matrix = response.copy()
            self.response_matrix.index.name = 'user_id'
            # 矩阵形式生成序列数据
            self.response_sequence = pd.melt(self.response_matrix.reset_index(), id_vars=['user_id'],
                                             var_name="item_id",
                                             value_name='answer')
            # 去掉空值
            self.response_sequence.dropna(inplace=True)

        labels = set(self.response_sequence.columns).intersection({'item_id', 'a', 'b', 'c'})
        self.item_vector = self.response_sequence[list(labels)].drop_duplicates(subset=['item_id'])
        self.item_vector.set_index('item_id', inplace=True)
        self.item_count = len(self.item_vector)
        self.item_ids = list(self.item_vector.index)

        if 'a' not in labels:
            self.item_vector.loc[:, 'a'] = 1
        if 'b' not in labels:
            self.item_vector.loc[:, 'b'] = 0
        if 'c' not in labels:
            self.item_vector.loc[:, 'c'] = 0

        if self.model == "U1PL":
            self.item_vector.loc[:, 'a'] = 1
            self.item_vector.loc[:, 'c'] = 0
        elif self.model == "U2PL":
            self.item_vector.loc[:, 'c'] = 0
        elif self.model == "U3PL":
            pass
        else:
            raise ValueError("unknown model " + self.model)

        labels = set(self.response_sequence.columns).intersection({'user_id', 'theta'})
        self.user_vector = self.response_sequence[list(labels)].drop_duplicates(subset=['user_id'])
        self.user_vector.set_index('user_id', inplace=True)
        self.user_count = len(self.user_vector)
        self.user_ids = list(self.user_vector.index)
        if 'theta' not in labels:
            self.user_vector.loc[:, 'theta'] = 0

        self.item_vector.loc[:, 'iloc'] = np.arange(self.item_count)
        self.user_vector.loc[:, 'iloc'] = np.arange(self.user_count)

        self.response_sequence = self.response_sequence.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id',
                                                             how='left')
        self.response_sequence = self.response_sequence.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id',
                                                             how='left')
        # 统计每个应试者的作答情况
        user_stat = self.response_sequence.groupby('user_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})

        # 注意：难度是浮点数，需要先转换为整型，然后在统计每个难度的分布
        # 统计每个难度的作答情况
        # x = self.response_sequence.astype({'b': 'int32'}).groupby(['user_id', 'b']).aggregate(
        #     {'answer': ['count', 'sum']})
        # y = x.unstack()
        # y.columns = [(col[1] + "_" + str(int(col[2]))).strip().replace('sum', 'right') for col in y.columns.values]
        #
        # for i in range(1, 6):
        #     i = int(i)
        #     if 'right_%s' % i in y.columns:
        #         y.loc[:, 'accuracy_%s' % i] = y['right_%s' % i] / y['count_%s' % i]
        #
        # y.loc[:, 'count_all'] = y.filter(regex='^count_', axis=1).sum(axis=1)
        # y.loc[:, 'right_all'] = y.filter(regex='^right_', axis=1).sum(axis=1)
        # y.loc[:, 'accuracy_all'] = y['right_all'] / y['count_all']
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
            self.user_ids = list(values.index)

            self.user_vector = pd.DataFrame({
                'iloc': np.arange(self.user_count),
                'user_id': self.user_ids,
                'theta': values.loc[:, 'theta'].values.flatten(),
            },
                index=self.user_ids)

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.user_vector.loc[values.index, 'theta'] = values.loc[:, 'theta'].values.flatten()

            elif isinstance(values, np.ndarray):
                assert len(self.user_vector) == len(values)
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
            self.item_ids = list(values.index)

            self.item_vector = pd.DataFrame({
                'iloc': np.arange(self.item_count),
                'item_id': self.item_ids,
                'a': np.ones(self.item_count),
                'b': np.zeros(self.item_count),
                'c': np.zeros(self.item_count),

            },
                index=self.item_ids)

            self.item_vector.loc[:, columns] = values.loc[:, columns].values

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.item_vector.loc[values.index, columns] = values.loc[:, columns].values

            elif isinstance(values, np.ndarray):
                self.item_vector.loc[:, columns] = values

            else:
                raise TypeError("values的类型必须是pandas.DataFrame或numpy.ndarray")

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.item_vector.to_pickle(os.path.join(path, 'item_vector.pk'))
        self.user_vector.to_pickle(os.path.join(path, 'user_vector.pk'))

    def load_model(self, path):
        self.item_vector = pd.read_pickle(os.path.join(path, 'item_vector.pk'))
        self.user_vector = pd.read_pickle(os.path.join(path, 'user_vector.pk'))
        self.user_count = len(self.user_vector)
        self.item_count = len(self.item_vector)
        self.user_ids = self.user_vector['user_id'].unique()
        self.item_ids = self.item_vector['item_id'].unique()

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_model(path)
        return model

    def info(self):
        if self.response_sequence is None:
            return "no data"
        d = self.response_sequence['answer'].value_counts()
        return '\n'.join([
            u"用户数量：%d" % self.user_count,
            u"项目数量：%d" % self.item_count,
            u"记录总数：%d" % len(self.response_sequence),
            u'正确数量：%d' % d[1],
            u'错误数量：%d' % d[0],
            u'正确比例：%f%%' % (d[1] * 100.0 / d.sum()),
        ])

    def _get_abc(self):
        if self.model == "1PL":
            # a = np.ones(shape=(1, self.item_count))
            b = self.item_vector.loc[:, 'b'].values
            # c = np.zeros(shape=(1, self.item_count))
            return None, b, None
        elif self.model == "2PL":
            a = self.item_vector.loc[:, 'a'].values
            b = self.item_vector.loc[:, 'b'].values
            # c = np.zeros(shape=(1, self.item_count))
            return a, b, None
        elif self.model == "3PL":
            a = self.item_vector.loc[:, 'a'].values
            b = self.item_vector.loc[:, 'b'].values
            c = self.item_vector.loc[:, 'c'].values
            return a, b, c

    def estimate_theta(self, **kwargs):
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
        a, b, c = self._get_abc()
        self.estimator = MLE(model=self.model)
        self.estimator.fit(response=self.response_matrix.values, a=a, b=b, c=c)
        self.estimator.estimate_theta()
        self.user_vector.loc[:, 'theta'] = self.estimator.theta

    def estimate_item(self, **kwargs):
        a, b, c = self._get_abc()
        theta = self.user_vector[:, 'theta'].values
        self.estimator = MLE(model=self.model)
        self.estimator.fit(response=self.response_matrix.values, a=a, b=b, c=c, theta=theta)
        self.estimator.estimate_item(**kwargs)

        if self.model == "3PL":
            self.item_vector.loc[:, 'a'] = self.estimator.a
            self.item_vector.loc[:, 'b'] = self.estimator.b
            self.item_vector.loc[:, 'c'] = self.estimator.c
        elif self.model == "2PL":

            self.item_vector.loc[:, 'a'] = self.estimator.a
            self.item_vector.loc[:, 'b'] = self.estimator.b
        else:
            self.item_vector.loc[:, 'b'] = self.estimator.b

    def estimate_join(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        a, b, c = self._get_abc()
        self.estimator = BockAitkinEM(model=self.model)

        self.estimator.fit(response=self.response_matrix.values, a=a, b=b, c=c)
        self.estimator.estimate_join(**kwargs)

    def predict_records(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"

        user_v = self.user_vector.loc[users, ['theta']]
        # item_v = self.item_vector.loc[items, ['a', 'b', 'c']]
        theta = user_v['theta'].values

        if self.model == "3PL":
            a = self.item_vector.loc[items, 'a'].values
            b = self.item_vector.loc[items, 'b'].values
            c = self.item_vector.loc[items, 'c'].values
            return uirt_lib.u3irt_sequence(theta=theta, a=a, b=b, c=c, contant=self.D)
        elif self.model == "2PL":
            a = self.item_vector.loc[items, 'a'].values
            b = self.item_vector.loc[items, 'b'].values
            return uirt_lib.u2irt_sequence(theta=theta, a=a, b=b, contant=self.D)
        else:
            b = self.item_vector.loc[items, 'b'].values
            return uirt_lib.u1irt_sequence(theta=theta, b=b, contant=self.D)

    def predict_matrix(self, users=None, items=None):

        theta = self.user_vector.loc[users, 'theta'].values
        if self.model == "3PL":
            a = self.item_vector.loc[items, 'a'].values
            b = self.item_vector.loc[items, 'b'].values
            c = self.item_vector.loc[items, 'c'].values
            return uirt_lib.u3irt_matrix(theta=theta, a=a, b=b, c=c, contant=self.D)
        elif self.model == "2PL":
            a = self.item_vector.loc[items, 'a'].values
            b = self.item_vector.loc[items, 'b'].values
            return uirt_lib.u2irt_matrix(theta=theta, a=a, b=b, contant=self.D)
        else:
            b = self.item_vector.loc[items, 'b'].values
            return uirt_lib.u1irt_matrix(theta=theta, b=b, contant=self.D)
