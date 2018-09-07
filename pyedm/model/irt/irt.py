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
# import sys
# import abc
import shutil
from pymc3.backends.base import MultiTrace
from scipy.special import expit as sigmod
from pyedm.utils.pymc import TextTrace, SQLiteTrace
# from pymc3.sampling import _cpu_count
from scipy.optimize import minimize
from tqdm import tqdm
import logging

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


class BaseIrt(object):

    def __init__(self, k=1, D=1.702):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        self.D = D
        self.k = k
        self.user_vector = None
        self.item_vector = None
        self.item_count = 0
        self.user_count = 0
        self.item_ids = None
        self.user_ids = None
        self.response_matrix = None
        self.response_sequence = None
        self.logger = logging.getLogger()
        self.model = "U2PL"

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

    def name(self):
        return type(self).__name__

    def get_trace(self, model, chains, trace_class=SQLiteTrace):
        return None
        trace_name = "trace_" + self.name() + '.db'
        if os.path.exists(trace_name):
            shutil.rmtree(trace_name)
        return MultiTrace([trace_class(chain=i, name=trace_name, model=model) for i in range(chains)])

    def _prob(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
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
        # print('prob', a.ndim, b.ndim, c.ndim,theta.ndim)
        z = self.D * a * (theta.reshape(len(theta), 1) - b)
        # print(type(z))
        if c is None:
            return sigmod(z)
        return c + (1 - c) * sigmod(z)

    def _object_func(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray,
                     c: np.ndarray, y: np.ndarray):
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
        y_hat = self._prob(theta=theta, a=a, b=b, c=c)
        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        obj = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        # 用where处理不了空值，如果是空值，where认为是真
        # obj = - np.sum(np.where(y, np.log(y_hat), np.log(1 - y_hat)))
        # print('obj', obj)
        # 目标函数没有求平均
        return - np.sum(np.nan_to_num(obj, copy=False))

    def _jac_theta(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray,
                   c: np.ndarray, y: np.ndarray):
        # print('jac',a.ndim, b.ndim, c.ndim, theta.ndim)
        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b, c=c)
        # 一阶导数
        # 每一列是一个样本，求所有样本的平均值
        all = self.D * a * (y_hat - y)

        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        grd = np.sum(np.nan_to_num(all, copy=False), axis=1)
        # grd = grd.reshape(len(grd), 1)
        # print(grd.shape, file=sys.stderr)
        return grd

    def _hessian_theta(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, y: np.ndarray):
        theta = theta.reshape(len(theta), 1)
        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b, c=c)

        # return np.sum(y_hat * (1 - y_hat) * self.D ** 2 * a * a, axis=1)
        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        tmp = self.D * self.D * a * y_hat * (1 - y_hat)
        np.where(np.isnan(y), 0, tmp)
        hess = np.dot(tmp, a.T)
        # print(hess.shape, file=sys.stderr)
        return hess

    def estimate_theta(self, method='CG', tol=None, options=None, bounds=None, progressbar=True):
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
        if method not in ["CG", "Newton-CG", "L-BFGS-B"]:
            raise ValueError('不支持的优化算法')
        item_count = len(self.item_vector)
        if self.model == "U1PL":
            a = np.ones(shape=(1, item_count))
            b = self.item_vector.loc[:, 'b'].values.reshape(1, item_count)
            c = np.zeros(shape=(1, self.item_count))
        elif self.model == "U2PL":
            a = self.item_vector.loc[:, 'a'].values.reshape(1, item_count)
            b = self.item_vector.loc[:, 'b'].values.reshape(1, item_count)
            c = np.zeros(shape=(1, self.item_count))
        elif self.model == "U3PL":
            a = self.item_vector.loc[:, 'a'].values.reshape(1, item_count)
            b = self.item_vector.loc[:, 'b'].values.reshape(1, item_count)
            c = self.item_vector.loc[:, 'c'].values.reshape(1, item_count)
        else:
            raise ValueError("unknown model " + self.model)

        if method == "Newton-CG":
            hessian = self._hessian_theta
        else:
            hessian = None

        success = []

        # if join:
        #
        #     # 注意y可能有缺失值
        #     y = self.response_matrix.values
        #     theta = self.user_vector.loc[:, ['theta']].values.reshape(self.user_count, 1)
        #
        #     res = minimize(self._object_func, x0=theta, args=(y, a, b, c), method=method, jac=self._jac_theta,
        #                    bounds=bounds, hess=hessian, options=options, tol=tol)
        #
        #     self.user_vector.loc[:, ['theta']] = res.x
        #
        #     # y_list.append(y)
        #     # theta_list.append(theta)
        #     success.append(res.success)
        #     self._es_res_theta.append(res)
        # else:

        if progressbar:
            iter_rows = tqdm(self.response_matrix.iterrows(), total=len(self.response_matrix))
        else:
            iter_rows = self.response_matrix.iterrows()
        # 每个人独立估计
        res_list = []
        for index, row in iter_rows:
            # 注意y可能有缺失值
            y = row.values.reshape(1, len(row))
            theta = self.user_vector.loc[index, 'theta']

            res = minimize(self._object_func, x0=np.array([theta]), args=(a, b, c, y), method=method,
                           jac=self._jac_theta,
                           bounds=bounds, hess=hessian, options=options, tol=tol)
            self.user_vector.loc[index, 'theta'] = res.x[0]
            success.append(res.success)
            res_list.append(res)

        return all(success), res_list

    def estimate_both_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(5, 17)
            # theta (proficiency params) are sampled from a normal distribution
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, 1))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self.item_count))
            if self.model == 'U1PL':
                a = pm.Deterministic(name='a', var=np.ones(shape=(1, self.item_count)))
            else:
                a = pm.Lognormal("a", mu=0, tau=1, shape=(1, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))

            if self.model == "U3PL":
                c = pm.Beta("c", alpha=5, beta=17, shape=(1, self.item_count))
            else:
                c = pm.Deterministic(name='c', var=np.zeros(shape=(1, self.item_count)))

            # z = pm.Deterministic(name="z", var=a.repeat(self.user_count, axis=0) * (
            #         theta.repeat(self.item_count, axis=1) - b.repeat(self.user_count, axis=0)))
            # irt = pm.Deterministic(name="irt",
            #                        var=pm.math.sigmoid(z))
            irt = pm.Deterministic(name="irt", var=(1 - c) * pm.math.sigmoid(theta * a - a * b) + c)
            # irt = pm.Deterministic(name="irt", var=pm.math.sigmoid(theta * a - a * b))
            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self.response_sequence['user_iloc'], self.response_sequence['item_iloc']])
            # observed = pm.Bernoulli('observed', p=irt, observed=self.response_matrix)
            observed = pm.Bernoulli('observed', p=output, observed=self.response_sequence["answer"].values)

            kwargs['discard_tuned_samples'] = False
            # kwargs['start'] = pm.find_MAP()

            self.trace = pm.sample(**kwargs)

        # self.alpha = self.trace['a'].mean(axis=0)[0, :]
        if self.model != "U1PL":
            self.item_vector['a'] = self.trace['a'].mean(axis=0)[0, :]

        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        if self.model == 'U3PL':
            self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]

        self.user_vector['theta'] = self.trace['theta'].mean(axis=0)[:, 0]
        return True, self.trace

    def predict_records(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"

        user_v = self.user_vector.loc[users, ['theta']]
        item_v = self.item_vector.loc[items, ['a', 'b', 'c']]
        theta = user_v['theta'].values

        a = item_v['a'].values
        b = item_v['b'].values
        c = item_v['c'].values

        z = self.D * a * (theta - b)
        e = np.exp(z)
        s = (1 - c) * e / (1.0 + e) + c
        return s

    def predict_matrix(self, users, items):
        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, 'theta'].values.reshape((user_count, 1))
        a = self.item_vector.loc[items, 'a'].values.reshape((1, item_count))
        b = self.item_vector.loc[items, 'b'].values.reshape((1, item_count))
        c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))

        z = a.repeat(user_count, axis=0) * (
                theta.repeat(item_count, axis=1) - b.repeat(user_count, axis=0))

        e = np.exp(z)
        c = c.repeat(user_count, axis=0)
        s = c + (1 - c) * e / (1.0 + e)
        return s


class UIrt1PL(BaseIrt):

    def __init__(self, **kwargs):
        super(UIrt1PL, self).__init__(**kwargs)
        self.model = 'U1PL'
        self.k = 1


class UIrt2PL(BaseIrt):

    def __init__(self, **kwargs):
        super(UIrt2PL, self).__init__(**kwargs)
        self.model = 'U2PL'
        self.k = 1


class UIrt3PL(BaseIrt):
    def __init__(self, **kwargs):
        super(UIrt3PL, self).__init__(**kwargs)
        self.model = 'U3PL'
        self.k = 1


class MIrt2PL(BaseIrt):
    """
    补偿型2参数多维irt模型
    # :param Q: shape=(k,n_items) ，Q矩阵，用于表示每个项目考察了哪些维度的属性，k代表属性维度数量。
    # Q[i,j]=1,表示第j个项目考察了第一个维度的属性； Q[i,j]=0,表示没有考察

    """

    #
    def __init__(self, k: int = 5, *args, **kwargs):
        super(MIrt2PL, self).__init__(*args, **kwargs)
        #     self.Q = Q.join(self.item_vector['iloc']).set_index('iloc').sort_index().values
        #     m, self.k = self.Q.shape
        #     self.Q = self.Q.reshape(self.k, m)
        #     assert m == self.item_count
        self.k = k
        self.model = "M2PL"

    def estimate_theta(self, method='CG', tol=None, options=None, bounds=None, progressbar=True):
        raise NotImplemented

    def predict_records(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"
        user_v = self.user_vector.loc[users, ['theta_%d' % i for i in range(self.k)]].values
        item_a = self.item_vector.loc[items, ['a_%d' % i for i in range(self.k)]].values
        item_b = self.item_vector.loc[items, ['b']].values.flatten()
        item_c = self.item_vector.loc[items, ['c']].values.flatten()
        # 注意本函数是按顺序求u和item的估计值，不是矩阵求解每个被试和每个项目的作答。这里不要用矩阵的dot
        z = (user_v * item_a).sum(axis=1) - item_b
        e = np.exp(z)
        s = item_c + (1 - item_c) * e / (1.0 + e)
        return s

    def predict_matrix(self, users, items):
        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, ['theta_%d' % i for i in range(self.k)]].values  # shape=(user_count,k)
        a = self.item_vector.loc[items, ['a_%d' % i for i in range(self.k)]].values.T  # shape = (k, item_count)
        b = self.item_vector.loc[items, 'b'].values.reshape((1, item_count))
        c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))
        b = b.repeat(user_count, axis=0)
        c = c.repeat(user_count, axis=0)
        z = np.dot(theta, a) - b

        e = np.exp(z)
        s = c + (1 - c) * e / (1.0 + e)
        return s

    def estimate_both_mcmc(self, **kwargs):
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta \sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            # theta (proficiency params) are sampled from a normal distribution
            # theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, self.k))
            theta = pm.MvNormal(name="theta", mu=np.zeros(self.k), cov=np.identity(self.k),
                                shape=(self.user_count, self.k))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self.item_count))
            a = pm.Lognormal(name="a", mu=0, tau=1, shape=(self.k, self.item_count))
            b = pm.Normal(name="b", mu=0, sd=1, shape=(1, self.item_count))
            if self.model == "M3PL":
                c = pm.Beta(name="c", alpha=2, beta=5, shape=(1, self.item_count))
            else:
                c = pm.Deterministic(name='c', var=np.zeros(shape=(1, self.item_count)))

            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a * self.Q) - b.repeat(self.user_count, axis=0))
            z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))

            irt = pm.Deterministic(name="irt",
                                   var=(1 - c.repeat(self.user_count, axis=0)) * pm.math.sigmoid(z) + c.repeat(
                                       self.user_count, axis=0))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self.response_sequence['user_iloc'], self.response_sequence['item_iloc']])
            observed = pm.Bernoulli('observed', p=output, observed=self.response_sequence["answer"].values)
            # njobs = kwargs.get('njobs', 1)
            # chains = kwargs.get('chains', None)
            # if njobs is None:
            #     njobs = min(4, _cpu_count())
            # if chains is None:
            #     chains = max(2, njobs)
            # m_trace = self.get_trace(basic_model, chains)
            # kwargs['trace'] = m_trace
            kwargs['discard_tuned_samples'] = False
            # kwargs['start'] = pm.find_MAP()
            self.trace = pm.sample(**kwargs)

        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        if self.model == "M3PL":
            self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
        else:
            self.item_vector['c'] = np.zeros(shape=(self.item_count,)).flatten()

        theta = pd.DataFrame(self.trace['theta'].mean(axis=0),
                             columns=['theta_%d' % i for i in range(self.k)])
        # a = pd.DataFrame(self.trace['a'].mean(axis=0).T*self.Q.T,
        a = pd.DataFrame(self.trace['a'].mean(axis=0).T,
                         columns=['a_%d' % i for i in range(self.k)])
        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')


class MIrt3PL(MIrt2PL):
    R"""
        补偿型3参数多维irt模型
    """

    def __init__(self, k: int = 5, *args, **kwargs):
        super(MIrt3PL, self).__init__(*args, **kwargs)
        #     self.Q = Q.join(self.item_vector['iloc']).set_index('iloc').sort_index().values
        #     m, self.k = self.Q.shape
        #     self.Q = self.Q.reshape(self.k, m)
        #     assert m == self.item_count
        self.k = k
        self.model = "M3PL"


class MIrt2PLN(MIrt2PL):
    R"""
    非补偿型2参数多维irt模型
    """

    def estimate_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        Parameters
        ----------

        """
        basic_model = pm.Model()

        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.k, self.user_count, 1))
            # theta = pm.MvNormal("theta", mu=np.zeros(self.k), cov=np.identity(self.k),
            #                     shape=(self.k, self.user_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(self.k, 1, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(self.k, 1, self.item_count))
            # a(theta-b)
            z = pm.Deterministic(name="z",
                                 var=a.repeat(self.user_count, axis=1) * (
                                         theta.repeat(self.item_count, axis=2) - b.repeat(self.user_count,
                                                                                          axis=1)))

            irt = pm.Deterministic(name="irt",
                                   var=pm.math.prod(pm.math.sigmoid(z), axis=0))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self.response_sequence['user_iloc'], self.response_sequence['item_iloc']])
            observed = pm.Bernoulli('observed', p=output, observed=self.response_sequence["answer"].values)
            # njobs = kwargs.get('njobs', 1)
            # chains = kwargs.get('chains', None)
            # if njobs is None:
            #     njobs = min(4, _cpu_count())
            # if chains is None:
            #     chains = max(2, njobs)
            # m_trace = self.get_trace(basic_model, chains)
            # kwargs['trace'] = m_trace
            kwargs['discard_tuned_samples'] = False
            # kwargs['start'] = pm.find_MAP()
            self.trace = pm.sample(**kwargs)
        theta = self.trace['theta'].mean(axis=0)[:, :, 0]
        theta = pd.DataFrame(theta.T, columns=['theta_%d' % i for i in range(self.k)])

        a = self.trace['a'].mean(axis=0)[:, 0, :]
        b = self.trace['b'].mean(axis=0)[:, 0, :]
        a = pd.DataFrame(a.T, columns=['a_%d' % i for i in range(self.k)])
        b = pd.DataFrame(b.T, columns=['b_%d' % i for i in range(self.k)])

        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')
        self.item_vector = self.item_vector.join(b, on="iloc", how='left')

    def predict_proba(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"
        user_v = self.user_vector.loc[users, ['theta_%d' % i for i in range(self.k)]].values  # shape=(user_count,k)
        item_a = self.item_vector.loc[items, ['a_%d' % i for i in range(self.k)]].values  # shape = (item_count,k)
        item_b = self.item_vector.loc[items, ['b_%d' % i for i in range(self.k)]].values  # shape = (item_count,k)
        item_c = self.item_vector.loc[items, 'c'].values
        z = (user_v - item_b) * item_a
        e = np.exp(z)
        s = e / (1.0 + e)
        return item_c + (1 - item_c) * np.prod(s, axis=1)

    def predict_proba_x(self, users, items):
        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, ['theta_%d' % i for i in range(self.k)]].values  # shape=(user_count,k)
        a = self.item_vector.loc[items, ['a_%d' % i for i in range(self.k)]].values  # shape = (item_count,k)
        b = self.item_vector.loc[items, ['b_%d' % i for i in range(self.k)]].values  # shape = (item_count,k)
        c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))
        theta = theta.reshap(user_count, 1, self.k).repeat(item_count, 1)

        a = a.reshape(1, item_count, self.k).repeat(user_count, axis=0)
        b = b.reshape(1, item_count, self.k).repeat(user_count, axis=0)
        c = c.repeat(user_count, axis=0)
        z = a * (theta - b)
        e = np.exp(z)
        s = e / (1.0 + e)
        return c + (1 - c) * np.prod(s, axis=2)


class MIrt3PLN(MIrt2PLN):
    R"""
    非补偿型3参数多维irt模型
    .. math::
        P(U_{ij}=1|\theta_i,a_j,b_j,c_j) = c_j + (1-c_j) \left( \prod_{k=1}^K \frac{e^{Da_{jk}(\theta_{ik}-b_{jk})}}{1+e^{Da_{jk}(\theta_{ik}-b_{jk})}} \right)

    Parameters
    ----------


    """

    def estimate_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.k, self.user_count, 1))
            # theta = pm.MvNormal("theta", mu=np.zeros(self.k), cov=np.identity(self.k),
            #                     shape=(self.k, self.user_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(self.k, 1, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(self.k, 1, self.item_count))
            c = pm.Beta("c", alpha=2, beta=5, shape=(1, self.item_count))
            # a(theta-b)
            z = pm.Deterministic(name="z",
                                 var=a.repeat(self.user_count, axis=1) * (
                                         theta.repeat(self.item_count, axis=2) - b.repeat(self.user_count,
                                                                                          axis=1)))
            # c + (1-c)*prod_k(sigmod(z))
            irt = pm.Deterministic(name="irt",
                                   var=c.repeat(self.user_count, axis=0) + (
                                           1 - c.repeat(self.user_count, axis=0)) * pm.math.prod(pm.math.sigmoid(z),
                                                                                                 axis=0))
            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self.response_sequence['user_iloc'], self.response_sequence['item_iloc']])
            observed = pm.Bernoulli('observed', p=output, observed=self.response_sequence["answer"].values)
            # njobs = kwargs.get('njobs', 1)
            # chains = kwargs.get('chains', None)
            # if njobs is None:
            #     njobs = min(4, _cpu_count())
            # if chains is None:
            #     chains = max(2, njobs)
            # m_trace = self.get_trace(basic_model, chains)
            # kwargs['trace'] = m_trace
            # 通过修改源码，不存储被burn的记录，所以这里一定要要False
            kwargs['discard_tuned_samples'] = False
            # kwargs['start'] = pm.find_MAP()
            self.trace = pm.sample(**kwargs)
        theta = self.trace['theta'].mean(axis=0)[:, :, 0]
        theta = pd.DataFrame(theta.T, columns=['theta_%d' % i for i in range(self.k)])

        a = self.trace['a'].mean(axis=0)[:, 0, :]
        b = self.trace['b'].mean(axis=0)[:, 0, :]
        a = pd.DataFrame(a.T, columns=['a_%d' % i for i in range(self.k)])
        b = pd.DataFrame(b.T, columns=['b_%d' % i for i in range(self.k)])

        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')
        self.item_vector = self.item_vector.join(b, on="iloc", how='left')
        self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
