# coding=utf-8
from __future__ import print_function
import warnings
from itertools import combinations
import numpy as np
# from psy.utils import inverse_logistic, get_nodes_weights
# from psy.fa import GPForth, Factor
# from psy.settings import X_WEIGHTS, X_NODES
import pymc3 as pm
import pandas as pd
import theano.tensor as tt
from theano.tensor.basic import as_tensor_variable

import abc

import sys


class BaseIrt(object):

    def __init__(self, response: pd.DataFrame):
        """

        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        """

        if 'user_id' not in response.columns or 'item_id' not in response.columns or 'answer' not in response.columns:
            raise ValueError("input dataframe have no user_id or item_id  or answer")

        self._response = response[['user_id', 'item_id', 'answer']]
        # 被试者id
        self._user_ids = self._response['user_id'].unique()
        self.user_count = len(self._user_ids)
        # 项目id
        self._item_ids = self._response['item_id'].unique()
        self.item_count = len(self._item_ids)

        self.user_vector = pd.DataFrame({
            'iloc': np.arange(self.user_count),
            'user_id': self._user_ids,
            'theta': np.zeros(self.user_count)},
            index=self._user_ids)
        self.item_vector = pd.DataFrame(
            {'iloc': np.arange(self.item_count),
             'item_id': self._item_ids,
             'a': np.zeros(self.item_count),
             'b': np.zeros(self.item_count),
             'c': np.zeros(self.item_count)}, index=self._item_ids)

        self._response = self._response.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id', how='left')
        self._response = self._response.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id', how='left')
        # 统计每个应试者的作答情况
        user_stat = self._response.groupby('user_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})
        self.user_vector = self.user_vector.join(user_stat, how='left')
        self.user_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.user_vector['accuracy'] = self.user_vector['right'] / self.user_vector['count']
        # 统计每个项目的作答情况
        item_stat = self._response.groupby('item_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})
        self.item_vector = self.item_vector.join(item_stat, how='left')
        self.item_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.item_vector['accuracy'] = self.item_vector['right'] / self.item_vector['count']

        self.trace = None

    def info(self):
        d = self._response['answer'].value_counts()
        return '\n'.join([
            u"用户数量：%d" % self.user_count,
            u"项目数量：%d" % self.item_count,
            u"记录总数：%d" % len(self._response),
            u'正确数量：%d' % d[1],
            u'错误数量：%d' % d[0],
            u'正确比例：%f%%' % (d[1] * 100.0 / d.sum()),
        ])

    def name(self):
        return type(self).__name__

    @abc.abstractmethod
    def predict_proba(self, users, items):
        raise NotImplemented

    @abc.abstractmethod
    def predict_proba_x(self, users, items):
        raise NotImplemented

    @classmethod
    def predict(cls, users, items, threshold=0.5):
        proba = cls.predict_proba(users, items)
        proba[proba >= threshold] = 1
        proba[proba < threshold] = 0
        return proba


class UIrt2PL(BaseIrt):

    def estimate_em(self, max_iter=1000, tol=1e-5):
        pass

    def estimate_mcmc(self, **kwargs):
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
            a = pm.Lognormal("a", mu=0, tau=1, shape=(1, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))
            z = pm.Deterministic(name="z", var=a.repeat(self.user_count, axis=0) * (
                    theta.repeat(self.item_count, axis=1) - b.repeat(self.user_count, axis=0)))

            irt = pm.Deterministic(name="irt",
                                   var=pm.math.sigmoid(z))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
            self.trace = pm.sample(**kwargs)

            # run an interactive MCMC sampling session
            # m.isample()

        # self.alpha = self.trace['a'].mean(axis=0)[0, :]
        self.item_vector['a'] = self.trace['a'].mean(axis=0)[0, :]
        # self.beta = self.trace['b'].mean(axis=0)[0, :]
        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        self.user_vector['theta'] = self.trace['theta'].mean(axis=0)[:, 0]

        # print(pm.summary(self.trace))
        # _ = pm.traceplot(trace)

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


class UIrt3PL(UIrt2PL):

    def estimate_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c \sim beta(5, 17)
            # theta (proficiency params) are sampled from a normal distribution
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, 1))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self.item_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(1, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))
            c = pm.Beta("c", alpha=5, beta=17, shape=(1, self.item_count))

            z = pm.Deterministic(name="z", var=a.repeat(self.user_count, axis=0) * (
                    theta.repeat(self.item_count, axis=1) - b.repeat(self.user_count, axis=0)))
            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))

            irt = pm.Deterministic(name="irt",
                                   var=(1 - c.repeat(self.user_count, axis=0)) * pm.math.sigmoid(z) + c.repeat(
                                       self.user_count, axis=0))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            self.trace = pm.sample(**kwargs)

        self.item_vector['a'] = self.trace['a'].mean(axis=0)[0, :]
        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
        self.user_vector['theta'] = self.trace['theta'].mean(axis=0)[:, 0]


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

    def estimate_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, self.k))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(self.k, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))
            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a * self.Q) - b.repeat(self.user_count, axis=0))
            z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))

            irt = pm.Deterministic(name="irt",
                                   var=pm.math.sigmoid(z))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            self.trace = pm.sample(**kwargs)

        theta = pd.DataFrame(self.trace['theta'].mean(axis=0),
                             columns=['theta_%d' % i for i in range(self.k)])

        # a = pd.DataFrame(self.trace['a'].mean(axis=0).T * self.Q.T,
        a = pd.DataFrame(self.trace['a'].mean(axis=0).T,
                         columns=['a_%d' % i for i in range(self.k)])

        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')
        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]

    def predict_proba(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"
        user_v = self.user_vector.loc[users, ['theta_%d' % i for i in range(self.k)]].values
        item_a = self.item_vector.loc[items, ['a_%d' % i for i in range(self.k)]].values
        item_b = self.item_vector.loc[items, ['b']].values.flatten()
        item_c = self.item_vector.loc[items, ['c']].values.flatten()
        # 注意这里不要用矩阵的dot
        z = (user_v * item_a).sum(axis=1) - item_b
        e = np.exp(z)
        s = item_c + (1 - item_c) * e / (1.0 + e)
        return s

    def predict_proba_x(self, users, items):
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


class MIrt3PL(MIrt2PL):
    """
    补偿型3参数多维irt模型
    # :param Q: shape=(k,n_items) ，Q矩阵，用于表示每个项目考察了哪些维度的属性，k代表属性维度数量。
    # Q[i,j]=1,表示第j个项目考察了第一个维度的属性； Q[i,j]=0,表示没有考察
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
            # theta (proficiency params) are sampled from a normal distribution
            theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, self.k))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self.item_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(self.k, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))
            c = pm.Beta("c", alpha=2, beta=5, shape=(1, self.item_count))

            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a * self.Q) - b.repeat(self.user_count, axis=0))
            z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))

            irt = pm.Deterministic(name="irt",
                                   var=(1 - c.repeat(self.user_count, axis=0)) * pm.math.sigmoid(z) + c.repeat(
                                       self.user_count, axis=0))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            self.trace = pm.sample(**kwargs)

        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
        theta = pd.DataFrame(self.trace['theta'].mean(axis=0),
                             columns=['theta_%d' % i for i in range(self.k)])
        # a = pd.DataFrame(self.trace['a'].mean(axis=0).T*self.Q.T,
        a = pd.DataFrame(self.trace['a'].mean(axis=0).T,
                         columns=['a_%d' % i for i in range(self.k)])
        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')


class MIrt2PLN(MIrt2PL):
    """
    非补偿型2参数多维irt模型
    :param Q: shape=(k,n_items) ，Q矩阵，用于表示每个项目考察了哪些维度的属性，k代表属性维度数量。
    Q[i,j]=1,表示第j个项目考察了第一个维度的属性； Q[i,j]=0,表示没有考察
    :param args:
    :param kwargs:
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
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
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
        return c + (1 - c) * np.prod(s, axis=1)


class MIrt3PLN(MIrt2PLN):
    R"""
    非补偿型3参数多维irt模型
    .. math::
        P(U_{ij}=1|\theta_i,a_j,b_j,c_j) = c_j + (1-c_j)\prod

    Parameters
    ----------
    Q: numpy.ndarray
        shape=(k,n_items) ，Q矩阵，用于表示每个项目考察了哪些维度的属性，k代表属性维度数量。
        Q[i,j]=1,表示第j个项目考察了第一个维度的属性； Q[i,j]=0,表示没有考察
    :param args:
    :param kwargs:
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
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
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
