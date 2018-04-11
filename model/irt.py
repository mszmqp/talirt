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
from sklearn import metrics
import abc
import matplotlib.pyplot as plt
import sys


class BaseIrt(object):

    def __init__(self, response_df: pd.DataFrame):
        """

        :param response_df: 必须包含三列 user_id item_id answer
        """

        if 'user_id' not in response_df.columns or 'item_id' not in response_df.columns or 'answer' not in response_df.columns:
            raise ValueError("input dataframe have no user_id or item_id  or answer")

        self._response = response_df[['user_id', 'item_id', 'answer']]
        # 被试者id
        self._user_ids = self._response['user_id'].unique()
        self._user_count = len(self._user_ids)
        # 项目id
        self._item_ids = self._response['item_id'].unique()
        self._item_count = len(self._item_ids)

        # if padding_item > 0 or padding_user > 0:
        #     self._padding(user_count=padding_user, item_count=padding_item)
        #     self._user_ids = self._response['user_id'].unique()
        #     self._user_count = len(self._user_ids)
        #     self._item_ids = self._response['item_id'].unique()
        #     self._item_count = len(self._item_ids)
        # 被试id和下标的映射关系 key:id value:iloc
        # self._user_id_loc = {value: index for index, value in enumerate(self._user_ids)}
        # self._item_id_loc = {value: index for index, value in enumerate(self._item_ids)}

        self.user_vector = pd.DataFrame({
            'iloc': np.arange(self._user_count),
            'user_id': self._user_ids,
            'theta': np.zeros(self._user_count)},
            index=self._user_ids)
        self.item_vector = pd.DataFrame(
            {'iloc': np.arange(self._item_count),
             'item_id': self._item_ids,
             'a': np.zeros(self._item_count),
             'b': np.zeros(self._item_count),
             'c': np.zeros(self._item_count)}, index=self._item_ids)

        self._response = self._response.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id', how='left')
        self._response = self._response.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id', how='left')
        # 统计每个应试者的作答情况
        user_stat = self._response.groupby('user_id')['answer'].aggregate({'count': np.size, 'right': np.sum})
        self.user_vector = self.user_vector.join(user_stat, how='left')
        self.user_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.user_vector['accuracy'] = self.user_vector['right'] / self.user_vector['count']
        # 统计每个项目的作答情况
        item_stat = self._response.groupby('item_id')['answer'].aggregate({'count': np.size, 'right': np.sum})
        self.item_vector = self.item_vector.join(item_stat, how='left')
        self.item_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.item_vector['accuracy'] = self.item_vector['right'] / self.item_vector['count']

    def __str__(self):
        d = self._response['answer'].value_counts()
        return '\n'.join([
            u"用户数量：%d" % self._user_count,
            u"项目数量：%d" % self._item_count,
            u"记录总数：%d" % len(self._response),
            u'正确数量：%d' % d[1],
            u'错误数量：%d' % d[0],
            u'正确比例：%f%%' % (d[1] * 100.0 / d.sum()),
        ])

    @abc.abstractmethod
    def predict_proba(self, users, items):
        pass

    @abc.abstractmethod
    def predict(self, threshold=0.5):
        pass

    def metric_mean_error(self, y_true, y_proba):
        # y_hat_matrix = self.predict_proba()
        # y_hat = y_hat_matrix[self._user_response_locs, self._item_response_locs]
        # y_true = self._response["answer"].values
        assert len(y_proba) == len(y_true)
        error = {
            'mse': metrics.mean_squared_error(y_true, y_proba),
            'mae': metrics.mean_absolute_error(y_true, y_proba),
        }
        print('=' * 20 + 'mean_error' + "=" * 20, file=sys.stderr)
        print("mse", error['mse'], file=sys.stderr)
        print("mae", error['mae'], file=sys.stderr)
        return error

    def plot_prc(self, y_true, y_proba):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)
        print('=' * 20 + 'precision_recall_curve' + "=" * 20, file=sys.stderr)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        return precision, recall, thresholds

    def confusion_matrix(self, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        cm = metrics.confusion_matrix(y_true, y_pred)
        print('=' * 20 + 'confusion_matrix' + "=" * 20, file=sys.stderr)
        print("_\t预假\t预真", file=sys.stderr)
        print("实假\tTN(%d)\tFP(%d)" % (cm[0][0], cm[0][1]), file=sys.stderr)
        print("实真\tFN(%d)\tTP(%d)" % (cm[1][0], cm[1][1]), file=sys.stderr)
        return cm

    def classification_report(self, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        print('=' * 20 + 'classification_report' + "=" * 20, file=sys.stderr)
        print(metrics.classification_report(y_true, y_pred, target_names=[u'答错', u'答对'], digits=8), file=sys.stderr)

    def accuracy_score(self, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        print('=' * 20 + 'accuracy_score' + "=" * 20, file=sys.stderr)
        print(metrics.accuracy_score(y_true, y_pred), file=sys.stderr)


class UIrt2PL(BaseIrt):
    def __init__(self, response_df: pd.DataFrame, beta_bounds=(1, 5),
                 init_theta=None, init_alpha=None, init_beta=None,
                 *args, **kwargs):
        # print(args)
        # print(kwargs)
        super(UIrt2PL, self).__init__(response_df, *args, **kwargs)
        """
           :param init_alpha: 斜率，题目参数
           :param init_beta: 题目难度
           :param max_iter: EM算法最大迭代次数
           :param tol: 精度
           :param gp_size: Gauss–Hermite积分点数
           """
        #
        # self._beta_bounds = beta_bounds
        #
        # if init_alpha is not None:
        #     self._init_alpha = init_alpha
        # else:
        #     self._init_alpha = np.ones(self._item_count)
        #     # self._init_alpha.reshape()
        # if init_beta is not None:
        #     self._init_beta = init_beta
        # else:
        #     self._init_beta = np.linspace(beta_bounds[0], beta_bounds[1] + 1, self._item_count)
        #
        # if init_theta is None:
        #     self._init_theta = np.linspace(beta_bounds[0], beta_bounds[1] + 1, self._user_count)
        #     # self._init_theta.reshape(self._user_count, 1)

        self.trace = None

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
            theta = pm.Normal("theta", mu=2, sd=0.662551, shape=(self._user_count, 1))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self._item_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(1, self._item_count))
            b = pm.Normal("b", mu=2, sd=0.662551, shape=(1, self._item_count))
            # z = pm.Deterministic(name="z", var=b.repeat(self._user_count, axis=0) - pm.math.dot(theta, a))
            z = pm.Deterministic(name="z", var=a.repeat(self._user_count, axis=0) * (
                    theta.repeat(self._item_count, axis=1) - b.repeat(self._user_count, axis=0)))

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
        user_v = self.user_vector.loc[users, ['theta']]
        item_v = self.item_vector.loc[items, ['a', 'b']]
        z = item_v['a'].values * (user_v['theta'].values - item_v['b'].values)
        e = np.exp(z)
        s = e / (1.0 + e)
        return s

    def predict(self, threshold=0.5):
        proba = self.predict_proba()
        # proba


class UIrt3PL(UIrt2PL):

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
            theta = pm.Normal("theta", mu=2, sd=0.662551, shape=(self._user_count, 1))
            # a = pm.Normal("a", mu=1, tau=1, shape=(1, self._item_count))
            a = pm.Lognormal("a", mu=0, tau=1, shape=(1, self._item_count))
            b = pm.Normal("b", mu=2, sd=0.662551, shape=(1, self._item_count))
            c = pm.Normal("c", mu=2, sd=0.662551, shape=(1, self._item_count))
            # z = pm.Deterministic(name="z", var=b.repeat(self._user_count, axis=0) - pm.math.dot(theta, a))
            z = pm.Deterministic(name="z", var=a.repeat(self._user_count, axis=0) * (
                    theta.repeat(self._item_count, axis=1) - b.repeat(self._user_count, axis=0)))

            irt = pm.Deterministic(name="irt",
                                   var=(1 - c.repeat(self._user_count, axis=0)) * pm.math.sigmoid(z)+c.repeat(self._user_count, axis=0))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self._response['user_iloc'], self._response['item_iloc']])
            correct = pm.Bernoulli('correct', p=output, observed=self._response["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
            self.trace = pm.sample(**kwargs)

        self.item_vector['a'] = self.trace['a'].mean(axis=0)[0, :]
        self.item_vector['b'] = self.trace['b'].mean(axis=0)[0, :]
        self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
        self.user_vector['theta'] = self.trace['theta'].mean(axis=0)[:, 0]

    def predict_proba(self, users, items):
        user_v = self.user_vector.loc[users, ['theta']]
        item_v = self.item_vector.loc[items, ['a', 'b']]
        z = item_v['a'].values * (user_v['theta'].values - item_v['b'].values)
        # z = alpha * (theta - beta)
        e = np.exp(z)
        s = (1 - item_v['c']) * e / (1.0 + e)
        return s
