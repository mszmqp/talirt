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

        self._response_df = response_df[['user_id', 'item_id', 'answer']]

        self._user_ids = self._response_df['user_id'].unique()
        self._user_count = len(self._user_ids)
        self._item_ids = self._response_df['item_id'].unique()
        self._item_count = len(self._item_ids)

        # if padding_item > 0 or padding_user > 0:
        #     self._padding(user_count=padding_user, item_count=padding_item)
        #     self._user_ids = self._response_df['user_id'].unique()
        #     self._user_count = len(self._user_ids)
        #     self._item_ids = self._response_df['item_id'].unique()
        #     self._item_count = len(self._item_ids)

        self._user_id_loc = {value: index for index, value in enumerate(self._user_ids)}
        self._item_id_loc = {value: index for index, value in enumerate(self._item_ids)}

        self._user_response_locs = []
        self._item_response_locs = []
        for index, row in self._response_df.iterrows():
            u_loc = self._user_id_loc[row['user_id']]
            q_loc = self._item_id_loc[row['item_id']]
            self._user_response_locs.append(u_loc)
            self._item_response_locs.append(q_loc)
        self.alpha = None
        self.beta = None
        self.c = None
        self.theta = None
        self.trace = None

    @staticmethod
    def p(z):
        # 回答正确的概率函数
        e = np.exp(z)
        p = e / (1.0 + e)
        return p

    def __str__(self):
        d = self._response_df['answer'].value_counts()
        return '\n'.join([
            u"用户数量：%d" % self._user_count,
            u"项目数量：%d" % self._item_count,
            u"记录总数：%d" % len(self._response_df),
            u'正确数量：%d' % d[1],
            u'错误数量：%d' % d[0],
            u'正确比例：%f%%' % (d[1] * 100.0 / d.sum()),
        ])

    @abc.abstractmethod
    def predict_proba(self):
        pass

    @abc.abstractmethod
    def predict(self, threshold=0.5):
        pass

    def metric_mean_error(self, y_true, y_proba):
        # y_hat_matrix = self.predict_proba()
        # y_hat = y_hat_matrix[self._user_response_locs, self._item_response_locs]
        # y_true = self._response_df["answer"].values
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

    def _lik(self, p_val):
        # 似然函数
        scores = self.scores
        loglik_val = np.dot(np.log(p_val + 1e-200), scores.transpose()) + \
                     np.dot(np.log(1 - p_val + 1e-200), (1 - scores).transpose())
        return np.exp(loglik_val)

    def _get_theta_dis(self, p_val, weights):
        # 计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val) * weights
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        _temp = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(_temp, axis=1)
        # theta下回答正确的人数分布
        right_dis = np.dot(_temp, scores)
        full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        print(np.sum(np.log(lik_wt_sum)))
        return full_dis, right_dis


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

        self._beta_bounds = beta_bounds

        if init_alpha is not None:
            self._init_alpha = init_alpha
        else:
            self._init_alpha = np.ones(self._item_count)
            # self._init_alpha.reshape()
        if init_beta is not None:
            self._init_beta = init_beta
        else:
            self._init_beta = np.linspace(beta_bounds[0], beta_bounds[1] + 1, self._item_count)

        if init_theta is None:
            self._init_theta = np.linspace(beta_bounds[0], beta_bounds[1] + 1, self._user_count)
            # self._init_theta.reshape(self._user_count, 1)

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
                                      var=as_tensor_variable(irt)[self._user_response_locs, self._item_response_locs])
            correct = pm.Bernoulli('correct', p=output, observed=self._response_df["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
            self.trace = pm.sample(**kwargs)

            # run an interactive MCMC sampling session
            # m.isample()

        # trace['a'].shape==(nsamples, 1, item_count)
        # trace['b'].shape==(nsamples, 1, item_count)
        self.alpha = self.trace['a'].mean(axis=0)[0, :]
        self.beta = self.trace['b'].mean(axis=0)[0, :]
        # trace['theta'].shape=(nsamples, user_count, 1)
        self.theta = self.trace['theta'].mean(axis=0)[:, 0]

        # print(pm.summary(self.trace))
        # _ = pm.traceplot(trace)

    def predict_proba(self, users, items):

        user_locs = [self._user_id_loc[u] for u in users]
        item_locs = [self._item_id_loc[i] for i in items]
        theta = np.array([self.theta[loc] for loc in user_locs])
        alpha = np.array([self.alpha[loc] for loc in item_locs])
        beta = np.array([self.beta[loc] for loc in item_locs])

        # theta = self.theta.reshape(user_count, 1).repeat(item_count, axis=1)
        # alpha = self.alpha.reshape(1, item_count).repeat(user_count, axis=0)
        # beta = self.beta.reshape(1, item_count).repeat(user_count, axis=0)

        z = alpha * (theta - beta)
        e = np.exp(z)
        s = e / (1.0 + e)
        return s

    def _predict_proba2(self):

        user_count = len(self.theta)
        assert len(self.alpha) == len(self.beta)
        item_count = len(self.alpha)

        theta = self.theta.reshape(user_count, 1).repeat(item_count, axis=1)
        alpha = self.alpha.reshape(1, item_count).repeat(user_count, axis=0)
        beta = self.beta.reshape(1, item_count).repeat(user_count, axis=0)

        z = alpha * (theta - beta)
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
                                   var=c + (1 - c) * pm.math.sigmoid(z))

            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[self._user_response_locs, self._item_response_locs])
            correct = pm.Bernoulli('correct', p=output, observed=self._response_df["answer"].values)

            # map_estimate = pm.find_MAP()
            # create a pymc simulation object, including all the above variables
            self.trace = pm.sample(**kwargs)

            # run an interactive MCMC sampling session
            # m.isample()

        # trace['a'].shape==(nsamples, 1, item_count)
        # trace['b'].shape==(nsamples, 1, item_count)
        self.alpha = self.trace['a'].mean(axis=0)[0, :]
        self.beta = self.trace['b'].mean(axis=0)[0, :]
        self.c = self.trace['c'].mean(axis=0)[0, :]
        # trace['theta'].shape=(nsamples, user_count, 1)
        self.theta = self.trace['theta'].mean(axis=0)[:, 0]

        # print(pm.summary(self.trace))
        # _ = pm.traceplot(trace)

    def predict_proba(self, users, items):
        user_locs = [self._user_id_loc[u] for u in users]
        item_locs = [self._item_id_loc[i] for i in items]
        theta = np.array([self.theta[loc] for loc in user_locs])
        alpha = np.array([self.alpha[loc] for loc in item_locs])
        beta = np.array([self.beta[loc] for loc in item_locs])
        c = np.array([self.c[loc] for loc in item_locs])

        # theta = self.theta.reshape(user_count, 1).repeat(item_count, axis=1)
        # alpha = self.alpha.reshape(1, item_count).repeat(user_count, axis=0)
        # beta = self.beta.reshape(1, item_count).repeat(user_count, axis=0)

        z = alpha * (theta - beta)
        e = np.exp(z)
        s = (1 - c) * e / (1.0 + e)
        return s
