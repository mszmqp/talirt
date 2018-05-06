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
import abc
import shutil
from pymc3.backends.base import MultiTrace
from scipy.special import expit as sigmod
from talirt.utils.pymc import TextTrace, SQLiteTrace
from pymc3.sampling import _cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

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

    def __init__(self, response: pd.DataFrame = None, sequential=True, k=1, D=1):
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

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.item_vector.to_pickle(os.path.join(path, 'item_vector.pl'))
        self.user_vector.to_pickle(os.path.join(path, 'user_vector.pl'))

    def load_model(self, path):
        self.item_vector = pd.read_pickle(os.path.join(path, 'item_vector.pl'))
        self.user_vector = pd.read_pickle(os.path.join(path, 'user_vector.pl'))
        self.user_count = len(self.user_vector)
        self.item_count = len(self.item_vector)
        self._user_ids = self.user_vector['user_id'].unique()
        self._item_ids = self.item_vector['item_id'].unique()

    @classmethod
    def load(cls, path):
        model = cls(response=None)
        model.load_model(path)
        return model

    def info(self):
        if self.response_sequence is None:
            return "none data"
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

    def get_trace(self, model, chains, trace_class=SQLiteTrace):
        return None
        trace_name = "trace_" + self.name() + '.db'
        if os.path.exists(trace_name):
            shutil.rmtree(trace_name)
        return MultiTrace([trace_class(chain=i, name=trace_name, model=model) for i in range(chains)])

    @abc.abstractmethod
    def estimate_mcmc(self, **kwargs):
        """Draw samples from the posterior using the given step methods.

        Multiple step methods are supported via compound step methods.

        Parameters
        ----------
        draws : int
            The number of samples to draw. Defaults to 500. The number of tuned
            samples are discarded by default. See discard_tuned_samples.
        step : function or iterable of functions
            A step function or collection of functions. If there are variables
            without a step methods, step methods for those variables will
            be assigned automatically.
        init : str
            Initialization method to use for auto-assigned NUTS samplers.

            * auto : Choose a default initialization method automatically.
              Currently, this is `'jitter+adapt_diag'`, but this can change in
              the future. If you depend on the exact behaviour, choose an
              initialization method explicitly.
            * adapt_diag : Start with a identity mass matrix and then adapt
              a diagonal based on the variance of the tuning samples. All
              chains use the test value (usually the prior mean) as starting
              point.
            * jitter+adapt_diag : Same as `adapt_diag`, but add uniform jitter
              in [-1, 1] to the starting point in each chain.
            * advi+adapt_diag : Run ADVI and then adapt the resulting diagonal
              mass matrix based on the sample variance of the tuning samples.
            * advi+adapt_diag_grad : Run ADVI and then adapt the resulting
              diagonal mass matrix based on the variance of the gradients
              during tuning. This is **experimental** and might be removed
              in a future release.
            * advi : Run ADVI to estimate posterior mean and diagonal mass
              matrix.
            * advi_map: Initialize ADVI with MAP and use MAP as starting point.
            * map : Use the MAP as starting point. This is discouraged.
            * nuts : Run NUTS and estimate posterior mean and mass matrix from
              the trace.
        n_init : int
            Number of iterations of initializer
            If 'ADVI', number of iterations, if 'nuts', number of draws.
        start : dict, or array of dict
            Starting point in parameter space (or partial point)
            Defaults to trace.point(-1)) if there is a trace provided and
            model.test_point if not (defaults to empty dict). Initialization
            methods for NUTS (see `init` keyword) can overwrite the default.
        trace : backend, list, or MultiTrace
            This should be a backend instance, a list of variables to track,
            or a MultiTrace object with past values. If a MultiTrace object
            is given, it must contain samples for the chain number `chain`.
            If None or a list of variables, the NDArray backend is used.
            Passing either "text" or "sqlite" is taken as a shortcut to set
            up the corresponding backend (with "mcmc" used as the base
            name).
        chain_idx : int
            Chain number used to store sample in backend. If `chains` is
            greater than one, chain numbers will start here.
        chains : int
            The number of chains to sample. Running independent chains
            is important for some convergence statistics and can also
            reveal multiple modes in the posterior. If `None`, then set to
            either `njobs` or 2, whichever is larger.
        njobs : int
            The number of chains to run in parallel. If `None`, set to the
            number of CPUs in the system, but at most 4. Keep in mind that
            some chains might themselves be multithreaded via openmp or
            BLAS. In those cases it might be faster to set this to one.
        tune : int
            Number of iterations to tune, if applicable (defaults to 500).
            Samplers adjust the step sizes, scalings or similar during
            tuning. These samples will be drawn in addition to samples
            and discarded unless discard_tuned_samples is set to True.
        nuts_kwargs : dict
            Options for the NUTS sampler. See the docstring of NUTS
            for a complete list of options. Common options are

            * target_accept: float in [0, 1]. The step size is tuned such
              that we approximate this acceptance rate. Higher values like 0.9
              or 0.95 often work better for problematic posteriors.
            * max_treedepth: The maximum depth of the trajectory tree.
            * step_scale: float, default 0.25
              The initial guess for the step size scaled down by `1/n**(1/4)`.

            If you want to pass options to other step methods, please use
            `step_kwargs`.
        step_kwargs : dict
            Options for step methods. Keys are the lower case names of
            the step method, values are dicts of keyword arguments.
            You can find a full list of arguments in the docstring of
            the step methods. If you want to pass arguments only to nuts,
            you can use `nuts_kwargs`.
        progressbar : bool
            Whether or not to display a progress bar in the command line. The
            bar shows the percentage of completion, the sampling speed in
            samples per second (SPS), and the estimated remaining time until
            completion ("expected time of arrival"; ETA).
        model : Model (optional if in `with` context)
        random_seed : int or list of ints
            A list is accepted if `njobs` is greater than one.
        live_plot : bool
            Flag for live plotting the trace while sampling
        live_plot_kwargs : dict
            Options for traceplot. Example: live_plot_kwargs={'varnames': ['x']}
        discard_tuned_samples : bool
            Whether to discard posterior samples of the tune interval.
        compute_convergence_checks : bool, default=True
            Whether to compute sampler statistics like gelman-rubin and
            effective_n.

        Returns
        -------
        trace : pymc3.backends.base.MultiTrace
            A `MultiTrace` object that contains the samples.

        Examples
        --------

        """
        raise NotImplemented

    def _object_func(self, y: np.ndarray, theta: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                     c: np.ndarray = None):
        """
        目标函数
        Parameters
        ----------
        theta
        a
        b
        c

        Returns
        -------

        """
        raise NotImplemented

    def _jac_theta(self, y: np.ndarray, theta: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                   c: np.ndarray = None):
        """
        返回雅克比矩阵，也就是一阶导数
        Returns
        -------

        """
        raise NotImplemented

    def _hessian_theta(self, y: np.ndarray, theta: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                       c: np.ndarray = None):
        """
        返回海森矩阵，也就是二阶导数
        Returns
        -------

        """
        raise NotImplemented


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
            # z = pm.Deterministic(name="z", var=a.repeat(self.user_count, axis=0) * (
            #         theta.repeat(self.item_count, axis=1) - b.repeat(self.user_count, axis=0)))
            # irt = pm.Deterministic(name="irt",
            #                        var=pm.math.sigmoid(z))
            irt = pm.Deterministic(name="irt", var=pm.math.sigmoid(theta * a - a * b))
            output = pm.Deterministic(name="output",
                                      var=as_tensor_variable(irt)[
                                          self.response_sequence['user_iloc'], self.response_sequence['item_iloc']])
            # observed = pm.Bernoulli('observed', p=irt, observed=self.response_matrix)
            observed = pm.Bernoulli('observed', p=output, observed=self.response_sequence["answer"].values)

            kwargs['discard_tuned_samples'] = False
            # kwargs['start'] = pm.find_MAP()

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
        # print('grd', grd)
        return grd

    def _hessian_theta(self, theta: np.ndarray, y: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                       c: np.ndarray = None):
        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b)

        # return np.sum(y_hat * (1 - y_hat) * self.D ** 2 * a * a, axis=1)
        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        tmp = self.D * self.D * a * y_hat * (1 - y_hat)
        np.where(np.isnan(y), 0, tmp)
        return np.dot(tmp, a.T)

    def estimate_theta(self, method='CG', tol=None, options=None, bounds=None, join=True, progressbar=True):
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

        if 'a' in self.item_vector.columns:
            a = self.item_vector.loc[:, 'a'].values.reshape(1, self.item_count)
        else:
            a = None

        b = self.item_vector.loc[:, 'b'].values.reshape(1, self.item_count)
        if 'c' in self.item_vector.columns:
            c = self.item_vector.loc[:, 'c'].values.reshape(1, self.item_count)
        else:
            c = None

        if method == "Newton-CG":
            hessian = self._hessian_theta
        else:
            hessian = None

        success = []

        self._es_res_theta = []
        if join:

            # 注意y可能有缺失值
            y = self.response_matrix.values
            theta = self.user_vector.loc[:, ['theta']].values.reshape(self.user_count, 1)

            res = minimize(self._object_func, x0=theta, args=(y, a, b, c), method=method, jac=self._jac_theta,
                           bounds=bounds, hess=hessian, options=options, tol=tol)

            self.user_vector.loc[:, ['theta']] = res.x

            # y_list.append(y)
            # theta_list.append(theta)
            success.append(res.success)
            self._es_res_theta.append(res)
        else:
            if progressbar:
                iter_rows = tqdm(self.response_matrix.iterrows(), total=len(self.response_matrix))
            else:
                iter_rows = self.response_matrix.iterrows()
            for index, row in iter_rows:
                # 注意y可能有缺失值
                y = row.values.reshape(1, len(row))
                theta = self.user_vector.loc[index, 'theta'].values.reshape(1, 1)

                res = minimize(self._object_func, x0=theta, args=(y, a, b, c), method=method, jac=self._jac_theta,
                               bounds=bounds, hess=hessian, options=options, tol=tol)
                self.user_vector.loc[index, 'theta'] = res.x[0]
                success.append(res.success)
                self._es_res_theta.append(res)

        return all(success)


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

            # z = pm.Deterministic(name="z", var=a.repeat(self.user_count, axis=0) * (
            #         theta.repeat(self.item_count, axis=1) - b.repeat(self.user_count, axis=0)))
            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))

            # irt = pm.Deterministic(name="irt",
            #                        var=(1 - c.repeat(self.user_count, axis=0)) * pm.math.sigmoid(z) + c.repeat(
            #                            self.user_count, axis=0))
            irt = pm.Deterministic(name="irt", var=(1 - c) * pm.math.sigmoid(theta * a - a * b) + c)
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
    # def __init__(self, k: int = 5, *args, **kwargs):
    #     super(MIrt2PL, self).__init__(*args, **kwargs)
    #     self.Q = Q.join(self.item_vector['iloc']).set_index('iloc').sort_index().values
    #     m, self.k = self.Q.shape
    #     self.Q = self.Q.reshape(self.k, m)
    #     assert m == self.item_count
    # self.k = k

    def estimate_mcmc(self, **kwargs):
        """
        参数说明参考 http://docs.pymc.io/api/inference.html#module-pymc3.sampling
        :param kwargs:
        :return:
        """
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta\sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            # theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, self.k))
            theta = pm.MvNormal("theta", mu=np.zeros(self.k), cov=np.identity(self.k),
                                shape=(self.user_count, self.k))

            a = pm.Lognormal("a", mu=0, tau=1, shape=(self.k, self.item_count))
            b = pm.Normal("b", mu=0, sd=1, shape=(1, self.item_count))
            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a * self.Q) - b.repeat(self.user_count, axis=0))
            # z = pm.Deterministic(name="z", var=pm.math.dot(theta, a) - b.repeat(self.user_count, axis=0))
            # irt = pm.Deterministic(name="irt", var=pm.math.sigmoid(z))
            irt = pm.Deterministic(name="irt", var=pm.math.sigmoid(pm.math.dot(theta, a) - b))

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
        # 注意本函数是按顺序求u和item的估计值，不是矩阵求解每个被试和每个项目的作答。这里不要用矩阵的dot
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
    R"""
        补偿型3参数多维irt模型
    """

    def estimate_mcmc(self, **kwargs):
        basic_model = pm.Model()
        with basic_model:
            # 我们假设 \theta \sim N(0, 1) ， a \sim lognormal(0, 1) （对数正态分布），b\sim N(0, 1) ， c\sim beta(2, 5)
            # theta (proficiency params) are sampled from a normal distribution
            # theta = pm.Normal("theta", mu=0, sd=1, shape=(self.user_count, self.k))
            theta = pm.MvNormal("theta", mu=np.zeros(self.k), cov=np.identity(self.k),
                                shape=(self.user_count, self.k))
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
        self.item_vector['c'] = self.trace['c'].mean(axis=0)[0, :]
        theta = pd.DataFrame(self.trace['theta'].mean(axis=0),
                             columns=['theta_%d' % i for i in range(self.k)])
        # a = pd.DataFrame(self.trace['a'].mean(axis=0).T*self.Q.T,
        a = pd.DataFrame(self.trace['a'].mean(axis=0).T,
                         columns=['a_%d' % i for i in range(self.k)])
        self.user_vector = self.user_vector.join(theta, on="iloc", how='left')
        self.item_vector = self.item_vector.join(a, on="iloc", how='left')


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
