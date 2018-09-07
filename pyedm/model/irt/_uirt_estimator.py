# from cython cimport view
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from joblib import Parallel, delayed
import os
import time
from . import _uirt_clib, _uirt_lib
from pyedm.utils import trunk_split
# import numexpr

_parallel = Parallel(n_jobs=os.cpu_count(), backend='threading')


class Estimator:
    def __init__(self, model="2PL", max_iter=500):
        assert model in ["1PL", "2PL", "3PL"]
        self.model = model
        self.response = None
        self.a = None
        self.b = None
        self.c = None
        self.theta = None
        self.user_count = 0
        self.item_count = 0
        self._bound_a = (0.25, 5.0)
        self._bound_b = (-6.0, 6.0)
        self._bound_c = (0.0, 0.5)
        self._bound_theta = [(-6.0, 6.0)]
        self.max_iter = max_iter

        if self.model == '1PL':
            self._bounds = [self._bound_b]
        elif self.model == '2PL':
            self._bounds = [self._bound_a, self._bound_b]
        else:
            self._bounds = [self._bound_a, self._bound_b, self._bound_c]

        self._llh = None
        self.iter = 0
        self.cost_time = None

    def fit(self, response: np.ndarray,
            a: np.ndarray = None, b: np.ndarray = None, c: np.ndarray = None,
            theta: np.ndarray = None, **kwargs):
        self.response = response
        self.user_count = response.shape[0]
        self.item_count = response.shape[1]

        if theta is None:
            self.theta = np.zeros(self.user_count)
        else:
            assert theta.size == self.user_count
            self.theta = theta.flatten()

        if self.model == "1PL" or a is None:
            self.a = np.ones(self.item_count)
        else:
            assert a.size == self.item_count
            self.a = a.flatten()

        if self.model != '3PL' or c is None:
            self.c = np.zeros(self.item_count)
        else:
            assert c.size == self.item_count
            self.c = c.flatten()

        if b is None:
            self.b = np.zeros(self.item_count)
        else:
            assert b.size == self.item_count
            self.b = b.flatten()
        self._llh = None
        self.iter = 0

    def _get_abc_from_x(self, x):
        if self.model == '1PL':
            a = np.array([1.0])
            b = x
            c = np.array([0.0])

        elif self.model == '2PL':
            a = x[0:1]
            b = x[1:2]
            c = np.array([0.0])
        else:
            a = x[0:1]
            b = x[1:2]
            c = x[2:3]
        return a, b, c

    def mse(self):
        predict = _uirt_lib.uirt(theta=self.theta, slope=self.a, intercept=self.b, guess=self.c)
        return np.sqrt(((self.response - predict) ** 2).mean())

    def accuracy(self):
        predict = _uirt_lib.uirt(theta=self.theta, slope=self.a, intercept=self.b, guess=self.c)
        return (np.round(predict) == self.response).mean()

    def predict(self):
        return _uirt_clib.uirt_matrix(theta=self.theta,
                                     slope=self.a,
                                     intercept=self.b,
                                     guess=self.c)


class MLE(Estimator):

    def estimate_theta(self, **kwargs):
        def object_fun(x, response):
            ret = -_uirt_clib.log_likelihood(response=response, theta=x, slope=self.a, intercept=self.b, guess=self.c)
            # print('theta llh=%f' % ret, " theta=%f" % x[0], end="\n")
            return ret

        def gradient(x, response):
            ret = _uirt_clib.uirt_theta_jac(response=response, theta=x, slope=self.a, intercept=self.b, guess=self.c)
            # print(" a=%f" % x[0], "theta=%f" % x[0], 'grade=%f' % -ret[0])
            return -ret

        for j in range(self.user_count):
            theta = self.theta[j]
            y = self.response[j:j + 1, :]
            x0 = np.array([theta])
            # res = minimize(object_fun, x0=x0, args=(self.theta, y), method='SLSQP', bounds=self._bounds)
            res = minimize(object_fun, x0=x0, args=(y,), method='L-BFGS-B', bounds=self._bound_theta,
                           jac=gradient)
            self.theta[j] = res.x[0]

    def estimate_item(self, **kwargs):
        def object_fun(x, theta, response):
            a, b, c = self._get_abc_from_x(x)
            ret = -_uirt_clib.log_likelihood(response=response, theta=theta, slope=a, intercept=b, guess=c)
            # print('item llh=%f' % ret, " a=%f" % x[0], "b=%f" % x[1], end="\n")
            return ret

        def gradient(x, theta, response):
            a, b, c = self._get_abc_from_x(x)
            ret = _uirt_clib.u2irt_item_jac(response=response, theta=theta, slope=a, intercept=b, guess=c)[0, :]
            if self.model == '1PL':
                return -ret[1:2]
            elif self.model == '2PL':
                return -ret[:2]
            else:
                return -ret

        for j in range(self.item_count):
            a = self.a[j]
            b = self.b[j]
            c = self.c[j]
            y = self.response[:, j:j + 1]
            if self.model == '1PL':
                x0 = np.array([b])
            elif self.model == '2PL':
                x0 = np.array([a, b])
            else:
                x0 = np.array([a, b, c])
            # res = minimize(negative_ln_liklihood, x0=x0, args=(y), method='BFGS',
            #                jac=gradient, tol=tol)
            # res = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='nelder-mead',
            #                options={'disp': False})
            # res = minimize(object_fun, x0=x0, args=(self.theta, y), method='SLSQP', bounds=self._bounds)
            #
            res = minimize(object_fun, x0=x0, args=(self.theta, y), method='L-BFGS-B', bounds=self._bounds,
                           jac=gradient)
            if self.model == '1PL':
                self.b[j] = res.x[0]
            elif self.model == '2PL':
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
            else:
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
                self.c[j] = res.x[2]


class MAP(MLE):

    def __init__(self, mu=0.0, sigma=1.0, model="2PL", max_iter=500):
        super(MAP, self).__init__(model=model, max_iter=max_iter)
        self.e_cost = 0
        self.m_cost = 0
        self.mu = mu
        self.sigma = sigma

    def estimate_theta(self, **kwargs):
        def object_fun(x, response):
            ret = -_uirt_clib.log_likelihood(response=response, theta=x, slope=self.a, intercept=self.b, guess=self.c)
            # print('theta llh=%f' % ret, " theta=%f" % x[0], end="\n")
            ret -= np.log(norm.pdf(x,loc=self.mu, scale=self.sigma)).sum()
            return ret

        def gradient(x, response):
            ret = _uirt_clib.uirt_theta_jac(response=response, theta=x, slope=self.a, intercept=self.b, guess=self.c)
            # scipy.stats.norm.pdf(x, mu, sigma)*(mu - x)/sigma**2
            # norm.pdf(x)*(-x)
            ret += (self.mu - x)/self.sigma**2
            # print(" a=%f" % x[0], "theta=%f" % x[0], 'grade=%f' % -ret[0])
            return -ret

        for j in range(self.user_count):
            theta = self.theta[j]
            y = self.response[j:j + 1, :]
            x0 = np.array([theta])
            # res = minimize(object_fun, x0=x0, args=(self.theta, y), method='SLSQP', bounds=self._bounds)
            res = minimize(object_fun, x0=x0, args=(y,), method='L-BFGS-B', bounds=self._bound_theta,
                           jac=gradient)
            self.theta[j] = res.x[0]


class EM(Estimator):

    def __init__(self, model="2PL", max_iter=500):
        super(EM, self).__init__(model=model, max_iter=max_iter)
        self.e_cost = 0
        self.m_cost = 0

    def estimate_join(self, **kwargs):
        self.e_cost = 0
        self.m_cost = 0
        s0 = time.time()
        for self.iter in range(self.max_iter):
            s1 = time.time()
            self._exp_step()
            s2 = time.time()
            self.e_cost += s2 - s1
            # print('e-step cost', e - s)
            # s = time.time()
            # self._max_step_1()
            self._max_step()
            s3 = time.time()
            # print('m-step cost', e - s)
            self.m_cost += s3 - s2
            if self._check_stop():
                break
        # e = time.time()
        self.iter += 1
        self.cost_time = time.time() - s0
        return True

    def _check_stop(self):
        cur_llh = _uirt_clib.log_likelihood(response=self.response, theta=self.theta,
                                           slope=self.a, intercept=self.b, guess=self.c)
        # print("-" * 20)
        # print('total ll', self.iter, cur_llh)
        # print(self.a)
        # print(self.b)
        if self._llh is None:
            self._llh = cur_llh
            return False
        if cur_llh >= self._llh and cur_llh - self._llh <= 1e-6:
            self._llh = cur_llh
            return True
        self._llh = cur_llh
        return False

    def _exp_step(self):
        raise NotImplemented

    def _max_step(self):
        raise NotImplemented


class BockAitkinEM(EM):
    def fit(self, response: np.ndarray,
            a: np.ndarray = None, b: np.ndarray = None, c: np.ndarray = None,
            theta: np.ndarray = None, **kwargs):
        super(BockAitkinEM, self).fit(response=response, a=a, b=b, c=c, theta=theta, **kwargs)
        """
        Bock-Aitkin EM 算法的初始化
        :param theta_min:
        :param theta_max:
        :param num_theta:
        :param dist:
        :return:
        """
        theta_min = kwargs.get('theta_min', -6)
        theta_max = kwargs.get('theta_max', 6)
        theta_num = kwargs.get('theta_num', 40)
        theta_distribution = kwargs.get('theta_distribution', 'normal')

        self.Q = theta_num
        # 从指定区间等距离取出点，作为先验分布的积分点
        self.theta_prior_value = np.linspace(theta_min, theta_max, num=theta_num)

        if self.Q != len(self.theta_prior_value):
            raise Exception('wrong number of inintial theta values')
        # 先验分布是均匀分布
        if theta_distribution == 'uniform':
            self.theta_prior_distribution = np.ones(self.Q) / self.Q
        # 先验分布是标准正态分布
        elif theta_distribution == 'normal':
            norm_pdf = [norm.pdf(x) for x in self.theta_prior_value]
            normalizer = sum(norm_pdf)
            self.theta_prior_distribution = np.array([x / normalizer for x in norm_pdf])
        else:
            raise Exception('invalid theta prior distribution %s' % theta_distribution)
        # theta后验分布初始值
        self.theta_posterior_distribution = np.zeros((self.user_count, self.Q))

        self.r = np.zeros((self.Q, self.item_count))
        self.w = np.zeros((self.Q, self.item_count))

    def _process_posterior(self, k):
        theta = np.asarray([self.theta_prior_value[k]] * self.user_count).flatten()
        theta_prob = self.theta_prior_distribution[k]
        # 每个学生独立的log似然值
        independent_user_lld = _uirt_clib.log_likelihood_user(response=self.response, theta=theta, slope=self.a,
                                                             intecept=self.b, guess=self.c)
        # 乘上当theta值的先验概率,这是后验概率分布的分子
        return k, independent_user_lld + np.log(theta_prob)

    def _update_posterior_distribution(self):
        """
        计算每个学生的后验概率分布
        self.theta_prior_distribution 是 theta的先验概率分布
        self.theta_posterior_distribution theta的后验概率分布
        Returns
        -------

        """
        def logsum(logp: np.ndarray):
            """
            后验概率的分母部分的计算。
            注意是加了对数的。
            """
            w = logp.max(axis=1)
            shape = (w.size, 1)
            w = w.reshape(shape)
            logSump = w + np.log(np.sum(np.exp(logp - w), axis=1)).reshape(shape)
            return logSump

        # self.Q 是能力值theta离散化的取值数量
        for k in range(self.Q):
            # 对于theta的每一个可能取值都进行

            # 假设每个学生的能力值都是theta_k
            theta_k = np.asarray([self.theta_prior_value[k]] * self.user_count).flatten()
            # theta取值为theta_k的先验概率
            theta_k_prior_prob = self.theta_prior_distribution[k]
            # 每个学生独立计算，各自作答数据的的log似然值。
            # 注意实际公式中是连乘符号，乘法会造成小数溢出，所以我们计算其对数值，把乘法转换成加法，注意最后还得换回去
            independent_user_lld = _uirt_clib.log_likelihood_user(response=self.response,
                                                                 theta=theta_k,
                                                                 slope=self.a,
                                                                 intercept=self.b, guess=self.c)
            # 乘上当theta值的先验概率,这是后验概率分布公式中的分子
            self.theta_posterior_distribution[:, k] = independent_user_lld + np.log(theta_k_prior_prob)
        # 上述循环，计算出了theta每个取值theta_k的分子部分
        # 后验概率的分母不是很好求
        # 后验概率分布更新，分子减分母，差值再求自然指数
        self.theta_posterior_distribution = np.exp(
            self.theta_posterior_distribution - logsum(self.theta_posterior_distribution))
        # 检查后验概率分布的概率和是否为1
        self.__check_theta_posterior()
        return 1

    def __check_theta_posterior(self):
        if np.any(np.abs((self.theta_posterior_distribution.sum(axis=1)) - 1) > 1e-6):
            raise Exception('theta posterior_distribution does not sum up to 1')

        if self.theta_posterior_distribution.shape != (self.user_count, self.Q):
            raise Exception(
                'theta posterior_distribution has wrong shape (%s,%s)' % self.theta_posterior_distribution.shape)

    def _expected_count(self):

        # 按照 k 循环计算
        # 一次性计算所有题目的
        for k in range(self.Q):
            theta_k = self.theta_posterior_distribution[:, k].reshape((self.user_count, 1))
            #  空值的处理
            self.r[k, :] = np.nan_to_num(theta_k * self.response, copy=False).sum(axis=0).reshape(1, self.item_count)
            self.w[k, :] = np.nan_to_num(theta_k * (1 - self.response)).sum(axis=0).reshape(1, self.item_count)

        return 1

    def _exp_step(self):
        # 更新能力值的后验分布
        self._update_posterior_distribution()
        # 计算作答期望
        self._expected_count()
        return True

    def _max_step(self):

        def object_func(x, rjk, wjk):
            """
            本函数一次计算一道题目的负的似然值
            Parameters
            ----------
            x
            rjk
            wjk

            Returns
            -------

            """
            theta = self.theta_prior_value.flatten()
            a, b, c = self._get_abc_from_x(x)

            # 预测值
            y_hat = _uirt_clib.uirt_matrix(theta=theta, slope=a, intercept=b, guess=c)
            obj = rjk.reshape(self.Q, 1) * np.log(y_hat) + wjk.reshape(self.Q, 1) * np.log(1 - y_hat)

            res = - np.sum(obj)
            # print('item llh=%f' % res, " a=%f" % x[0], "b=%f" % x[1], end="")
            return res

        def jac_func(x, rjk, wjk):
            """一道题目的导数"""
            a, b, c = self._get_abc_from_x(x)
            grade_a = 0
            grade_b = 0
            grade_c = 0
            # 在EM算法中，这里貌似不用处理空值？？
            for k in range(self.Q):
                theta = np.array([self.theta_prior_value[k]])
                # 预测值
                y_hat = _uirt_clib.uirt_matrix(theta=theta, slope=a, intercept=b, guess=c)
                m = rjk[k] - (rjk[k] + wjk[k]) * y_hat[0][0]
                grade_a += (1 - c[0]) * m * theta[0]
                grade_b += (1 - c[0]) * m
                # todo c的偏导数 c的偏导有点复杂啊！！！！！

            # print(" a=%f" % x[0], "b=%f" % x[1], 'g_a=%f' % -grade_a, 'g_b=%f' % -grade_b)

            if self.model == '1PL':
                return -np.array([grade_b])
            elif self.model == '2PL':
                return -np.array([grade_a, grade_b])
            else:
                return -np.array([grade_a, grade_b, grade_c])

        for j in range(self.item_count):
            a = self.a[j]
            b = self.b[j]
            c = self.c[j]
            # response = self.response[:, j]
            rjk = self.r[:, j]
            wjk = self.w[:, j]

            if self.model == '1PL':
                x0 = np.array([b])
            elif self.model == '2PL':
                x0 = np.array([a, b])
            else:
                x0 = np.array([a, b, c])
            # res1 = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='BFGS',
            #                jac=gradient)
            # res = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='nelder-mead',
            #                options={'disp': False})
            # res1 = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='SLSQP', bounds=bounds,
            #                options={'disp': False})
            res = minimize(object_func, x0=x0, args=(rjk, wjk), method='L-BFGS-B', bounds=self._bounds,
                           jac=jac_func)

            if self.model == '1PL':
                self.b[j] = res.x[0]
            elif self.model == '2PL':
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
            else:
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
                self.c[j] = res.x[2]
            # print(res)

        # 利用所有人的后验分布，更新先验分布
        # todo 这一步是否必须？？？如果加上就和论文中的结果有偏差
        # todo 后续搞定误差计算方法后，可以试试
        # self.theta_prior_distribution = self.theta_posterior_distribution.sum(
        #     axis=0) / self.theta_posterior_distribution.sum()

        # 利用所有人的后验分布计算得到每个人的能力值期望值，作为这个人的最终能力值
        self.theta = np.dot(self.theta_posterior_distribution, self.theta_prior_value.reshape(self.Q, 1)).flatten()


class MHRM(EM):
    def __init__(self, sample_count=1000, burn_in=10, model="2PL", max_iter=500):
        super(MHRM, self).__init__(model=model, max_iter=max_iter)
        self.e_cost = 0
        self.m_cost = 0
        self.sample_count = sample_count
        self.burn_in = burn_in
        self._t = None
        self._theta_sample = None

    def fit(self, response: np.ndarray,
            a: np.ndarray = None, b: np.ndarray = None, c: np.ndarray = None,
            theta: np.ndarray = None, **kwargs):
        super(MHRM, self).fit(response=response, a=a, b=b, c=c, theta=theta, **kwargs)

        if self.model == "2PL":
            self._t = np.zeros((self.item_count, 2, 2))
        elif self.model == '3PL':
            self._t = np.zeros((self.item_count, 3, 3, 3))

        self._theta_sample = np.zeros((self.user_count, self.sample_count))

        # theta 的初始值
        v = self.response.sum(axis=1)
        self.theta = (v - v.mean()) / v.std()

    def _sample_theta(self, start, end):
        ret = {}
        # np.array().astype(np.double)
        burn = self.burn_in + self.iter * 2
        for i in range(start, end):
            data = _uirt_clib.sample_theta(
                theta=self.theta[i], slope=self.a, intercept=self.b, guess=self.c,
                response=self.response[i:i + 1, :],
                burn_in=burn,
                n=self.sample_count + burn)
            # self._theta_sample[i, :] = data
            ret[i] = data
        return ret

    def _acceptance(self, theta):

        return _uirt_lib.log_likelihood(response=self.response, theta=theta,
                                       slope=self.a, intercept=self.b, guess=self.c
                                       ).sum(axis=1) + np.log(norm.pdf(theta, loc=1))

    def _mh(self, theta):

        delta = np.random.normal(loc=0, scale=1, size=self.user_count)
        v1 = self._acceptance(theta)
        v2 = self._acceptance(theta + delta)

        # 均匀分布的抽样
        sample = np.random.random(self.user_count)
        # print(iter,pre_theta,next_theta,r)
        # 这里的dot最慢了
        # x = np.diag(np.log(sample) < v2 - v1)
        x = np.log(sample) < v2 - v1
        # theta += x.dot(deta)
        # theta += np.einsum('ij,j->i', x, delta)
        theta += np.where(x, delta, 0)
        # theta += numexpr.evaluate("where(x,delta,0)")
        return theta

    def _exp_step(self):

        # theta_sample = [self.theta]
        # for _ in range(self.sample_count + self.burn_in):
        #     theta_sample.append(self._mh(theta_sample[-1]))
        #
        # self._theta_sample = np.column_stack(theta_sample[self.burn_in + 1:])

        # for i in range(self.user_count):
        #     response = self.response[i:i + 1, :]
        #     theta = self.theta[i]
        #     x = _uirt_clib.sample_theta(theta=theta,
        #                                slope=self.a, intercept=self.b, guess=self.c, response=response,
        #                                burn_in=self.burn_in, n=self.sample_count + self.burn_in)
        #     self._theta_sample[i, :] = x[1]
        #
        global _parallel
        result = _parallel(delayed(self._sample_theta)(s, e) for s, e in trunk_split(self.user_count, _parallel.n_jobs))
        for job_dict in result:
            for index, data in job_dict.items():
                self._theta_sample[index, :] = data

        self.theta = self._theta_sample.mean(axis=1)

    def _max_step_1(self):

        for j in range(self.item_count):
            response = self.response[:, j:j + 1]
            s = np.zeros((2, 1))
            h = np.zeros((2, 2))
            a = self.a[j:j + 1]
            b = self.b[j:j + 1]
            c = self.c[j:j + 1]
            for i in range(self.sample_count):
                theta = self._theta_sample[:, i]
                _s, _h = _uirt_clib.u2irt_item_jac_and_hessian(theta=theta, slope=a, intercept=b, guess=c,
                                                              response=response)
                s += _s[0, :].reshape((2, 1))
                h += _h[0, :, :]
            print('item', j)
            s = s / self.sample_count
            h = -h / self.sample_count
            print('s', s)
            print('h', h)
            gamma = 1.0 / (self.iter + 1.0)
            t = self._t[j, :, :] + gamma * (h - self._t[j, :, :])
            self._t[j, :, :] = t
            print(t)
            _next = np.array([a, b]) + gamma * np.dot(np.linalg.inv(t), s)
            print(_next)
            # print(gamma * np.dot(np.linalg.inv(t), s))
            # self.a[j] = max(min(_next[0, 0], self._bound_a[1]), self._bound_a[0])
            # self.b[j] = max(min(_next[1, 0], self._bound_b[1]), self._bound_b[0])
            # print('-----%d-----' % self._cur_iter)
            # print(next)

    def _max_step(self):

        jac = np.zeros((self.item_count, 2))
        hession = np.zeros((self.item_count, 2, 2))
        for i in range(self.sample_count):
            _jac, _hession = _uirt_lib.u2irt_item_jac_and_hessian(
                response=self.response,
                theta=self._theta_sample[:, i],
                slope=self.a,
                intercept=self.b,
                guess=self.c
            )
            jac += _jac
            hession -= _hession

        jac /= self.sample_count
        hession /= self.sample_count

        gamma = 1.0 / (self.iter + 1.0)

        self._t += gamma * (hession - self._t)
        Ginv = np.stack(np.linalg.inv(self._t[i, :, :]) for i in range(self.item_count))
        _next = gamma * np.einsum('ijk,ik->ij', Ginv, jac)  # s.shape =(n,p+1)
        # self.a[j] = max(min(_next[0, 0], self._bound_a[1]), self._bound_a[0])
        # self.b[j] = max(min(_next[1, 0], self._bound_b[1]), self._bound_b[0])
        self.a += _next[:, 0]
        self.b += _next[:, 1]


class MHMLE(MHRM):
    def _max_step(self):

        def object_fun(x, theta, response):
            a, b, c = self._get_abc_from_x(x)
            ret = -_uirt_clib.log_likelihood(response=response, theta=theta, slope=a, intercept=b, guess=c)
            print('item llh=%f' % ret, " a=%f" % x[0], "b=%f" % x[1], end="\n")
            return ret

        def gradient(x, theta, response):
            a, b, c = self._get_abc_from_x(x)
            ret = _uirt_clib.u2irt_item_jac(response=response, theta=theta, slope=a, intercept=b, guess=c)[0, :]
            # print(" a=%f" % x[0], "b=%f" % x[1], 'g_a=%f' % -ret[0], 'g_b=%f' % -ret[1])
            if self.model == '1PL':
                return -ret[1:2]
            elif self.model == '2PL':
                return -ret[:2]
            else:
                return -ret

        for j in range(self.item_count):
            a = self.a[j]
            b = self.b[j]
            c = self.c[j]
            y = self.response[:, j:j + 1]
            if self.model == '1PL':
                x0 = np.array([b])
            elif self.model == '2PL':
                x0 = np.array([a, b])
            else:
                x0 = np.array([a, b, c])
            # print('-' * 20)
            # print('item_id', j)
            # res = minimize(object_fun, x0=x0, args=(self.theta, y), method='SLSQP', bounds=self._bounds)
            #
            res = minimize(object_fun, x0=x0, args=(self.theta, y), method='L-BFGS-B', bounds=self._bounds,
                           jac=gradient)
            # print(res.success, res.message)
            if self.model == '1PL':
                self.b[j] = res.x[0]
            elif self.model == '2PL':
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
            else:
                self.a[j] = res.x[0]
                self.b[j] = res.x[1]
                self.c[j] = res.x[2]


if __name__ == "__main__":
    from pyedm.model.simulator import Simulator
    from scipy import stats
    import pandas as pd
    from pyedm.model.irt import UIrt2PL

    # df = pd.read_pickle("response.pk")
    # data = []
    # with open("../../data/lsat6.txt") as fh:
    #     for line in fh:
    #         line = [int(x) for x in line.strip().split()]
    #         data.append(line)
    # lsat6 = np.array(data, dtype=np.int16)
    # lsat6[lsat6 == 2] = 0
    # sim = Simulator(model="UIrt2PL", n_users=100, n_items=10)
    # response = sim.simulate().values
    # response[response >= 0.5] = 1
    # response[response < 0.5] = 0
    # for i in range(100):
    #     for j in range(10):
    #         response[i, j] = stats.bernoulli.rvs(response[i, j])

    # print(np.count_nonzero(response))
    # print('real theta')
    # print(sim.user)
    # sim.user
    # response = lsat6
    from pyedm.data import social_life_feelings as slf
    from sklearn.metrics import mean_squared_error, mean_absolute_error


    def test_baem():
        print('-----BockAitkinEM------')
        model = BockAitkinEM(model="3PL")

        # slf.response[0,0] = np.nan
        model.fit(response=slf.response)
        model.estimate_join()
        print('cost', model.cost_time, model.e_cost, model.m_cost)
        """
        论文的 z=a*theta+b
        我们的实现 z=a*(theta-b)
        二者的b值不一样，但是可以转换
        """
        tol = 1e-4
        print("a-mse", mean_squared_error(model.a, slf.a) < tol,
              'a-mae', mean_absolute_error(model.a, slf.a) < tol)

        print("b-mse", mean_squared_error(model.b, slf.b) < tol,
              'b-mae', mean_absolute_error(model.b, slf.b) < tol)
        print("theta-mse", mean_squared_error(model.theta, slf.theta) < tol,
              'theta-mae', mean_absolute_error(model.theta, slf.theta) < tol)
        print('mse', model.mse(), 'acc', model.accuracy())
        print('-----theta-----')
        print('estimate', model.theta.tolist())
        print('real', slf.theta.tolist())
        print('-----a-----')
        print('estimate', model.a.tolist())
        print('real', slf.a.tolist())
        print('-----b-----')
        print('estimate', model.b.tolist())
        print('real', slf.b.tolist())
        print('mse', model.mse(), 'acc', model.accuracy())
        print('llh', model._llh)


    def test_MLE():
        print('-----MLE theta------')
        model = MLE()
        model.fit(response=slf.response, a=slf.a, b=slf.b)
        model.estimate_theta()

        print(np.unique(model.theta))
        print("mse", mean_squared_error(model.theta, slf.theta))
        print("mae", mean_absolute_error(model.theta, slf.theta))

        print('-----MLE item------')
        model.fit(response=slf.response, theta=slf.theta)
        model.estimate_item()
        print('a', model.a)
        print('b', model.b)
        print('c', model.c)
        print("mse", mean_squared_error(model.a, slf.a))
        print("mae", mean_absolute_error(model.a, slf.a))


    def test_mhrm():
        print('-----MHRM------')
        em = MHRM(model="2PL", max_iter=80)
        # em = MHMLE(model="2PL", D=1)
        em.fit(response=slf.response)
        # em.a[:] = 1.0
        # em.b[:] = np.array([-1.5, -1, 0, 1, 1.5])
        s = time.time()
        em.estimate_join()
        print(em.theta.tolist())
        print(em.a)
        print(em.b)
        print('cost', em.cost_time, em.e_cost, em.m_cost)
        print('mse', em.mse(), 'acc', em.accuracy())
        print('llh', em._llh)


    def test_map(size=5):
        print('-----MAP theta------')
        model = MAP(mu=2.4,sigma=1)

        a = np.ones(size, dtype=np.float)
        b = np.array([1.0] * size, dtype=np.float)
        response = np.ones((1, size), dtype=np.float)
        # model.fit(response=slf.response, a=slf.a, b=slf.b)
        model.fit(response=response, a=a, b=-b)
        model.estimate_theta()
        print('size', size,'theta:%.5f'%model.theta[0],'acc', model.accuracy())

        # print(model.predict())

        # print(np.unique(model.theta))
        # print("mse", mean_squared_error(model.theta, slf.theta))
        # print("mae", mean_absolute_error(model.theta, slf.theta))


    test_map(1)
    test_map(2)
    test_map(3)
    test_map(4)
    test_map(5)
    test_map(6)
    test_map(8)
    # test_baem()
    # test_mhrm()
    quit()
