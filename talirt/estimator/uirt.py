# from cython cimport view
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, uniform
import random
from joblib import Parallel, delayed
import os
import time
# from talirt.utils.uirt_lib import sample_theta, uirt_item_jac_and_hessian, log_likelihood, uirt_item_jac
from talirt.utils import uirt_lib
from talirt.utils import trunk_split

_parallel = Parallel(n_jobs=os.cpu_count())


class Estimator:
    def __init__(self, model="2PL", max_iter=500):
        # self.theta_prior_distribution = theta_prior_distribution
        # self.Q = len(theta_prior_value.flatten())
        # self.theta_prior_value = theta_prior_value
        assert model in ["1PL", "2PL", "3PL"]
        self.D = 1
        self.model = model
        self.response = None
        self.a = None
        self.b = None
        self.c = None
        self.theta = None
        self.user_count = 0
        self.item_count = 0
        self._bound_a = (0.25, 5)
        self._bound_b = (-6, 6)
        self._bound_c = (0, 0.5)

        self._bound_theta = [(-6, 6)]
        self.max_iter = max_iter

        if self.model == '1PL':
            self._bounds = [self._bound_b]
        elif self.model == '2PL':
            self._bounds = [self._bound_a, self._bound_b]
        else:
            self._bounds = [self._bound_a, self._bound_b, self._bound_c]

        self._llh = None
        self.iter = 0

    def _pre_init(self, **kwargs):

        return True

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
        self._pre_init(**kwargs)


class MLE(Estimator):

    def estimate_theta(self, **kwargs):
        def object_fun(x, response):
            ret = -uirt_lib.log_likelihood(theta=x, slope=self.a, intercept=self.b, guess=self.c, response=response)
            # print('theta llh=%f' % ret, " theta=%f" % x[0], end="\n")
            return ret

        def gradient(x, response):
            ret = uirt_lib.uirt_theta_jac(theta=x, slope=self.a, intercept=self.b, guess=self.c, response=response)
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
            ret = -uirt_lib.log_likelihood(theta=theta, slope=a, intercept=b, guess=c, response=response)
            # print('item llh=%f' % ret, " a=%f" % x[0], "b=%f" % x[1], end="\n")
            return ret

        def gradient(x, theta, response):
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
            ret = uirt_lib.uirt_item_jac(theta=theta, slope=a, intercept=b, guess=c, response=response)[0, :]
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


class EM(Estimator):

    def estimate_join(self, **kwargs):
        for self.iter in range(self.max_iter):
            # s = time.time()
            self._exp_step()
            # e = time.time()
            # print('e-step cost', e - s)
            # s = time.time()
            self._max_step()
            # e = time.time()
            # print('m-step cost', e - s)

            if self._check_stop():
                break
        self.iter += 1
        return True

    def _check_stop(self):
        cur_llh = uirt_lib.log_likelihood(theta=self.theta, slope=self.a, intercept=self.b, guess=self.c, response=self.response)
        # print("-" * 20)
        # print('total lld', self.iter, cur_llh)
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
    def _pre_init(self, **kwargs):
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
        # use a normal approximation
        if theta_distribution == 'uniform':
            self.theta_prior_distribution = np.ones(self.Q) / self.Q
        elif theta_distribution == 'normal':
            norm_pdf = [norm.pdf(x) for x in self.theta_prior_value]
            normalizer = sum(norm_pdf)
            self.theta_prior_distribution = np.array([x / normalizer for x in norm_pdf])
        else:
            raise Exception('invalid theta prior distribution %s' % theta_distribution)
        #
        self.theta_posterior_distribution = np.zeros((self.user_count, self.Q))
        self.r = np.zeros((self.Q, self.item_count))
        self.w = np.zeros((self.Q, self.item_count))

    def _process_posterior(self, k):
        theta = np.asarray([self.theta_prior_value[k]] * self.user_count).flatten()
        theta_prob = self.theta_prior_distribution[k]
        # 每个学生独立的log似然值
        independent_user_lld = uirt_lib.log_likelihood_user(theta, self.a, self.b, self.c, self.response)
        # 乘上当theta值的先验概率,这是后验概率分布的分子
        return (k, independent_user_lld + np.log(theta_prob))

    def _update_posterior_distribution(self):

        def logsum(logp: np.ndarray):
            w = logp.max(axis=1)
            shape = (w.size, 1)
            w = w.reshape(shape)
            logSump = w + np.log(np.sum(np.exp(logp - w), axis=1)).reshape(shape)
            return logSump

        # todo 搞成并行,我擦，并行反而变慢
        # global _parallel
        # result = _parallel(delayed(self._process_posterior)(k) for k in range(self.Q))
        # for k,data in result:
        #     self.theta_posterior_distribution[:, k] = data

        for k in range(self.Q):
            theta_k = np.asarray([self.theta_prior_value[k]] * self.user_count).flatten()
            theta_k_prior_prob = self.theta_prior_distribution[k]
            #     每个学生独立的log似然值
            independent_user_lld = uirt_lib.log_likelihood_user(theta_k, self.a, self.b, self.c, self.response)
            # 乘上当theta值的先验概率,这是后验概率分布的分子
            self.theta_posterior_distribution[:, k] = independent_user_lld + np.log(theta_k_prior_prob)

        # 后验概率的分母不是很好求
        # 后验概率分布更新
        self.theta_posterior_distribution = np.exp(
            self.theta_posterior_distribution - logsum(self.theta_posterior_distribution))

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
            # todo 空值的处理
            self.r[k, :] = (theta_k * self.response).sum(axis=0).reshape(1, self.item_count)
            self.w[k, :] = (theta_k * (1 - self.response)).sum(axis=0).reshape(1, self.item_count)

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

            # 预测值
            # y_hat = self._prob(theta=theta, a=a, b=b, c=c)
            y_hat = uirt_lib.u3irt_matrix(theta=theta, slope=a, intercept=b, guess=c)

            # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
            obj = rjk.reshape(self.Q, 1) * np.log(y_hat) + wjk.reshape(self.Q, 1) * np.log(1 - y_hat)
            res = - np.sum(np.nan_to_num(obj, copy=False))
            # print('item llh=%f' % res, " a=%f" % x[0], "b=%f" % x[1], end="")
            return res

        def jac_func(x, rjk, wjk):

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

            grade_a = 0
            grade_b = 0
            for k in range(self.Q):
                theta = np.array([self.theta_prior_value[k]])
                # 预测值
                # y_hat = self._prob(theta=theta, a=a, b=b, c=c)
                y_hat = uirt_lib.u3irt_matrix(theta=theta, slope=a, intercept=b, guess=c)
                m = rjk[k] - (rjk[k] + wjk[k]) * y_hat[0][0]
                grade_a += m * self.D * theta[0]
                grade_b += m * self.D

            # print(" a=%f" % x[0], "b=%f" % x[1], 'g_a=%f' % -grade_a, 'g_b=%f' % -grade_b)

            if self.model == '1PL':
                return -np.array([grade_b])
            elif self.model == '2PL':
                return -np.array([grade_a, grade_b])
            else:
                # todo c的偏导数
                return -np.array([grade_a, grade_b, 0])

        for j in range(self.item_count):
            a = self.a[j]
            b = self.b[j]
            c = self.c[j]
            # y = self.response[:, j]
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
            # print("L-BFGS-B")
            res = minimize(object_func, x0=x0, args=(rjk, wjk), method='L-BFGS-B', bounds=self._bounds, jac=jac_func)

            # res3 = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='Newton-CG',jac=gradient)
            # print("CG")
            # res = minimize(negative_ln_liklihood, x0=x0, args=(rjk, wjk), method='CG', jac=gradient)
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

    def _pre_init(self):
        self.sample_count = 5000
        self.burn_in = 100
        self._t = np.ones((self.item_count, 2, 2))
        self._theta_sample = np.zeros((self.user_count, self.sample_count))

        # theta 的初始值
        v = self.response.sum(axis=1)
        self.theta = (v - v.mean()) / v.std()

    def _sample_theta(self, start, end):
        ret = {}
        for i in range(start, end):
            data = uirt_lib.sample_theta(
                theta=self.theta[i], a=self.a, b=self.b, c=self.c,
                response=self.response[i:i + 1, :],
                burn_in=self.burn_in,
                n=self.sample_count + self.burn_in)
            # self._theta_sample[i, :] = data
            ret[i] = data
        return ret

    def _exp_step(self):

        # for i in range(self.user_count):
        #     response = self.response[i:i + 1, :]
        #     theta = self.theta[i]
        #     x = sample_theta(theta=theta,a=self.a,b=self.b,c=self.c,response=response,burn_in=self.burn_in,
        #         n=self.sample_count + self.burn_in)
        #     self._theta_sample[i, :] = x[1]
        #     if i > 200:
        #         break

        # todo 改成并发
        global _parallel
        result = _parallel(delayed(self._sample_theta)(s, e) for s, e in trunk_split(self.user_count, _parallel.n_jobs))
        for job_dict in result:
            for index, data in job_dict.items():
                self._theta_sample[index, :] = data

        self.theta = self._theta_sample.mean(axis=1)

    def _max_step(self):

        for j in range(self.item_count):
            response = self.response[:, j:j + 1]
            s = np.zeros((2, 1))
            h = np.zeros((2, 2))
            a = self.a[j:j + 1]
            b = self.b[j:j + 1]
            c = self.c[j:j + 1]
            for i in range(self.sample_count):
                theta = self._theta_sample[:, i]
                _s, _h = uirt_lib.uirt_item_jac_and_hessian(theta=theta, slope=a, intercept=b, guess=c, response=response)
                s += _s[0, :].reshape((2, 1))
                h += _h[0, :, :]
            s = s / self.sample_count
            h = -h / self.sample_count
            e = 1.0 / (self.iter + 200)
            t = self._t[j, :, :] + e * (h - self._t[j, :, :])
            self._t[j, :, :] = t
            _next = np.array([a, b]) + e * np.dot(np.linalg.inv(t), s)
            self.a[j] = _next[0, 0]
            self.b[j] = _next[1, 0]
            # print('-----%d-----' % self._cur_iter)
            # print(next)


class MHMLE(MHRM):
    def _max_step(self):

        def object_fun(x, theta, response):
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
            ret = -uirt_lib.log_likelihood(theta=theta, slope=a, intercept=b, guess=c, response=response)
            print('item llh=%f' % ret, " a=%f" % x[0], "b=%f" % x[1], end="\n")
            return ret

        def gradient(x, theta, response):
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
            ret = uirt_lib.uirt_item_jac(theta=theta, slope=a, intercept=b, guess=c, response=response)[0, :]
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
    from talirt.model.simulator import Simulator
    from scipy import stats
    import pandas as pd
    from talirt.model.irt import UIrt2PL

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
    from talirt.data import social_life_feelings as  slf
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print('-----BockAitkinEM------')
    model = BockAitkinEM()
    model.fit(response=slf.response)
    model.estimate_join()
    print("a-mse", mean_squared_error(model.a, slf.a), 'a-mae', mean_absolute_error(model.a, slf.a))
    print("b-mse", mean_squared_error(model.b, slf.b), 'b-mae', mean_absolute_error(model.b, slf.b))

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
    quit()
    # em = BockAitkinEM(model="2PL", D=1)
    em = MHRM(model="2PL")
    # em = MHMLE(model="2PL", D=1)
    em.fit(response=slf.response, a=slf.a, b=slf.b)
    # em.a[:] = 1.0
    # em.b[:] = np.array([-1.5, -1, 0, 1, 1.5])
    s = time.time()
    em.estimate_join()
    print(em.a)
    print(em.b)

    print('cost', time.time() - s)
    print(em.theta)
    quit()
    # model = MCEM(model="1PL")
    model = MHRM(model="1PL")
    model.fit(response=response, max_iter=100)

    model.a[0, :] = 0.75
    model.estimate()
    print(model.a)
    print(model.b)
    print(model.c)
    from talirt.model.metrics import Metric

    # print('theta mse')
    # print(Metric.metric_mean_error(model.theta.flatten(), sim.user.values.flatten()))
    # print('a mse')
    # print(Metric.metric_mean_error(model.a.flatten(), sim.item['a'].values.flatten()))
    # print('b mse')
    # print(Metric.metric_mean_error(model.b.flatten(), sim.item['b'].values.flatten()))
