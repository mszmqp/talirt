#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/4/13 14:18
"""
import sys
import argparse
from scipy import stats
import pandas
import numpy
import scipy
from scipy.special import expit
import math
import json
import sys

sys.path.append("./")
# lognorm的坑 https://book.douban.com/annotation/41953169/
from model.irt import UIrt2PL, UIrt3PL, MIrt2PL, MIrt3PL, MIrt2PLN, MIrt3PLN

_model_class = {
    "UIrt2PL": UIrt2PL,
    "UIrt3PL": UIrt3PL,
    "MIrt2PL": MIrt2PL,
    "MIrt3PL": MIrt3PL,
    "MIrt2PLN": MIrt2PLN,
    "MIrt3PLN": MIrt3PLN
}


class Simulator(object):

    def __init__(self, n_items: int = 10, n_users: int = 10,
                 model="UIrt2PL",
                 theta=stats.norm(0, 1),
                 a=stats.lognorm(s=1, scale=math.exp(0)),
                 b=stats.norm(0, 1),
                 c=stats.beta(5, 17),
                 k: int = 3,
                 item_index=None,
                 user_index=None):
        """

        Parameters
        ----------
        n_items
        n_users
        model:str
            model in ["U2PL","U3PL","M2PL","M3PL","M2PLN","M3PLN"]
        theta
        a
        b
        c
        k
        item_index
        user_index
        """
        self.dist_theta = theta
        self.dist_a = a
        self.dist_b = b
        self.dist_c = c
        self.n_items = n_items
        self.n_users = n_users
        self.k = k
        self.model = model
        if item_index is None:
            self.item_index = numpy.arange(self.n_items)
        else:
            self.item_index = item_index
        if user_index is None:
            self.user_index = numpy.arange(self.n_users)
        else:
            self.user_index = user_index
        self.user = None
        self.item = None

    def simulate(self):
        """

        Parameters
        ----------


        Returns
        -------
         : pandas.DataFrame

        """
        if self.model in ["MIrt2PL", "MIrt3PL"]:
            # self.dist_theta.rvs(self.n_users * self.k)
            self.user = pandas.DataFrame(
                {"theta_%d" % i: self.dist_theta.rvs(size=self.n_users) for i in range(self.k)}, index=self.user_index)

            a = {"a_%d" % i: self.dist_a.rvs(size=self.n_items) for i in range(self.k)}
            a['b'] = self.dist_b.rvs(size=self.n_items)
            a['c'] = self.dist_c.rvs(size=self.n_items)
            self.item = pandas.DataFrame(a, index=self.item_index)
        elif self.model in ["MIrt2PLN", "MIrt3PLN"]:
            # self.dist_theta.rvs(self.n_users * self.k)
            self.user = pandas.DataFrame(
                {"theta_%d" % i: self.dist_theta.rvs(size=self.n_users) for i in range(self.k)}, index=self.user_index)

            a = {"a_%d" % i: self.dist_a.rvs(size=self.n_items) for i in range(self.k)}
            a.update({"b_%d" % i: self.dist_b.rvs(size=self.n_items) for i in range(self.k)})
            a['c'] = self.dist_c.rvs(size=self.n_items)
            self.item = pandas.DataFrame(a, index=self.item_index)

        elif self.model in ['UIrt2PL', 'UIrt3PL']:
            self.user = pandas.DataFrame({
                'theta': self.dist_theta.rvs(size=self.n_users)
            }, index=self.user_index)

            self.item = pandas.DataFrame({
                'a': self.dist_a.rvs(size=self.n_items),
                'b': self.dist_b.rvs(size=self.n_items),
                'c': self.dist_c.rvs(size=self.n_items),
            }, index=self.item_index)

        if self.model == "UIrt2PL":
            self.item['c'] = numpy.zeros(self.n_items)
            return self.uirt(self.user, self.item)
        elif self.model == "UIrt3PL":
            return self.uirt(self.user, self.item)
        elif self.model == "MIrt2PL":
            self.item['c'] = numpy.zeros(self.n_items)
            return self.mirt(self.user, self.item)
        elif self.model == "MIrt3PL":
            return self.mirt(self.user, self.item)

    def uirt(self, user, item):
        theta = user['theta'].values
        a = item['a'].values
        b = item['b'].values
        c = item['c'].values

        theta = theta.reshape(len(theta), 1)
        a = a.reshape(1, len(a))
        b = b.reshape(1, len(b))
        c = c.reshape(1, len(c))

        z = a.repeat(theta.shape[0], axis=0) * (theta - b)
        c = c.repeat(theta.shape[0], axis=0)
        p = c + (1 - c) * expit(z)

        return pandas.DataFrame(p, columns=item.index, index=user.index)

    def mirt(self, user, item):
        user_count = len(user)
        item_count = len(item)
        theta = user.loc[:, ['theta_%d' % i for i in range(self.k)]].values  # shape=(user_count,k)
        a = item.loc[:, ['a_%d' % i for i in range(self.k)]].values.T  # shape = (k, item_count)
        b = item.loc[:, 'b'].values.reshape((1, item_count))
        c = item.loc[:, 'c'].values.reshape((1, item_count))
        b = b.repeat(user_count, axis=0)
        c = c.repeat(user_count, axis=0)
        z = numpy.dot(theta, a) - b
        p = c + (1 - c) * expit(z)
        return pandas.DataFrame(p, columns=item.index, index=user.index)


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
    parser.add_argument("-r", "--run", dest="runner", default="",
                        help=u"")
    return parser


def test(n_items=100, n_users=200, model="UIrt2PL", draws=500, tune=1000, njobs=1):
    import math
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    # from model import irt

    sim = Simulator(n_items=n_items, n_users=n_users, model=model)
    df = sim.simulate()
    user = []
    item = []
    answer = []
    for i in sim.user_index:
        for j in sim.item_index:
            user.append(i)
            item.append(j)
            answer.append(stats.bernoulli.rvs(df.loc[i, j], size=1)[0])
            # print(i, j, df.loc[i, j])
    real_user = sim.user
    real_item = sim.item

    response = pandas.DataFrame({'user_id': user, 'item_id': item, 'answer': answer})
    Model = _model_class[model]
    model = Model(response=response)
    model.estimate_mcmc(draws=draws, tune=tune, njobs=njobs, progressbar=True)

    estimate_user = model.user_vector
    estimate_item = model.item_vector

    model_info = {
        'draws': draws,
        'tune': tune,
        'njobs': njobs,
        'n_items': n_items,
        'n_users': n_users,
        'model_name': model.name(),

    }

    for param in ['a', 'b', 'c']:
        if param not in real_item.columns:
            continue
        mse = mean_absolute_error(real_item[param], estimate_item[param])
        mae = mean_squared_error(real_item[param], estimate_item[param])
        rmse = math.sqrt(mse)
        model_info[param] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,

        }

    param = 'theta'
    if param in real_user.columns:
        mse = mean_absolute_error(real_user[param], estimate_user[param])
        mae = mean_squared_error(real_user[param], estimate_user[param])
        rmse = math.sqrt(mse)
        model_info[param] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,

        }

    if 'theta_0' in real_user.columns:
        param = 'theta'
        real_value = real_user[["%s_%d" % (param, i) for i in range(model.k)]]
        estimate_value = estimate_user[["%s_%d" % (param, i) for i in range(model.k)]]
        mse = mean_absolute_error(real_value, estimate_value)
        mae = mean_squared_error(real_value, estimate_value)
        rmse = math.sqrt(mse)
        model_info[param] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,

        }
    if 'a_0' in real_item.columns:
        param = 'a'
        real_value = real_item[["%s_%d" % (param, i) for i in range(model.k)]]
        estimate_value = estimate_item[["%s_%d" % (param, i) for i in range(model.k)]]
        mse = mean_absolute_error(real_value, estimate_value)
        mae = mean_squared_error(real_value, estimate_value)
        rmse = math.sqrt(mse)
        model_info[param] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,

        }
    if 'b_0' in real_item.columns:
        param = 'b'
        real_value = real_item[["%s_%d" % (param, i) for i in range(model.k)]]
        estimate_value = estimate_item[["%s_%d" % (param, i) for i in range(model.k)]]
        mse = mean_absolute_error(real_value, estimate_value)
        mae = mean_squared_error(real_value, estimate_value)
        rmse = math.sqrt(mse)
        model_info[param] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,

        }
    return model_info


def main(options):
    hehe = {
        'model_name': [],
        'n_items': [],
        'n_users': [],
        'draws': [],
        'tune': [],
        'njobs': [],
        'theta-mae': [],
        'theta-mse': [],
        'theta-rmse': [],
        'a-mae': [],
        'a-mse': [],
        'a-rmse': [],
        'b-mae': [],
        'b-mse': [],
        'b-rmse': [],
        'c-mae': [],
        'c-mse': [],
        'c-rmse': [],
    }
    info = test(n_items=50, n_users=100, model="UIrt2PL", draws=500, tune=100, njobs=1)
    for t in ['model_name', 'n_items', 'n_users', 'draws', 'tune', 'njobs']:
        hehe[t].append(info[t])
    for t in ['theta', 'a', 'b', 'c']:
        hehe[t + '-mae'] = info[t]['mae']
        hehe[t + '-mse'] = info[t]['mse']
        hehe[t + '-rmse'] = info[t]['rmse']

    print(pandas.DataFrame(hehe))


def mapper(options):
    # train_df = pandas.read_pickle("simulator_df.pickle")
    for line in options.input:
        line = line.strip().split('\t')
        # hadoop nlineinputformat 会多一列行号
        if len(line) == 6:
            line.pop(0)
        model, n_items, n_users, tune,njobs = line

        model_info = test(n_users=int(n_users), n_items=int(n_items), tune=int(tune), njobs=int(1))
        print(json.dumps(model_info))


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin

    if options.runner == 'mapper':
        mapper(options)

    elif options.runner == 'reducer':
        pass
        # reducer(options)
    else:
        main(options)
