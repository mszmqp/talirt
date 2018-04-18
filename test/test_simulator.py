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
import os
sys.path.append("../")
sys.path.append("./")
sys.path.append("./talirt")
"""
要想在hadoop集群运行，需要修改文件 theano/configdefaults.py
增加下面两行
    1884 elif os.getenv('COMPILEDIR') is not None:
    1885     default_base_compiledir = os.getenv('COMPILEDIR')

然后重新把python打包上传到集群
"""
if os.getenv('map_input_file'):
    os.environ['COMPILEDIR'] = './.theano'


from model.simulator import Simulator
from model.irt import UIrt2PL, UIrt3PL, MIrt2PL, MIrt3PL, MIrt2PLN, MIrt3PLN
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

_model_class = {
    "UIrt2PL": UIrt2PL,
    "UIrt3PL": UIrt3PL,
    "MIrt2PL": MIrt2PL,
    "MIrt3PL": MIrt3PL,
    "MIrt2PLN": MIrt2PLN,
    "MIrt3PLN": MIrt3PLN
}


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
    model.estimate_mcmc(draws=draws, tune=tune, njobs=njobs, progressbar=True, chains=1)

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
    info = test(n_items=50, n_users=100, model="UIrt2PL", draws=500, tune=100, njobs=1)
    print(json2DataFrame([info]))


def json2DataFrame(inputs):
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
    for info in inputs:
        if isinstance(info, str):
            info = json.loads(info)
        for t in ['model_name', 'n_items', 'n_users', 'draws', 'tune', 'njobs']:
            hehe[t].append(info[t])
        for t in ['theta', 'a', 'b', 'c']:
            hehe[t + '-mae'] = info[t]['mae']
            hehe[t + '-mse'] = info[t]['mse']
            hehe[t + '-rmse'] = info[t]['rmse']

    return pandas.DataFrame(hehe)


def mapper(options):
    # train_df = pandas.read_pickle("simulator_df.pickle")
    for line in options.input:
        line = line.strip()
        if not line or '#' in line:
            continue
        line = line.split('\t')
        # hadoop nlineinputformat 会多一列行号
        if len(line) == 6:
            line.pop(0)
        if len(line) < 5 or line[0] == '#':
            continue

        model, n_items, n_users, tune, njobs = line

        model_info = test(model=model, n_users=int(n_users), n_items=int(n_items), tune=int(tune), njobs=int(njobs))
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
    elif options.runner == 'report':
        print(json2DataFrame(options.input))
        # reducer(options)
    else:
        main(options)
