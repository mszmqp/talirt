#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/5/5 09:48
"""
import sys
import argparse
import pandas as pd
import numpy as np
from talirt.model import irt
__version__ = 1.0


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
    return parser


def main(options):
    from talirt.model import irt
    import matplotlib.pyplot as plt
    users = np.random.randint(low=0, high=3, size=20)
    items = np.random.randint(low=0, high=5, size=20)
    answers = np.random.randint(low=0, high=2, size=20)
    response = pd.DataFrame({'user_id': users, 'item_id': items, 'answer': answers}).drop_duplicates(
        ['item_id', 'user_id'])
    model2 = irt.UIrt2PL(response, D=1.702)
    model2.set_abc(pd.DataFrame({'a': np.ones(5), 'b': [1,2,3,4,5]}), columns=['a', 'b'])

    res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': True})



def test_1():
    model = irt.UIrt2PL()
    items = pd.DataFrame({'a': np.ones(10), 'b': np.random.randint(1, 6, 10)})
    students = pd.DataFrame({'theta':np.random.randint(-1,8,5)})
    #print(items)
    model.set_abc(items, columns=['a', 'b'])
    model.set_theta(students)
    data=model.predict_proba_x(model.user_vector.index, model.item_vector.index)
    # pd.Series(data.flatten()).plot.hist()
    report = {
        '优化算法':[],
        '难度':[],
        '题目数':[],
        '迭代次数':[],
        '估计状态':[],
        '能力值':[],
        '目标函数值':[],
    }
    for diffculty in range(1,6):
        for count in range(1,11):

            response = pd.DataFrame({'user_id': [1] * count, 'item_id': np.arange(count), 'answer': [1] * count})
            model2 = UIrt2PL(response)
            model2.set_abc(pd.DataFrame({'a': np.ones(count), 'b': [diffculty]*count}))

            res=model2.estimate_theta(method='CG',options={'maxiter':20,'disp':False})

            # print(res.message)
            report['优化算法'].append('CG')
            report['难度'].append(diffculty)
            report['题目数'].append(count)
            report['迭代次数'].append(res.nit)
            report['估计状态'].append(res.success)
            report['能力值'].append(res.x[0])
            report['目标函数值'].append(res.fun)
            # print(res.success,res.x,res.nit,res.fun)

            res = model2.estimate_theta(method='Newton-CG', options={'maxiter': 20, 'disp': False})
            report['优化算法'].append('Newton-CG')
            report['难度'].append(diffculty)
            report['题目数'].append(count)
            report['迭代次数'].append(res.nit)
            report['估计状态'].append(res.success)
            report['能力值'].append(res.x[0])
            report['目标函数值'].append(res.fun)


    print(pd.DataFrame(report))



if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
