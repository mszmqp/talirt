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
import matplotlib.pyplot as plt

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
    # users = np.random.randint(low=0, high=3, size=20)
    # items = np.random.randint(low=0, high=5, size=20)
    # answers = np.random.randint(low=0, high=2, size=20)
    # response = pd.DataFrame({'user_id': users, 'item_id': items, 'answer': answers}).drop_duplicates(
    #     ['item_id', 'user_id'])
    # model2 = irt.UIrt2PL(response, D=1.702)
    # model2.set_abc(pd.DataFrame({'a': np.ones(5), 'b': [1, 2, 3, 4, 5]}), columns=['a', 'b'])
    #
    # res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': True})

    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()


def test_1():
    msg = """1个人，1道难度为1题目，答对，估计能力值"""
    items = pd.DataFrame({'item_id': ['item_1'], 'a': [1], 'b': [1]}).set_index('item_id')  # 题目区分度1，难度1
    students = pd.DataFrame({'user_id': ['user_1'], 'theta': [0]}).set_index('user_id')
    response = pd.DataFrame({'user_id': students.index,
                             'item_id': items.index,
                             'answer': [1]})

    model = irt.UIrt2PL(response=response)
    model.set_abc(items, columns=['a', 'b'])
    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok


def test_2():
    msg = """1个人，1道难度为1题目，答错，估计能力值"""
    items = pd.DataFrame({'item_id': ['item_1'], 'a': [1], 'b': [1]}).set_index('item_id')  # 题目区分度1，难度1
    students = pd.DataFrame({'user_id': ['user_1'],
                             'theta': [0]
                             }).set_index('user_id')
    response = pd.DataFrame({'user_id': students.index,
                             'item_id': items.index,
                             'answer': [0]})

    model = irt.UIrt2PL(response=response)
    # print(items)
    model.set_abc(items, columns=['a', 'b'])

    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok


def test_3():
    msg = """ 1个人，5道难度为5的题目，全答对，估计能力值"""
    items = pd.DataFrame({'item_id': np.arange(5),
                          'a': [1] * 5,
                          'b': [5] * 5,
                          }).set_index('item_id')  # 题目区分度1，难度1

    students = pd.DataFrame({'user_id': ['user_1'], 'theta': [0]}).set_index('user_id')

    response = pd.DataFrame({'user_id': list(students.index) * 5,
                             'item_id': items.index,
                             'answer': [1] * 5})

    model = irt.UIrt2PL(response=response)
    # print(items)
    model.set_abc(items, columns=['a', 'b'])
    # model.set_theta(students)
    # data = model.predict_proba_x(model.user_vector.index, model.item_vector.index)
    #
    # response = pd.DataFrame({'user_id': [1] * count, 'item_id': np.arange(count), 'answer': [1] * count})
    # model2 = UIrt2PL(response)
    # model2.set_abc(pd.DataFrame({'a': np.ones(count), 'b': [diffculty] * count}))

    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok

def test_4():
    msg = """ 1个人，5道难度为1的题目，全答错，估计能力值"""
    items = pd.DataFrame({'item_id': np.arange(5),
                          'a': [1] * 5,
                          'b': [1] * 5,
                          }).set_index('item_id')  # 题目区分度1，难度1

    students = pd.DataFrame({'user_id': ['user_1'], 'theta': [0]}).set_index('user_id')

    response = pd.DataFrame({'user_id': list(students.index) * 5,
                             'item_id': items.index,
                             'answer': [0] * 5})

    model = irt.UIrt2PL(response=response)
    # print(items)
    model.set_abc(items, columns=['a', 'b'])
    # model.set_theta(students)
    # data = model.predict_proba_x(model.user_vector.index, model.item_vector.index)
    #
    # response = pd.DataFrame({'user_id': [1] * count, 'item_id': np.arange(count), 'answer': [1] * count})
    # model2 = UIrt2PL(response)
    # model2.set_abc(pd.DataFrame({'a': np.ones(count), 'b': [diffculty] * count}))

    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok

def test_5():
    msg = """3个人，5道难度为1题目，全答对，估计能力值"""
    items = pd.DataFrame({'item_id': np.arange(5),
                          'a': [1] * 5,
                          'b': [1] * 5,
                          }).set_index('item_id')  # 题目区分度1，难度1

    students = pd.DataFrame({'user_id': np.arange(3),
                             'theta': [0] * 3,
                             }).set_index('user_id')

    response = pd.DataFrame(np.ones((3, 5)), index=students.index,
                            columns=items.index)

    # users = np.random.randint(low=0, high=3, size=20)
    # items = np.random.randint(low=0, high=5, size=20)
    # answers = np.random.randint(low=0, high=2, size=20)

    # response = pd.DataFrame({'user_id': users, 'item_id': items, 'answer': answers}).drop_duplicates(
    #     ['item_id', 'user_id'])
    model2 = irt.UIrt2PL(response, D=1.702, sequential=False)

    model2.set_abc(items)

    res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok


def test_6():
    msg = """3个人，5道难度为1题目，随机作答结果，有空值"""
    items = pd.DataFrame({'item_id': np.arange(5),
                          'a': [1] * 5,
                          'b': [1] * 5,
                          }).set_index('item_id')  # 题目区分度1，难度1

    students = pd.DataFrame({'user_id': np.arange(3),
                             'theta': [0] * 3,
                             }).set_index('user_id')

    response = pd.DataFrame(np.random.randint(low=0, high=2, size=(3, 5)), index=students.index,
                            columns=items.index)
    response.iloc[0, 1] = np.NAN
    response.iloc[1, 2] = np.NAN
    # print(response)
    # users = np.random.randint(low=0, high=3, size=20)
    # items = np.random.randint(low=0, high=5, size=20)
    # answers = np.random.randint(low=0, high=2, size=20)

    # response = pd.DataFrame({'user_id': users, 'item_id': items, 'answer': answers}).drop_duplicates(
    #     ['item_id', 'user_id'])
    model2 = irt.UIrt2PL(response, D=1.702, sequential=False)

    model2.set_abc(items)

    res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    ok = all(res.x < 20) and all(res.x > -20)
    print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return ok


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
