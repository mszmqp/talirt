#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2019/1/17 11:08
"""
import sys
import argparse
from pyedm.data.kddcup2010 import KddCup2010
from pyedm.model.bkt import BKTBatch
import numpy as np

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
    kdd = KddCup2010()
    df_a67_train = kdd.a67_train
    df_a67_test = kdd.a67_test
    # 计算题目难度，当前仅通过正确率，作答次数少的题目不准确，todo 优化题目难度计算方法
    item_info = df_a67_train[['item_name', 'Correct First Attempt']].groupby('item_name').agg(
        {'Correct First Attempt': ['mean', 'count']})

    item_info.columns = item_info.columns.droplevel(0)
    item_info.rename(columns={'mean': 'acc', 'count': 'do_count'}, inplace=True);
    # item_info.reset_index(inplace=True)
    item_info['difficulty'] = (1 - item_info['acc']) * 5
    item_info['slop'] = 1.0
    item_info['guess'] = 0
    item_info['item_id'] = np.arange(item_info.shape[0], dtype=np.int32)

    df_train = df_a67_train.join(item_info['item_id'], how='left', on='item_name')
    df_train.rename(columns={'Anon Student Id': 'user',
                             'KC(Default)': 'knowledge',
                             'Correct First Attempt': 'answer',
                             }, inplace=True)

    start_init = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    # assert start_init.sum() == 1
    n_stat = 7
    start_lb = np.array([0] * n_stat, dtype=np.float64)
    start_ub = np.array([1] * n_stat, dtype=np.float64)
    transition_init = np.array([
        [0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    transition_lb = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    transition_ub = np.array([
        [1, .5, .5, .5, .5, .5, .5],
        [0, 1, .5, .5, .5, .5, .5],
        [0, 0, 1, .5, .5, .5, .5],
        [0, 0, 0, 1, .5, .5, .5],
        [0, 0, 0, 0, 1, .5, .5],
        [0, 0, 0, 0, 0, 1, .5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # bkt = BKTBatch(bkt_model='IRT', n_stats=7, start_init=start_init, transition_init=transition_init)
    # bkt.set_bound_start(start_lb, start_ub)
    # bkt.set_bound_transition(transition_lb, transition_ub)
    # bkt.set_item_info(item_info)
    # bkt.fit_batch(df_train, trace_by=['knowledge', 'user'])

    bkt = BKTBatch(bkt_model='standard')
    bkt.fit_batch(df_train)


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
