#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2019/1/15 15:04
"""
import sys
import argparse
from pyedm.model.bkt import StandardBKT, IRTBKT
import pandas as pd
import numpy as np
import pyedm
import os
from hmmlearn.hmm import MultinomialHMM

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


def test():
    """
    https://github.com/myudelson/hmm-scalable
    对于空知识点的处理逻辑是：
    把空知识点看成独立的知识点，但是并不进行hmm训练，而是单纯统计其每个作答状态的比例，用这个比例作为预测值。
    """

    project_path = os.path.dirname(pyedm.__file__)
    data_path = os.path.join(project_path, "data", 'bkt')
    train_file = os.path.join(data_path, "train_sample.txt")
    result_file = os.path.join(data_path, "result_all.txt")
    # print("read data...", file=sys.stderr)
    print("=" * 10, "正确值", "=" * 10)

    print("=" * 10, "程序输出", "=" * 10)
    df_data = pd.read_csv(train_file, sep='\t', header=None, names=['answer', 'user', 'item', 'knowledge'])
    df_data.loc[:, 'answer'] -= 1

    bkt = BKT(
        # start_init=np.array([0.5, 0.5]),  # 0:不会 1:会
        # transition_init=np.array([[0.4, 0.6], [0, 1]]),  # 0-0:
        # emission_init=np.array([[0.8, 0.2], [0.2, 0.8]]),
    )
    bkt.fit(df_data, njobs=1)
    print('cost time', bkt.train_cost_time)
    # print("迭代次数",bkt.iter)
    for key, value in bkt.model.items():
        # print("--------", key, "-----------------")
        print("start_prob\n", value['start'])
        print("transmat\n", value['transition'])
        print("emissionprob\n", value['emission'])

    # for next_pro


def test_hmm_1():
    n_stat = 2
    start = np.asarray([0.5, 0.5])
    transition = np.asarray([[1.0, 0], [0.4, 0.6]])
    emmision = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    # 0,0, 1,0,0,0, 0,0,0,0.
    start_lb = np.asarray([0, 0], dtype=np.float64)
    transition_lb = np.asarray([[1, 0], [0, 0]], dtype=np.float64)
    emmision_lb = np.zeros((2, 2), dtype=np.float64)
    # 1,1, 1,1,1,1, 1,0.3,0.3,1
    start_up = np.ones(2, dtype=np.float64)
    transition_up = np.ones((2, 2), dtype=np.float64)
    emmision_up = np.asarray([[1, 0.3], [0.3, 1]], dtype=np.float64)

    x = np.asarray([2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32) - 1
    bkt = StandardBKT()
    bkt.init(start=start, transition=transition, emission=emmision)
    bkt.set_bounded_start(start_lb, start_up)
    bkt.set_bounded_transition(transition_lb, transition_up)
    bkt.set_bounded_emission(emmision_lb, emmision_up)
    bkt.estimate(x.flatten(), np.asarray([len(x)], dtype=np.int32))
    print('start')
    print(bkt.start)
    print('transition')
    print(bkt.transition)
    print("emission")
    print(bkt.emission)

    y = bkt.predict_next(np.asarray([], dtype=np.int32))
    print(y)


def test_hmm_2():
    df_train = pd.read_csv("/Users/zhangzhenhu/Documents/projects/hmm-scalable/cmake-build-debug/oneskill_train.txt",
                           sep='\t', header=None, names=['answer', 'user', 'item_name', 'knowledge'])

    df_test = pd.read_csv("/Users/zhangzhenhu/Documents/projects/hmm-scalable/cmake-build-debug/oneskill_test.txt",
                          sep='\t', header=None, names=['answer', 'user', 'item_name', 'knowledge'])

    # df_train.sort_values('user', inplace=True)
    x = df_train['answer'].values.astype(np.int32) - 1
    lengths = df_train['user'].value_counts().values.astype(np.int32)

    start = np.asarray([0.5, 0.5])
    transition = np.asarray([[1.0, 0], [0.4, 0.6]])
    emmision = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    # 0,0, 1,0,0,0, 0,0,0,0.
    start_lb = np.asarray([0, 0], dtype=np.float64)
    transition_lb = np.asarray([[1, 0], [0, 0]], dtype=np.float64)
    emmision_lb = np.zeros((2, 2), dtype=np.float64)
    # 1,1, 1,1,1,1, 1,0.3,0.3,1
    start_up = np.ones(2, dtype=np.float64)
    transition_up = np.ones((2, 2), dtype=np.float64)
    emmision_up = np.asarray([[1, 0.3], [0.3, 1]], dtype=np.float64)

    # x = np.asarray([2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    #                 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32) - 1
    bkt = StandardBKT()
    bkt.init(start=start, transition=transition, emission=emmision)
    bkt.set_bounded_start(start_lb, start_up)
    bkt.set_bounded_transition(transition_lb, transition_up)
    bkt.set_bounded_emission(emmision_lb, emmision_up)

    bkt.estimate(x.flatten(), lengths)

    # df_train.sort_values('user', inplace=True)
    x = df_test['answer'].values.astype(np.int32) - 1
    lengths = df_train['user'].value_counts().values.astype(np.int32)


def test_irt_bkt():
    from pyedm.model.bkt import IRTBKT

    project_path = os.path.dirname(pyedm.__file__)
    data_path = os.path.join(project_path, "data", 'bkt')
    df_data = pd.read_csv(os.path.join(data_path, "human_test.csv"))
    df_train = df_data.loc[df_data['label'] == 'train']
    df_test = df_data.loc[df_data['label'] == 'test']

    x = df_train['answer'].values.astype(np.int32)
    lengths = df_train['group'].value_counts().values.astype(np.int32)
    items = df_train['item_id'].values.astype(np.int32)
    item_info = df_data[['item_id', 'slop', 'difficulty', 'guess']].drop_duplicates('item_id')
    item_info.set_index('item_id', inplace=True)
    item_info_arr = item_info.values.astype(np.float64)
    x = np.asarray(
        [0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1], dtype=np.int32)
    difficulty = np.asarray(
        [2.42,2.35,2.19,0.94,2.5,1.43,1.92,0.83,1.47,0.59,1.25,0.42,1.88,1.33,1.88,2.19,1.92,1.54,2.33,1.33,1.56,0.94,1.18,0.59])
    item_info_arr = np.ones((len(difficulty), 3), dtype=np.float64)
    item_info_arr[:, 1] = difficulty
    item_info_arr[:, 2] = 0
    lengths = np.asarray([len(x)], dtype=np.int32)
    items = np.arange(len(difficulty), dtype=np.int32)

    model = IRTBKT(7)

    model.set_item_info(item_info_arr)
    model.estimate(x, lengths, items)
    print("start")
    print(model.start.round(3))
    print("transition")
    print(model.transition.round(3))

    # print("transition lower")
    # print(model.t)

    post = model.posterior_distributed(x)
    print("posterior_distributed")
    print(post.round(3))

    viterbi = model.viterbi(x)
    print("viterbi")
    print(viterbi.round(3))

    model.set_train_items(items)
    map_prob = model.predict_next(x, 6,'map')
    print('map_prob:',map_prob)
    print('viterbi_prob:',model.predict_next(x, 6,'viterbi'))

    # print("item info")
    # print(item_info)
    # print(item_info.values.astype(np.float64))


def main(options):
    # test_hmm_2()
    test_irt_bkt()


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
