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
from pyedm.model.bkt import BKTBatch, TrainHelper
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

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


def value2id(s: pd.Series):
    df_left = pd.DataFrame({'value': s})
    us = s.unique()
    df_right = pd.DataFrame({'index': np.arange(len(us))}, index=us)
    # df_right.reset_index(inplace=True)
    return df_left.join(df_right, how='left', on='value')['index'].values


def preprocess(df_train, df_test):
    # 计算题目难度，当前仅通过正确率，作答次数少的题目不准确，todo 优化题目难度计算方法
    item_info = df_train[['item_name', 'Correct First Attempt']].groupby('item_name').agg(
        {'Correct First Attempt': ['mean', 'count']})

    item_info.columns = item_info.columns.droplevel(0)
    item_info.rename(columns={'mean': 'acc', 'count': 'do_count'}, inplace=True);
    # item_info.reset_index(inplace=True)
    item_info['difficulty'] = (1 - item_info['acc']) * 5
    item_info['slop'] = 1.0
    item_info['guess'] = 0
    item_info['item_id'] = np.arange(item_info.shape[0], dtype=np.int32)

    df_train = df_train.join(item_info['item_id'], how='left', on='item_name')
    df_train.rename(columns={'Anon Student Id': 'user',
                             'KC(Default)': 'knowledge',  # algebra data
                             'KC(SubSkills)': 'knowledge',  # bridge_to_algebra data
                             'Correct First Attempt': 'answer',
                             }, inplace=True)

    df_test = df_test.join(item_info, how='left', on='item_name')

    df_test.rename(columns={'Anon Student Id': 'user',
                            'KC(Default)': 'knowledge',
                            'KC(SubSkills)': 'knowledge',  # bridge_to_algebra data
                            'Correct First Attempt': 'answer',
                            }, inplace=True)

    # df_train.fillna({'knowledge': 'NULL', 'user': 'NULL'}, inplace=True)

    df_train = df_train.sort_values(['knowledge', 'user'])
    trace = value2id(df_train['knowledge'])
    group = value2id(df_train['user'])
    df_train['trace'] = trace
    df_train['group'] = group

    return df_train, df_test, item_info


def test_irt_bkt(models, df_train, df_test, item_info):
    item_info_arr = item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64)
    df_train_g = df_train.groupby(['knowledge', 'user'])
    df_eva = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        # for _, row in df_test.iterrows():
        if row['knowledge'] is np.nan:  # 空知识点暂时不处理
            continue
        try:
            predict_item = item_info.loc[row['item_name']]
        except KeyError:
            # 没找到对应的题目
            continue

        key = (row['knowledge'], row['user'])
        try:
            # cur_train = df_train.loc[key]
            cur_train = df_train_g.get_group(key)
        except KeyError:
            # 没找到对应的作答序列
            # if cur_train.empty:
            continue

        trace = cur_train['trace']  # 对应的trace
        trace_index = trace.iloc[0]
        # group = cur_train['group']  # 对应的trace
        x = cur_train['answer'].values.astype(np.int32)  # 对应的 x
        train_items_id = cur_train['item_id'].values.astype(np.int32)  # 对应的 item_id
        predict_item_id = int(predict_item['item_id'])
        # train_items_id.
        # print("前置序列长度", len(trace))
        # trace_index = trace.iloc[0]
        model = models[trace_index]
        # 没有训练成功
        if not model.success:
            continue
        model.set_item_info(item_info_arr)
        model.set_train_items(train_items_id)
        # print('作答序列')
        # print(x)
        # print('题目难度序列')
        # print(item_info_arr[train_items_id, 1])
        # print('预测题目难度', item_info_arr[predict_item_id, 1].round(4))
        # print(len(item_info_arr))

        # print("能力分布")
        sd = model.stat_distributed(x)
        # print(sd[-1,].round(4))

        result = model.predict_next(x, predict_item_id)
        # print("下一题预测")
        # print(result)
        # print("=" * 50)
        row['prob'] = result[1]
        df_eva.append(row)
        # break
        # if len(df_eva) > 100:
        #     break
    df_eva = pd.DataFrame(df_eva)
    y_true = df_eva['answer'].values
    y_prob = df_eva['prob'].values
    metric(y_true, y_prob)
    return df_eva


def write_badcase(df_eva, filepath="irt_bkt_badcase.xlsx"):
    writer = pd.ExcelWriter(filepath)
    df_eva.to_excel(writer, 'Sheet1')
    # df2.to_excel(writer,


def metric(y_true, y_prob):
    y_pred = y_prob.copy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    mae = metrics.mean_absolute_error(y_true, y_prob)
    mse = metrics.mean_squared_error(y_true, y_prob)
    acc = metrics.accuracy_score(y_true, y_pred)

    print("origin data acc:%4f" % y_true.mean())
    print('-' * 50)
    print('mae:%4f' % mae, "mse:%4f" % mse, 'acc:%4f' % acc)
    print('-' * 50)
    print("confusion_matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print('-' * 50)
    print('classification_report')
    print(metrics.classification_report(y_true, y_pred))


def train_irt_bkt(df_train, item_info):
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

    th = TrainHelper(n_stat=7, n_obs=2, model_type=2)
    th.init(start=start_init, transition=transition_init)
    th.set_bound_start(start_lb, start_ub)
    th.set_bound_transition(transition_lb, transition_ub)
    th.set_item_info(item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64))

    trace = df_train['trace'].values
    group = df_train['group'].values
    x = df_train['answer'].values
    items = df_train['item_id'].values

    th.run(trace=trace.astype(np.int32),
           group=group.astype(np.int32),
           x=x.astype(np.int32),
           items=items.astype(np.int32))
    models = th.models
    # for m in models:
    #     print()
    #     print('=' * 50)
    #     m.show()
    print(th.model_count, file=sys.stderr)
    return models


def run_irt_bkt(train_data, test_data):
    df_train, df_test, item_info = preprocess(train_data, test_data)
    irt_models = train_irt_bkt(df_train, item_info)
    df_eva = test_irt_bkt(irt_models, df_train, df_test, item_info)
    write_badcase(df_eva)


def run_standard_bkt(train_data, test_data):
    df_train, df_test, item_info = preprocess(train_data, test_data)
    irt_models = train_irt_bkt(df_train, item_info)
    test_irt_bkt(irt_models, df_train, df_test, item_info)


def main(options):
    kdd = KddCup2010('/Users/zhangzhenhu/Documents/开源数据/kddcup2010/')
    # test bridge_to_algebra_2006_2007
    print('=' * 50)
    print("bridge_to_algebra_2006_2007")
    print('=' * 50)
    run_irt_bkt(kdd.ba67_train, kdd.ba67_test)
    return
    # algebra_2006_2007_new_20100409
    print('=' * 50)
    print("algebra_2006_2007_new_20100409")
    print('=' * 50)
    run_irt_bkt(kdd.a67_train, kdd.a67_test)
    # algebra_2005_2006
    print('=' * 50)
    print("algebra_2005_2006")
    print('=' * 50)
    run_irt_bkt(kdd.a56_train, kdd.a56_test)


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
