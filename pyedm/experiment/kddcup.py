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
import hmmlearn

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
    item_info = df_train.groupby('item_name').agg(
        {
            'Correct First Attempt': ['count', 'sum'],
            'Incorrects': ['sum', 'mean'],
            'Corrects': ['sum', 'mean'],
            'Problem View': ['sum', 'mean'],
            'Step Duration (sec)': ['sum', 'mean'],
            'Hints': ['sum', 'mean'],
            # 'Opportunity(SubSkills)': ['sum','mean'],
        })
    item_info.columns = ['_'.join(x) for x in item_info.columns.tolist()]
    # 拉普拉斯修正
    item_info['acc'] = (item_info['Correct First Attempt_sum'] + 1) / (item_info['Correct First Attempt_count'] + 2)

    # item_info['acc'] = item_info['Corrects_sum'] / (item_info['Incorrects_sum'] + item_info['Corrects_sum'])
    # item_info.loc[item_info['acc'].isna(), 'acc'] = item_info['Correct First Attempt_mean']
    # item_info.loc[item_info['Correct First Attempt_count'] < 5, 'acc'] = 0.5

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

    return df_train, df_test, item_info


def test_irt_bkt(models, df_train, df_test, item_info):
    item_info_arr = item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64)
    df_train_g = df_train.groupby(['knowledge', 'user'])
    df_eva = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        # for _, row in df_test.iterrows():
        if row['knowledge'] is np.nan:  # 空知识点暂时不处理
            continue
        # 要预测的题目 必须在训练数据中出现过。
        try:
            predict_item = item_info.loc[row['item_name']]
            predict_item_id = int(predict_item['item_id'])
        except KeyError:
            # 没找到对应的题目
            continue
        # 找到当前（知识点，用户）的训练数据（以前的作答序列）
        key = (row['knowledge'], row['user'])
        try:
            # cur_train = df_train.loc[key]
            cur_train = df_train_g.get_group(key)
            x = cur_train['answer'].values.astype(np.int32)  # 对应的 x

            train_items_id = cur_train['item_id'].values.astype(np.int32)  # 对应的 item_id
            # 获取 trace 编号，用于找到对应的模型
            trace = cur_train['trace']  # 对应的trace
            trace_index = trace.iloc[0]

        except KeyError:
            # 没找到对应的作答序列
            # if cur_train.empty:
            continue

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
        # sd = model.posterior_distributed(x)

        result = model.predict_next(x, predict_item_id, 'map')
        row['pred_prob'] = result[1]
        row['作答序列'] = ','.join([str(a) for a in x])
        row['难度序列'] = ','.join([str(a) for a in item_info_arr[train_items_id, 1].round(2)])
        row['viterbi'] = ','.join([str(a) for a in model.viterbi(x)])

        df_eva.append(row)
        # if predict_item_id == 71327 and row['user'] == '250ygnuvf':
        #     print('nimei', row)
        #     pass

        # break
        # if len(df_eva) > 100:
        #     break
    df_eva = pd.DataFrame(df_eva)

    df_eva['pred_label'] = df_eva['pred_prob']
    df_eva.loc[df_eva['pred_label'] >= 0.5, 'pred_label'] = 1
    df_eva.loc[df_eva['pred_label'] < 0.5, 'pred_label'] = 0
    df_eva['pred_result'] = df_eva['pred_label'] == df_eva['answer']
    y_true = df_eva['answer'].values
    y_prob = df_eva['pred_prob'].values

    metric(y_true, y_prob)
    return df_eva


def test_standard_bkt(models, df_train, df_test, item_info):
    # item_info_arr = item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64)
    df_train_g = df_train.groupby(['knowledge', 'user'])
    df_eva = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        # for _, row in df_test.iterrows():
        if row['knowledge'] is np.nan:  # 空知识点暂时不处理
            continue
        # 找到当前（知识点，用户）的训练数据（以前的作答序列）
        key = (row['knowledge'], row['user'])
        try:
            # cur_train = df_train.loc[key]
            cur_train = df_train_g.get_group(key)
            # 获取 trace 编号，用于找到对应的模型
            trace = cur_train['trace']  # 对应的trace
            trace_index = trace.iloc[0]
            # 之前的作答序列
            x = cur_train['answer'].values.astype(np.int32)  # 对应的 x
        except KeyError:
            # 没找到对应的作答序列 todo 首次作答的预测
            continue

        model = models[trace_index]
        # 没有训练成功
        if not model.success:
            continue
        # print('作答序列')
        # print(x)
        # print('题目难度序列')
        # print(item_info_arr[train_items_id, 1])
        # print('预测题目难度', item_info_arr[predict_item_id, 1].round(4))
        # print(len(item_info_arr))

        # print("能力分布")
        # sd = model.posterior_distributed(x)
        # print(sd[-1,].round(4))

        # result = model.predict_next(x, predict_item_id)
        result = model.predict_next(x, 'map')
        # print("下一题预测")
        # print(result)
        # print("=" * 50)
        row['pred_prob'] = result[1]
        df_eva.append(row)
        # break
        # if len(df_eva) > 100:
        #     break
    df_eva = pd.DataFrame(df_eva)

    df_eva['pred_label'] = df_eva['pred_prob']
    df_eva.loc[df_eva['pred_label'] > 0.5, 'pred_label'] = 1
    df_eva.loc[df_eva['pred_label'] <= 0.5, 'pred_label'] = 0
    df_eva['pred_result'] = df_eva['pred_label'] == df_eva['answer']
    y_true = df_eva['answer'].values
    y_prob = df_eva['pred_prob'].values

    metric(y_true, y_prob)
    return df_eva


def write_badcase(df_eva, df_train, df_test, item_info, filepath="irt_bkt_badcase.xlsx"):
    writer = pd.ExcelWriter(filepath)
    # df_train.to_excel(writer, 'train')
    df_eva.to_excel(writer, 'evaluation')
    df_test.to_excel(writer, 'test')
    # item_info.to_excel(writer, 'item_info')
    # df2.to_excel(writer,
    writer.save()
    writer.close()


def metric(y_true, y_prob):
    y_pred = y_prob.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    mae = metrics.mean_absolute_error(y_true, y_prob)
    mse = metrics.mean_squared_error(y_true, y_prob)
    acc = metrics.accuracy_score(y_true, y_pred)
    auc_score = metrics.roc_auc_score(y_true, y_prob)

    print("origin data %d acc:%.4f" % (y_true.shape[0], y_true.mean()))
    print('-' * 50)
    print('mae:%.4f' % mae, "mse:%.4f" % mse, 'acc:%.4f' % acc, 'auc_score:%.4f' % auc_score)
    print('-' * 50)
    print("confusion_matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print('-' * 50)
    print('classification_report')
    print(metrics.classification_report(y_true, y_pred))


def train_irt_bkt(df_train, item_info):
    # start_init = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    n_stat = 7
    start_init = np.array([1.0 / 7] * 7, dtype=np.float64)
    # assert start_init.sum() == 1
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

    th = TrainHelper(n_stat=7, model_type=2)
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
    success_num = np.asarray([m.success for m in models]).sum()
    print('total:', th.model_count, "success_num", success_num, file=sys.stderr)
    return models


def train_standard_bkt(df_train, item_info):
    # start_init = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    n_stat = 2
    start_init = np.array([1.0 / n_stat] * n_stat, dtype=np.float64)
    # assert start_init.sum() == 1
    start_lb = np.array([0] * n_stat, dtype=np.float64)
    start_ub = np.array([1] * n_stat, dtype=np.float64)

    transition_init = np.array([[0.4, 0.6], [0, 1]])
    transition_lb = np.array([[0, 0], [0, 1]]).astype(np.float64)
    transition_ub = np.array([[1, 1], [0, 1]]).astype(np.float64)
    emission_init = np.array([[0.8, 0.2], [0.2, 0.8]])

    emission_lb = np.array([[0.7, 0], [0, 0.7]]).astype(np.float64)
    emission_ub = np.array([[1, 0.3], [0.3, 1]]).astype(np.float64)

    th = TrainHelper(n_stat=n_stat, model_type=1)

    th.init(start=start_init, transition=transition_init, emission=emission_init)
    th.set_bound_start(start_lb, start_ub)
    th.set_bound_transition(transition_lb, transition_ub)
    th.set_bound_emission(emission_lb, emission_ub)

    # th.set_item_info(item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64))

    trace = df_train['trace'].values
    group = df_train['group'].values
    x = df_train['answer'].values
    # items = df_train['item_id'].values

    th.run(trace=trace.astype(np.int32),
           group=group.astype(np.int32),
           x=x.astype(np.int32))
    models = th.models
    # for m in models:
    #     print()
    #     print('=' * 50)
    #     m.show()
    print(th.model_count, file=sys.stderr)
    return models


def run_irt_bkt(train_data, test_data):
    df_train, df_test, item_info = preprocess(train_data, test_data)
    # df_train.fillna({'knowledge',})
    # 去掉空知识点的数据
    df_train = df_train.loc[~pd.isna(df_train['knowledge']), :]
    df_train.sort_values(['knowledge', 'user'], inplace=True)

    # 每个(知识点，用户)训练一个模型，所以 trace 用 'knowledge'+'user' 实现
    trace = value2id(df_train['knowledge'] + '_' + df_train['user'])
    # group = value2id(df_train['user'])
    df_train['trace'] = trace
    df_train['group'] = trace  # 每个trace下只有一个作答（观测）序列，这里只要保证每个trace的group是同一个值即可。

    irt_models = train_irt_bkt(df_train, item_info)
    df_eva = test_irt_bkt(irt_models, df_train, df_test, item_info)
    # write_badcase(df_eva, df_train, df_test, item_info)


def run_standard_bkt(train_data, test_data):
    df_train, df_test, item_info = preprocess(train_data, test_data)
    df_train.sort_values(['knowledge', 'user'], inplace=True)
    trace = value2id(df_train['knowledge'])
    group = value2id(df_train['user'])
    df_train['trace'] = trace
    df_train['group'] = group
    # df_train.set_index('knowledge', inplace=True)

    # df_test = df_test.join(df_train[['trace']], how='left', on='knowledge')

    models = train_standard_bkt(df_train, item_info)
    test_standard_bkt(models, df_train, df_test, item_info)


def main(options):
    kdd = KddCup2010('/Users/zhangzhenhu/Documents/开源数据/kddcup2010/')
    # test bridge_to_algebra_2006_2007
    print('*' * 50)
    print("bridge_to_algebra_2006_2007")
    print('*' * 50)

    print("=" * 20, 'irt bkt', '=' * 20)
    run_irt_bkt(kdd.ba67_train, kdd.ba67_test)

    print("=" * 20, 'standard bkt', '=' * 20)
    # run_standard_bkt(kdd.ba67_train, kdd.ba67_test)

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
