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
from pyedm.model.bkt import TrainHelper
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


def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    数据处理
    Parameters
    ----------
    df_train : pandas.DataFrame
        训练数据
    df_test : pandas.DataFrame
        测试数据

    Returns
    -------

    """
    # 计算题目难度，当前仅通过正确率，采用拉普拉斯修正计算正确率，作答次数少的题目不准确，todo 优化题目难度计算方法
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
    # groupby 之后，Dataframe 的 column 是m util index，不方便使用，这里转换一下
    item_info.columns = ['_'.join(x) for x in item_info.columns.tolist()]
    # 拉普拉斯修正 计算正确率
    item_info['acc'] = (item_info['Correct First Attempt_sum'] + 1) / (item_info['Correct First Attempt_count'] + 2)

    # item_info['acc'] = item_info['Corrects_sum'] / (item_info['Incorrects_sum'] + item_info['Corrects_sum'])
    # item_info.loc[item_info['acc'].isna(), 'acc'] = item_info['Correct First Attempt_mean']
    # item_info.loc[item_info['Correct First Attempt_count'] < 5, 'acc'] = 0.5
    # IRT中的题目难度，正确率映射到区间[0,5]
    item_info['difficulty'] = (1 - item_info['acc']) * 5
    # IRT中的区分度参数，全部固定值 1
    item_info['slop'] = 1.0
    # IRT中的猜测参数，全部固定为 0
    item_info['guess'] = 0
    # 题目整型数字ID，
    item_info['item_id'] = np.arange(item_info.shape[0], dtype=np.int32)
    # 把题目【整型id】关联到训练数据中
    df_train = df_train.join(item_info['item_id'], how='left', on='item_name')
    df_train.rename(columns={'Anon Student Id': 'user',
                             # 'KC(Default)': 'knowledge',  # algebra data
                             # 'KC(SubSkills)': 'knowledge',  # bridge_to_algebra data
                             'Problem Hierarchy': 'knowledge',
                             'Correct First Attempt': 'answer',
                             }, inplace=True)

    # 把题目信息关联到测试数据中
    df_test = df_test.join(item_info, how='left', on='item_name')

    df_test.rename(columns={'Anon Student Id': 'user',
                            # 'KC(Default)': 'knowledge',
                            # 'KC(SubSkills)': 'knowledge',  # bridge_to_algebra data
                            'Problem Hierarchy': 'knowledge',
                            'Correct First Attempt': 'answer',
                            }, inplace=True)

    # df_train.fillna({'knowledge': 'NULL', 'user': 'NULL'}, inplace=True)

    return df_train, df_test, item_info


def train_irt_bkt(df_train, item_info):
    """
    训练IRT-BKT模型
    Parameters
    ----------
    df_train : pandas.DataFrame
        训练数据
    item_info : pandas.DataFrame
        题目信息
    Returns
    -------

    """
    # start_init = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    # 隐状态设置为7个
    n_stat = 7
    # 初始概率矩阵，均匀分布
    start_init = np.array([1.0 / 7] * 7, dtype=np.float64)
    # assert start_init.sum() == 1
    # 初始概率矩阵的上下限约束
    start_lb = np.array([0] * n_stat, dtype=np.float64)
    start_ub = np.array([1] * n_stat, dtype=np.float64)
    # 转移矩阵的初始值
    transition_init = np.array([
        [0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)
    # 转移矩阵的下限约束
    transition_lb = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    # 转移矩阵的上限约束
    transition_ub = np.array([
        [1, .5, .5, .5, .5, .5, .5],
        [0, 1, .5, .5, .5, .5, .5],
        [0, 0, 1, .5, .5, .5, .5],
        [0, 0, 0, 1, .5, .5, .5],
        [0, 0, 0, 0, 1, .5, .5],
        [0, 0, 0, 0, 0, 1, .5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)
    # 训练器
    th = TrainHelper(n_stat=7, model_type=2)
    th.init(start=start_init, transition=transition_init)
    th.set_bound_start(start_lb, start_ub)
    th.set_bound_transition(transition_lb, transition_ub)
    #
    th.set_item_info(item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64))

    trace = df_train['trace'].values
    group = df_train['group'].values
    x = df_train['answer'].values
    items = df_train['item_id'].values

    th.fit(trace=trace.astype(np.int32),
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
    """
    训练标准BKT
    Parameters
    ----------
    df_train
    item_info

    Returns
    -------

    """
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

    th.fit(trace=trace.astype(np.int32),
           group=group.astype(np.int32),
           x=x.astype(np.int32))
    models = th.models
    # for m in models:
    #     print()
    #     print('=' * 50)
    #     m.show()
    print(th.model_count, file=sys.stderr)
    return models


def test_irt_bkt(models, df_train, df_test, item_info):
    """"""
    # 所有题目的参数信息
    item_info_arr = item_info[['slop', 'difficulty', 'guess']].values.astype(np.float64)
    # print(np.any(np.isnan(item_info_arr)))

    # 需要找到当前数据对应的前置观测序列，也就是当前(学生,知识点)下的前置作答数据，作为已知观测序列
    df_train_g = df_train.groupby(['knowledge', 'user'])
    df_eva = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        # for _, row in df_test.iterrows():
        if row['knowledge'] is np.nan:  # 空知识点暂时不处理
            continue
        # 要预测的题目 必须在训练数据中出现过，否则没有题目的信息
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
        model.set_obs_items(train_items_id)

        # print('作答序列')
        # print(x)
        # print('题目难度序列')
        # print(item_info_arr[train_items_id, 1])
        # print('预测题目难度', item_info_arr[predict_item_id, 1].round(4))
        # print(len(item_info_arr))

        # print("能力分布")
        # sd = model.posterior_distributed(x)

        result = model.predict_next(x, predict_item_id, 'posterior')
        if result is None or np.isnan(result[1]):
            print(result)
            # print(x)
            print(predict_item_id)
            # print(row)
            model.debug()
            quit()

        # if _ > 2000:
        #     quit()
        row['pred_prob'] = result[1]
        row['作答序列'] = ','.join([str(a) for a in x])
        row['难度序列'] = ','.join([str(a) for a in item_info_arr[train_items_id, 1].round(2)])
        row['viterbi'] = ','.join([str(a) for a in model.viterbi(x)])

        df_eva.append(row)
        # break
        # if len(df_eva) > 100:
        #     break
    # quit()
    df_eva = pd.DataFrame(df_eva)

    df_eva['pred_label'] = df_eva['pred_prob']
    df_eva.loc[df_eva['pred_label'] >= 0.5, 'pred_label'] = 1
    df_eva.loc[df_eva['pred_label'] < 0.5, 'pred_label'] = 0
    df_eva['pred_result'] = df_eva['pred_label'] == df_eva['answer']
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

        result = model.predict_next(x, 'posterior')
        # 预测做正确的概率
        row['pred_prob'] = result[1]
        df_eva.append(row)
        # break
        # if len(df_eva) > 100:
        #     break
    df_eva = pd.DataFrame(df_eva)
    # 预测的作答结果
    df_eva['pred_label'] = df_eva['pred_prob']
    df_eva.loc[df_eva['pred_label'] > 0.5, 'pred_label'] = 1
    df_eva.loc[df_eva['pred_label'] <= 0.5, 'pred_label'] = 0
    # 预测状态
    df_eva['pred_result'] = df_eva['pred_label'] == df_eva['answer']
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
    """
    二分类结果简单评估指标计算
    Parameters
    ----------
    y_true
    y_prob

    Returns
    -------

    """
    y_pred = y_prob.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    mae = metrics.mean_absolute_error(y_true, y_prob)
    mse = metrics.mean_squared_error(y_true, y_prob)
    acc = metrics.accuracy_score(y_true, y_pred)
    auc_score = metrics.roc_auc_score(y_true, y_prob)

    print("origin data, length:%d acc:%.4f" % (y_true.shape[0], y_true.mean()))
    print('-' * 50)
    print('mae:%.4f' % mae, "mse:%.4f" % mse, "rmse:%.4f" % np.sqrt(mse),
          'acc:%.4f' % acc,
          'auc_score:%.4f' % auc_score)
    print('-' * 50)
    print("confusion_matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print('-' * 50)
    print('classification_report')
    print(metrics.classification_report(y_true, y_pred))
    return {
        'test size': len(y_true),
        'correct rate': y_true.mean(),
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'acc': acc,
        'auc_score': auc_score,
        'delta_acc': acc - y_true.mean(),

    }


def run_irt_bkt(df_train, df_test, df_item_info):
    """
    IRT变种BKT模型，每个（知识点，学生）训练一个模型
    Parameters
    ----------
    df_train
    df_test
    df_item_info

    Returns
    -------

    """
    # df_train.fillna({'knowledge',})
    # 去掉空知识点的数据
    # df_train = df_train.loc[~pd.isna(df_train['knowledge']), :]
    # 每个(知识点，用户)训练一个模型，所以 trace 用 'knowledge'+'user' 实现
    df_train['trace_name'] = df_train['knowledge'] + '_' + df_train['user']
    df_train['group_name'] = df_train['user']
    # 按照trace_name排序，保证相同trace_name的行在一起
    #
    df_train.sort_values(['trace_name', 'group_name'], inplace=True)
    trace = value2id(df_train['trace_name'])
    group = value2id(df_train['group_name'])
    df_train['trace'] = trace
    df_train['group'] = group

    irt_models = train_irt_bkt(df_train, df_item_info)
    df_eva = test_irt_bkt(irt_models, df_train, df_test, df_item_info)
    y_true = df_eva['answer'].values
    y_prob = df_eva['pred_prob'].values

    # print(np.any(np.isnan(y_true)), y_true)
    # print(np.any(np.isnan(y_prob)), y_prob)

    m = metric(y_true, y_prob)
    m['model'] = 'IRT-BKT'
    return m
    # write_badcase(df_eva, df_train, df_test, item_info)


def run_standard_bkt(df_train, df_test, df_item_info):
    """
    标准BKT模型，每个知识点训练一个模型
    Parameters
    ----------
    df_train
    df_test
    df_item_info

    Returns
    -------

    """

    # df_train, df_test, item_info = preprocess(train_data, test_data)
    df_train['trace_name'] = df_train['knowledge']
    df_train['group_name'] = df_train['user']
    # 按照 ('trace_name', 'group_name') 排序，保证相同trace_name的行在一起
    #
    df_train.sort_values(['trace_name', 'group_name'], inplace=True)

    trace = value2id(df_train['trace_name'])
    group = value2id(df_train['group_name'])
    df_train['trace'] = trace
    df_train['group'] = group

    models = train_standard_bkt(df_train, df_item_info)
    df_eva = test_standard_bkt(models, df_train, df_test, df_item_info)
    y_true = df_eva['answer'].values
    y_prob = df_eva['pred_prob'].values

    m = metric(y_true, y_prob)
    m['model'] = 'Standard-BKT'
    return m


def run_standard_bkt_individual(df_train, df_test, df_item_info):
    """
    标准BKT模型，但是每个（知识点，学生）训练一个模型
    Parameters
    ----------
    df_train
    df_test
    df_item_info

    Returns
    -------

    """
    df_train['trace_name'] = df_train['knowledge'] + '_' + df_train['user']
    df_train['group_name'] = df_train['user']
    # 按照 ('trace_name', 'group_name') 排序，保证相同trace_name的行在一起
    #
    df_train.sort_values(['trace_name', 'group_name'], inplace=True)

    trace = value2id(df_train['trace_name'])
    group = value2id(df_train['group_name'])
    df_train['trace'] = trace
    df_train['group'] = group

    models = train_standard_bkt(df_train, df_item_info)
    df_eva = test_standard_bkt(models, df_train, df_test, df_item_info)
    y_true = df_eva['answer'].values
    y_prob = df_eva['pred_prob'].values
    m = metric(y_true, y_prob)
    m['model'] = 'Individual-Standard-BKT'
    return m


def load_tal_data():
    """
    智能练习作答数据
    Returns
    -------

    """
    path = "/Users/zhangzhenhu/Documents/开源数据/ai_response_2019-02-12.pkl"
    df_data = pd.read_pickle(path)
    df_data.rename(columns={'user_id': 'user', 'item_id': 'item_name',
                            'knowledge_id': 'knowledge'}, inplace=True)

    df_data = df_data[['knowledge', 'user', 'item_name', 'answer', 'date_time']]
    df_data.sort_values(['knowledge', 'user', 'date_time'])
    df_g = df_data.groupby(['knowledge', 'user'])
    mask = []

    for key, df_u in df_g:
        n = len(df_u)
        if n < 6:
            mask.extend([True] * n)
            continue
        mask.extend([True] * (n - 1))
        mask.append(False)
    mask = np.asarray(mask)
    df_train = df_data.loc[mask]
    df_test = df_data.loc[~mask]

    # 计算题目难度，当前仅通过正确率，作答次数少的题目不准确，采用拉普拉斯修正计算正确率
    item_info = df_train.groupby('item_name').agg(
        {
            'answer': ['count', 'sum'],
            # 'Opportunity(SubSkills)': ['sum','mean'],
        })
    # groupby 之后，Dataframe 的 column 是m util index，不方便使用，这里转换一下
    item_info.columns = ['_'.join(x) for x in item_info.columns.tolist()]
    # 拉普拉斯修正 计算正确率
    item_info['acc'] = (item_info['answer_sum'] + 1) / (item_info['answer_count'] + 2)

    # IRT中的题目难度，正确率映射到区间[0,5]
    item_info['difficulty'] = (1 - item_info['acc']) * 5
    # IRT中的区分度参数，全部固定值 1
    item_info['slop'] = 1.0
    # IRT中的猜测参数，全部固定为 0
    item_info['guess'] = 0
    # 题目整型数字ID，
    item_info['item_id'] = np.arange(item_info.shape[0], dtype=np.int32)
    # 把题目【整型id】关联到训练数据中
    df_train = df_train.join(item_info['item_id'], how='left', on='item_name')

    # 把题目信息关联到测试数据中
    df_test = df_test.join(item_info, how='left', on='item_name')

    return df_train, df_test, item_info


def main(options):
    kdd = KddCup2010('/Users/zhangzhenhu/Documents/开源数据/kddcup2010/')
    report = []

    for train, test, label in [
        (kdd.ba67_train, kdd.ba67_test, "bridge_to_algebra_2006_2007"),
        (kdd.a56_train, kdd.a56_test, "algebra_2005_2006"),
        (kdd.a67_train, kdd.a67_test, "algebra_2006_2007"),

    ]:
        print('\n' * 2)
        print('*' * 50)
        print(label)
        print('*' * 50)

        df_train, df_test, df_item_info = preprocess(train, test)

        print("=" * 20, 'irt bkt', '=' * 20)
        metric1 = run_irt_bkt(df_train, df_test, df_item_info)
        metric1['data'] = label
        report.append(metric1)

        print("=" * 20, 'individual standard bkt', '=' * 20)
        metric2 = run_standard_bkt_individual(df_train, df_test, df_item_info)
        metric2['data'] = label
        report.append(metric2)

        print("=" * 20, 'standard bkt', '=' * 20)
        metric3 = run_standard_bkt(df_train, df_test, df_item_info)
        metric3['data'] = label
        report.append(metric3)

    # 智能练习数据
    df_train, df_test, df_item_info = load_tal_data()
    print('\n' * 2)
    print('*' * 50)
    print("智能练习数据")
    print('*' * 50)
    print("=" * 20, 'irt bkt', '=' * 20)
    metric1 = run_irt_bkt(df_train.copy(), df_test.copy(), df_item_info)
    metric1['data'] = "智能练习数据"
    report.append(metric1)

    print("=" * 20, 'individual standard bkt', '=' * 20)
    metric2 = run_standard_bkt_individual(df_train.copy(), df_test.copy(), df_item_info)
    metric2['data'] = "智能练习数据"
    report.append(metric2)

    print("=" * 20, 'standard bkt', '=' * 20)
    metric3 = run_standard_bkt(df_train, df_test, df_item_info)
    metric3['data'] = "智能练习数据"
    report.append(metric3)

    # 打印表格报告
    import pytablewriter
    print('\n' * 2)
    print('*' * 50)
    print("Final Report")
    print('*' * 50)
    columns = ['data', 'model', 'test size', 'correct rate',
               'rmse', 'auc_score', 'acc', 'delta_acc']

    lines = []
    for rp in report:
        line = []
        for cl in columns:
            value = rp[cl]
            if isinstance(value, float):
                value = "%.4f" % value
            elif isinstance(value, int):
                value = str(value)
            line.append(value)
        # print('\t'.join(line))
        lines.append(line)

    writer = pytablewriter.RstGridTableWriter()
    writer.table_name = "Final Report"
    writer.headers = columns
    writer.value_matrix = lines

    writer.write_table()


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
