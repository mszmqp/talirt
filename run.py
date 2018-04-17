#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/3/12 15:57
"""
import sys
import argparse
from scipy import optimize
import ibis
import pandas
import os
import pandas as pd
import numpy as np
import pymc3 as pm

sys.path.append("../")
sys.path.append("./")
sys.path.append("./talirt")
from model.metrics import Metric
from utils.data import split_data, data_info
from model.irt import UIrt2PL, UIrt3PL, MIrt2PL, MIrt3PL, MIrt2PLN, MIrt3PLN
import json


def load_logs(cache_file="logs.pickle", from_cache=True):
    # _sql = """
    #         select
    #         dd.student_id,
    #         dd.student_name,
    #         lq.lq_origin_id,
    #         dd.ips_answer_status
    #         from dmdb.dm_ips_stu_answer_detail dd
    #         join odata.ods_ips_tb_level_question lq on lq.lq_id=dd.ips_lq_id
    #         where
    #             dd.cla_year="2017"
    #             and dd.city_code="0571"
    #             and dd.term_name='秋季班'
    #             and dd.subject_name='数学'
    #             and dd.lev_name='提高班'
    #             and dd.grade_name="初中一年级"
    #             and lq.lq_library_id='5'
    #     """
    _sql = """
        select
                        
              sa.fk_student as user_id,
              sa.fk_question as item_id,
              case when sa.asw_first_status='错误' then 0 else 1  end  as answer
        from dwdb.dwd_stdy_ips_level_answ sa
        where
            sa.qst_type_status='客观题'
            and sa.asw_first_status in ('正确','错误')
            and sa.fk_year='2017'
            and sa.city_name = '成都'
            and sa.grd_name='小学六年级'
            and sa.term_name='暑期班'
            and sa.subj_name='数学'
            and sa.lev_name='尖子班'
            and sa.cl_name='课后测'
       
        """
    # _sql = """
    # select
    #     lq.exp as item_id,
    #     sa.student_id as user_id,
    #     sa.answerstatus as answer,
    #     sa.start_time,
    #     sa.city_code,
    #     c.grade
    # from app_bi.mkt_student_answer_zhenhu  sa
    # join odata.ods_mkt_level_question lq  on lq.id = sa.levelquestionid
    # join odata.ods_mkt_course c on c.id = lq.course_id
    # where c.grade='1' and sa.city_code='0311' and lq.course_id='f75f4fa840484445be3972fe739ed0aa'
    # """
    if from_cache:
        # print >> sys.stderr, "从缓存读取题目画像数据"
        print("从缓存读取题目画像数据", file=sys.stderr)
        return pandas.read_pickle(cache_file)
    # print("从impala读取题目画像数据", file=sys.stderr)
    print("从impala读取题目画像数据", file=sys.stderr)
    # 默认情况下会限制只返回10000条数据
    ibis.options.sql.default_limit = None
    impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')
    df_question = impala_client.sql(_sql).execute()
    df_question.to_pickle(cache_file)
    impala_client.close()
    print("count:", len(df_question), file=sys.stderr)
    return df_question


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
    parser.add_argument("-p", "--padding", dest="padding", action='store_true', default=False,
                        help=u"if padding")
    parser.add_argument("-r", "--run", dest="runner", default="",
                        help=u"")
    return parser


_model_class = {
    "UIrt2PL": UIrt2PL,
    "UIrt3PL": UIrt3PL,
    "MIrt2PL": MIrt2PL,
    "MIrt3PL": MIrt3PL,
    "MIrt2PLN": MIrt2PLN,
    "MIrt3PLN": MIrt3PLN
}


def main(options):
    df = load_logs(from_cache=False)
    # df.loc[:, 'answer'] = df['answer'] == 1

    # from talirt.data import DataBank
    from utils.data import padding
    # data_bank = DataBank(logs=df)
    # print(data_bank)

    from model.metrics import Metric
    # 0：未做
    # 1：正确
    # 2：错误
    # 3：超时
    # 答题结果一定要处理成0-1二元值，否则会出错
    # df.loc[:, 'answer'] = df['answer'] == 1
    # df.loc[df["answer"] != 1, "answer"] = 0

    # df_data = df
    # all_users = df['user_id'].unique()
    # df = df[df['user_id'].isin(all_users[-50:])]

    # 删掉作答少的题目
    # hehe = df_target.groupby('item_id').count().sort_values('answer')
    # drop_item = hehe[hehe['answer'] < 10].index
    # df_target = df_target.set_index('item_id').drop(drop_item)
    # len(df_target)
    # 数据切割成训练集和测试机
    df_target = df
    train_df, test_df = split_data(df_target)
    print("-" * 10, "train data info", "-" * 10)
    print(data_info(train_df))
    print("-" * 10, "test data info", "-" * 10)
    print(data_info(test_df))

    train_df.to_pickle("train_df.pickle")
    test_df.to_pickle("test_df.pickle")

    train_df = df_target
    test_df = df_target
    model, model_info = run(train_df, test_df, UIrt2PL, draws=150, tune=1000, njobs=1)

    print('\n' * 2)
    print("=" * 10 + model_info['model_name'] + "=" * 10)

    print('-' * 10 + 'train info' + "=" * 10)
    print(data_info(train_df))
    print("mse", model_info['train']['mse'])
    print("mae", model_info['train']['mae'])
    for threshold, score in model_info['train']['accuracy_score']:
        print("accuracy_score", threshold, score)
    # Metric.print_confusion_matrix(model_info['confusion_matrix'])

    print('-' * 10 + 'test info' + "=" * 10)
    print(data_info(train_df))
    print("mse", model_info['test']['mse'])
    print("mae", model_info['test']['mae'])
    for threshold, score in model_info['test']['accuracy_score']:
        print("accuracy_score", threshold, score)


def mapper(options):
    train_df = pandas.read_pickle("train_df.pickle")
    test_df = pandas.read_pickle("test_df.pickle")

    for line in options.input:
        line = line.strip()
        if not line:
            continue
        line = line.split('\t')
        # hadoop nlineinputformat 会多一列行号
        if len(line) == 4:
            line.pop(0)
        if not line or line[0] == '#':
            continue

        model, tune, njobs = line

        Model = _model_class[model]
        model, model_info = run(train_df, test_df, Model=Model, tune=int(tune), njobs=int(njobs))
        print(json.dumps(model_info))


def run(train_df, test_df, Model, draws=500, tune=1000, njobs=1):
    test_true = test_df['answer'].values
    train_true = train_df['answer'].values

    model = Model(response=train_df)

    model.estimate_mcmc(draws=draws, tune=tune, njobs=njobs, progressbar=True)
    test_proba = model.predict_proba(list(test_df['user_id'].values), list(test_df['item_id'].values))
    train_proba = model.predict_proba(list(train_df['user_id'].values), list(train_df['item_id'].values))

    model_info = {'model_name': model.name()}

    model_info['train'] = {
        'y_proba': [float(x) for x in train_proba],
        'y_true': [float(x) for x in train_true],
        # 'confusion_matrix': Metric.confusion_matrix(train_true, train_proba, threshold=0.5),
        'accuracy_score': Metric.accuracy_score_list(train_true, train_proba),
        'mae': Metric.mean_absolute_error(train_true, train_proba),
        'mse': Metric.mean_squared_error(train_true, train_proba),
    }
    model_info['test'] = {
        'y_proba': [float(x) for x in test_proba],
        'y_true': [float(x) for x in test_true],
        # 'confusion_matrix': Metric.confusion_matrix(test_true, test_proba, threshold=0.5),
        'accuracy_score': Metric.accuracy_score_list(test_true, test_proba),
        'mae': Metric.mean_absolute_error(test_true, test_proba),
        'mse': Metric.mean_squared_error(test_true, test_proba),
    }

    model_info['parameters'] = {
        'draws': draws,
        'tune': tune,
        'njobs': njobs,

    }
    return model, model_info


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
