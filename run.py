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
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import HillClimbingEstimator
from catsim.stopping import MaxItemStopper
from catsim.cat import generate_item_bank
from catsim.simulation import Simulator
import pymc3 as pm

sys.path.append("../")
from utils.data import split_data


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
                sa.sa_stu_id as user_id,
                lq.lq_origin_id as item_id,
                sa.sa_answer_status as answer,
                lq.lq_qst_difct as difficult
            
            from odata.ods_ips_tb_stu_answer sa
            join odata.ods_ips_tb_level_question lq on lq.lq_id=sa.sa_lq_id
            where
                sa.sa_year="2017"
                and sa.sa_city_code="028"
                and sa.sa_term_id='3'
                and sa.sa_subj_id='ff80808127d77caa0127d7e10f1c00c4'
                and sa.sa_lev_id='ff80808145707302014582f9d9dc3658'
                and sa.sa_grd_id="7"
                and lq.lq_library_id='5'
                and sa.sa_is_fixup=0
                and sa.sa_answer_status in (1,2)
        """

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
    return parser


def main(options):
    df = load_logs(from_cache=True)
    # df.loc[:, 'answer'] = df['answer'] == 1

    # from talirt.data import DataBank
    from utils.data import padding
    # data_bank = DataBank(logs=df)
    # print(data_bank)
    from .model.irt import UIrt2PL
    # 0：未做
    # 1：正确
    # 2：错误
    # 3：超时
    # 答题结果一定要处理成0-1二元值，否则会出错
    df.loc[:, 'answer'] = df['answer'] == 1
    # df_data = df
    # all_users = df['user_id'].unique()
    # df = df[df['user_id'].isin(all_users[-50:])]

    df_target = df.sample(frac=1)

    if options.padding:
        df_target = padding(df_target, user_count=40, item_count=40)

    train_df, test_df = split_data(df_target)
    # count = len(df_target)
    # train = df_target.iloc[:int(count * 0.7)]
    # test = df_target.iloc[-int(count * 0.7):]
    # train = df_target
    # test = df_target
    print(len(df), len(train_df), len(test_df))
    print(test_df['answer'].value_counts())
    return

    irt0 = UIrt2PL(train_df)
    print(irt0, file=sys.stderr)
    irt0.estimate_mcmc(draws=150, tune=1000, njobs=2)
    # print(irt0.trace['theta'].shape)
    # print(irt0.trace['a'].shape)
    # print(irt0.trace['b'].shape)
    # pm.df_summary()
    y0_proba = irt0.predict_proba(list(test_df['user_id'].values), list(test_df['item_id'].values))
    y0_true = test_df['answer'].values
    irt0.metric_mean_error(y0_true, y0_proba)
    irt0.confusion_matrix(y0_true, y0_proba, threshold=0.7)
    irt0.accuracy_score(y0_true, y0_proba, threshold=0.7)
    irt0.classification_report(y0_true, y0_proba, threshold=0.7)
    # irt0.plot_prc(y0_true, y0_proba)
    return
    # v = df.groupby(by=['stu_id', 'item_id']).count().values
    # print("记录总数", len(df))
    # print("（学生-题目，次数）")
    # print(pd.value_counts(v[:, 0]))
    # stu_ids = df['stu_id'].unique()
    # item_ids = df['item_id'].unique()
    # print("学生数量", len(stu_ids))
    # print("题目数量", len(item_ids))

    # stu_info = pd.DataFrame(stu_ids, columns=['stu_id'])
    # stu_info.insert(1, 'theta', np.zeros(len(stu_ids)))

    # item_info = pd.DataFrame(item_ids, columns=['stu_id'])
    # item_info.insert(1, 'alpha', np.zeros(len(item_ids)))
    # item_info.insert(2, 'beta', np.zeros(len(item_ids)))
    # item_info.insert(3, 'c', np.zeros(len(item_ids)))

    # item_parameters = generate_item_bank(len(item_ids), itemtype="2PL")

    # initializer = RandomInitializer()
    # selector = MaxInfoSelector()
    # estimator = HillClimbingEstimator()
    # stopper = MaxItemStopper(20)

    # Simulator(item_parameters, len(stu_ids)).simulate(initializer, selector, estimator, stopper)


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
