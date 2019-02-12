#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2018/5/5 09:48
"""
import sys
import argparse
import pandas as pd
import numpy as np
from pyedm import irt
__version__ = 1.0
import ibis
ibis.options.sql.default_limit = None
impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')

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
    # from talirt.model import irt
    # import matplotlib.pyplot as plt
    # users = np.random.randint(low=0, high=3, size=20)
    # items = np.random.randint(low=0, high=5, size=20)
    # answers = np.random.randint(low=0, high=2, size=20)
    # response = pd.DataFrame({'user_id': users, 'item_id': items, 'answer': answers}).drop_duplicates(
    #     ['item_id', 'user_id'])
    # model2 = irt.UIrt2PL(response, D=1.702)
    # model2.set_abc(pd.DataFrame({'a': np.ones(5), 'b': [1,2,3,4,5]}), columns=['a', 'b'])
    #
    # res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': True})
    #
    test_2()

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
            model2 = irt.UIrt2PL(response)
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



def test_2():
    ## 从impala 拉取尖子班答题记录
    param = {"year": 2018, "city_name": "杭州", "grade_name": "初中一年级", "subject_name": "数学", 'term_name': "春季班",
             "level_name": ""}
    _sql = """
            select

                level_name,
                fk_course,
                c_sortorder,
                fk_student as stu_id,
                stu_name,
                fk_question as item_id,
                difficulty as b,
                1 as a,
                answer

            from
                (select 
                    lev_name as level_name,
                    fk_course,
                    c_sortorder,
                    fk_student,stu_name,
                    fk_question,
                    case when asw_first_status='错误' then 0 else 1  end   answer,
                    difficulty_id as difficulty

                from dwdb.dwd_stdy_ips_level_answ
               -- join dimdb.dim_level_question lq on lq.fk_question=sa.fk_question and sa.fk_courselevel=lq.lq_cl_id
                where
                     fk_year='%(year)s'
                    and city_name = '%(city_name)s'
                    and grd_name='%(grade_name)s'
                    and term_name='%(term_name)s'
                    and subj_name='%(subject_name)s'
                    and is_deleted='否'
                    and lev_name='尖子班'
                --group by lev_name,fk_course,c_sortorder,fk_student,stu_name,
                ) sa

            --group by
            --    level_name,fk_course,c_sortorder,fk_student,stu_name
    """ % param

    df_response = impala_client.sql(_sql).execute()

    # 验证下有没有重名的学生
    len(df_response['stu_name'].unique()) == len(df_response['stu_id'].unique())
    df_response['user_id'] = df_response['stu_name']
    model = irt.UIrt2PL(df_response, D=1.702)
    print(model.estimate_theta(method='Newton-CG', options={'disp': False}, join=False))
    print(model._es_res_theta)

if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
