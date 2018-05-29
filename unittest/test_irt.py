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
sys.path.append('./')
from talirt.model import irt
import matplotlib.pyplot as plt
import time
import logging

__version__ = 1.0
import kudu
import redis
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import traceback

class Storage:

    def __init__(self, options=None, logger=logging.getLogger()):
        self.logger = logger
        # self.bakend = bakend
        # if bakend == 'kudu':

        if options is not None:
            self.kudu_table = options.kudu_table
            self.es_index = options.es_index
            self.es_doctype = options.es_doctype
        else:
            self.kudu_table = 'tb_stu_answer_recommend'
            self.es_index = 'ips'
            self.es_doctype = 'tb_stu_answer'

        self.client_kudu = kudu.connect(
            host=['iz2ze5otogu0m1iff5wcttz', 'iz2ze5otogu0m1iff5wctvz', 'iz2ze5otogu0m1iff5wctsz'], port=7051)
        # elif bakend == 'es':
        self.client_es = Elasticsearch(
            ['http://elastic:Y0Vu72W5hIMTBiU@es-cn-mp90i4ycm0007a7ko.elasticsearch.aliyuncs.com:9200/'])
        logging.getLogger('elasticsearch').setLevel(logging.WARNING)

    def open_impala(self):
        import ibis
        ibis.options.sql.default_limit = None
        self.impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')

    def get_level_by_kudu(self, param,
                          columns=('sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status', 'data_source')):

        """
        kudu table schema
        http://iz2ze5otogu0m1iff5wctvz:8051/table?id=5313f2c5ed2e43c8b0914bd67815ed79
        """
        table = self.client_kudu.table(self.kudu_table)
        scanner = table.scanner()
        preds = [
            table['sa_year'] == param['year'],
            table['sa_city_code'] == param['city_id'],
            table['sa_grd_id'] == param['grade_id'],
            table['sa_term_id'] == param['term_id'],
            table['sa_subj_id'] == param['subject_id'],
            table['sa_lev_id'] == param['level_id'],
            table['sa_is_deleted'] == 0,
        ]
        scanner.add_predicates(preds)
        scanner.set_projected_column_names(columns)
        scanner.open()
        data = scanner.read_all_tuples()
        if len(data) == 0:
            self.logger.warning('kudu_read_empty')
            return None
        df_response = self._tuples_2_dataframe(data, columns)
        scanner.close()
        return df_response

    def get_stu_by_kudu(self, param,
                        columns=('sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status', 'data_source')):

        """
        # http://iz2ze5otogu0m1iff5wctvz:8051/table?id=5313f2c5ed2e43c8b0914bd67815ed79
        """
        table = self.client_kudu.table(self.kudu_table)
        scanner = table.scanner()
        preds = [
            table['sa_year'] == param['year'],
            table['sa_city_code'] == param['city_id'],
            table['sa_grd_id'] == param['grade_id'],
            table['sa_term_id'] == param['term_id'],
            table['sa_subj_id'] == param['subject_id'],
            table['sa_lev_id'] == param['level_id'],
            table['sa_stu_id'] == param['user_id'],
            table['sa_is_deleted'] == 0,
        ]
        scanner.add_predicates(preds)
        scanner.set_projected_column_names(columns)
        scanner.open()
        data = scanner.read_all_tuples()
        if len(data) == 0:
            self.logger.warning('kudu_read_empty')
            return None
        df_response = self._tuples_2_dataframe(data, columns)
        scanner.close()
        return df_response

    def get_sid_by_kudu(self, param,
                        columns=('sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status', 'data_source')):

        """
        # http://iz2ze5otogu0m1iff5wctvz:8051/table?id=5313f2c5ed2e43c8b0914bd67815ed79
        """
        table = self.client_kudu.table(self.kudu_table)
        scanner = table.scanner()
        preds = [
            table['sa_year'] == param['year'],
            table['sa_city_code'] == param['city_id'],
            table['sa_grd_id'] == param['grade_id'],
            table['sa_term_id'] == param['term_id'],
            table['sa_subj_id'] == param['subject_id'],
            table['sa_lev_id'] == param['level_id'],
            # table['sa_stu_id'] == param['user_id'],
            table['sa_id'] == param['sa_id'],
            table['sa_is_deleted'] == 0,
        ]
        scanner.add_predicates(preds)
        scanner.set_projected_column_names(columns)
        scanner.open()
        data = scanner.read_all_tuples()
        if len(data) == 0:
            self.logger.warning('kudu_read_empty')
            return None
        df_response = self._tuples_2_dataframe(data, columns)
        scanner.close()
        return df_response

    def get_stu_by_es(self, param):
        query = {
            'size': -1,
            'query':
                {
                    'bool': {
                        'filter': [
                            {'term': {'sa_year': param['year']}},
                            {'term': {'sa_city_code': param['city_id']}},
                            {'term': {'sa_grd_id': param['grade_id']}},
                            {'term': {'sa_term_id': param['term_id']}},
                            {'term': {'sa_subj_id': param['subject_id']}},
                            {'term': {'sa_lev_id': param['level_id']}},
                            {'term': {'sa_stu_id': param['user_id']}},
                        ]

                    }

                },
            '_source': ['sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status']
        }
        s = Search.from_dict(query).using(self.client_es).index(self.es_index).doc_type(self.es_doctype)
        data = [(record.sa_stu_id, record.sa_qst_id, record.sa_qst_difct, record.sa_answer_status,) for record in
                s.scan()]
        if len(data) == 0:
            self.logger.warning('es_read_empty')
            return None
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'difficulty', 'answer'])
        df.loc[df['answer'] != 1, 'answer'] = 0
        # df.loc[:, 'a'] = [1] * len(df)
        return df

    def get_level_by_es(self, param):
        query = {
            'size': -1,
            'query':
                {
                    'bool': {
                        'filter': [
                            {'term': {'sa_year': param['year']}},
                            {'term': {'sa_city_code': param['city_id']}},
                            {'term': {'sa_grd_id': param['grade_id']}},
                            {'term': {'sa_term_id': param['term_id']}},
                            {'term': {'sa_subj_id': param['subject_id']}},
                            {'term': {'sa_lev_id': param['level_id']}},
                        ]

                    }

                },
            '_source': ['sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status'],
            "sort": [
                {
                    "_doc": {
                        "order": "desc"
                    }
                }
            ],
        }
        s = Search.from_dict(query).using(self.client_es).index(self.es_index).doc_type(self.es_doctype)
        data = [(record.sa_stu_id, record.sa_qst_id, record.sa_qst_difct, record.sa_answer_status,) for record in
                s.scan()]

        if len(data) == 0:
            self.logger.warning('es_read_empty')
            return None
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'b', 'answer'])
        df.loc[df['answer'] != 1, 'answer'] = 0
        # df.loc[:, 'a'] = [1] * len(df)
        return df

    def get_level_by_es_test(self, param, size=-1):
        query = {
            'size': size,
            'query':
                {
                    'bool': {
                        'filter': [
                            {'term': {'sa_year': param['year']}},
                            {'term': {'sa_city_code': param['city_id']}},
                            {'term': {'sa_grd_id': param['grade_id']}},
                            {'term': {'sa_term_id': param['term_id']}},
                            {'term': {'sa_subj_id': param['subject_id']}},
                            {'term': {'sa_lev_id': param['level_id']}},
                        ]

                    }

                },
            '_source': ['sa_stu_id', 'sa_qst_id', 'sa_qst_difct', 'sa_answer_status'],
            "sort": [
                {
                    "_doc": {
                        "order": "desc"
                    }
                }
            ],
        }
        s = Search.from_dict(query).using(self.client_es).index(self.es_index).doc_type(self.es_doctype)
        return s.execute()

    @staticmethod
    def _tuples_2_dataframe(tuples, names):

        df = pd.DataFrame(tuples, columns=names)
        df = df.rename(
            columns={'sa_stu_id': 'user_id',
                     'sa_qst_id': 'item_id',
                     'sa_qst_difct': 'difficulty',
                     'sa_answer_status': 'answer',
                     'data_source': 'src',
                     })
        df.loc[df['answer'] != 1, 'answer'] = 0
        # df.loc[:, 'a'] = [1] * len(df)
        # df.loc[:, 'b'] = [2] * len(df)
        return df

    def get_level_response(self, param):
        try:

            df = self.get_level_by_kudu(param)
            if df is None or len(df) == 0:
                df = self.get_level_by_es(param)
            return df
        except Exception as e:
            self.logger.error("storage_error")
            msg = traceback.format_exc()
            self.logger.error(msg)
        return None

    def get_student_response(self, param):
        pass
        # if self.bakend == 'kudu':
        #     return self.get_stu_by_kudu(param)
        # elif self.bakend == 'es':
        #     return self.get_stu_by_es(param)

    def get_level_by_impala(self, param):
        """
        拉取当前班型下的所有答题记录
        Parameters
        ----------
        user_id

        Returns
        -------

        """
        _sql = """
                    select
                        sa.c_sortorder,
                        dim_student.old_stu_id as user_id,
                        sa.stu_name as user_name,
                        fk_question as item_id,
                        difficulty_id as difficulty,

                        case when asw_first_status_id =1 then 1 else 0  end   answer
                    from dwdb.dwd_stdy_ips_level_answ sa
                    join dimdb.dim_grade on sa.fk_grade=dim_grade.pk_grade
                    join dimdb.dim_term on sa.fk_term=dim_term.pk_term
                    join dimdb.dim_classlevel on sa.fk_classlevel=dim_classlevel.pk_classlevel
                    join dimdb.dim_subject on sa.fk_subject=dim_subject.pk_subject
                    join dimdb.dim_student on dim_student.pk_student=sa.fk_student

                    where
                            fk_year='%(year)s'
                            and fk_city = '%(city_id)s'
                            and dim_grade.grd_id='%(grade_id)s'
                            and dim_term.term_id='%(term_id)s'
                            and dim_subject.old_subj_id='%(subject_id)s'
                            and sa.is_deleted='否'
                            and dim_classlevel.old_lev_id='%(level_id)s'



            """ % param
        # print(_sql, file=sys.stderr)
        _level_response = self.impala_client.sql(_sql).execute()

        return _level_response

    def get_candidate_items(self, param):
        """
        获取候选题目信息
        Returns
        -------

        """
        _sql = """
                    select
                        que_id as item_id,
                        difficulty as difficulty,
                        1 as a
                    from dmdb.dm_qstportrait_qst
                    where
                        parent_id='0'
                        and	subject_id=2
                        and	grade_group_id=2
                        and qt_id in ('1','2','3','4','5')
                        and state=0
                        and kh_ids like '%%%(knowledge_id)s%%'
                        -- and subject_name = ''

            """ % param

        _candidate_items = self.impala_client.sql(_sql).execute().set_index('item_id')

        return _candidate_items

    def close(self):
        # self.client_es.close()
        self.client_kudu.close()
        # self.client_redis.close()
        self.impala_client.close()


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


def dump_reponse():
    store = Storage()
    param = {'year': '2018',
             'city_id': '0571',
             'grade_id': '7',
             'subject_id': 'ff80808127d77caa0127d7e10f1c00c4',
             'level_id': 'ff8080812fc298b5012fd3d3becb1248',
             'term_id': '1',
             'knowledge_id': "cb1471bd830c49c2b5ff8b833e3057bd",
             'user_id': '殷烨嵘',
             'user_response': {'user_id': [], 'item_id': [], 'answer': [], 'difficulty': []},
             'candidate_items': {'item_id': [], 'difficulty': []},
             }
    store.get_level_by_kudu(param).to_pickle('response.pk')


def main(options):
    # dump_reponse()
    response = pd.read_pickle("response.pk")

    model = irt.UIrt2PL()
    start = time.time()
    ret = model.fit(response=response, orient='records', estimate='user')
    end = time.time()
    print(ret, end - start)
    return

    # for x in [test_1(),
    # test_2(),
    # test_3(),
    # test_4(),
    # test_5(),
    # test_6(),]:
    #     print(x)


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
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


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
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


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

    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


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
    res = model.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


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

    model2 = irt.UIrt2PL(response, D=1.702, sequential=False)

    model2.set_abc(items)

    res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


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
    model2 = irt.UIrt2PL(response, D=1.702, sequential=False)

    model2.set_abc(items)

    res = model2.estimate_theta(method='CG', options={'maxiter': 20, 'disp': False})
    # ok = all(res.x < 20) and all(res.x > -20)
    # print('test:%s' % ok, 'result:%s' % res.x, 'iter:%d' % res.nit, 'estimate:%s' % res.success, "msg:%s" % msg)
    return res


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
