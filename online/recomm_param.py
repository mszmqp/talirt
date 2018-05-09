# -*-coding:utf-8-*-

'''
Created on 2018年1月27日
@author:mengshuai@100tal.com
run cf_algorithm
'''

import os
import time

import sys
import json
# import matplotlib.pyplot as plt
import numpy as np
# import ibis
import pandas as pd
import random
# from math import sqrt
# from pandas import DataFrame
# from numpy import exp, shape, mat
# from numpy import linalg as la
# import shutil
# from model.matrix_factorization import *
# from evaluate.evaluate_ import *
# from model.col_fil import *
# from utils.data_process import *
from scipy.special import expit as sigmod
from scipy.optimize import minimize
# from tqdm import tqdm
import tempfile
import abc
import argparse

import logging

sys.path.append('./')
import run_recomm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64) or isinstance(obj, np.int32) or isinstance(obj, np.int8):
            return int(obj)
        if isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def main(options):
    param = {'year': '2018',
             'city_id': '0571',
             'grade_id': '7',
             'subject_id': 'ff80808127d77caa0127d7e10f1c00c4',
             'level_id': 'ff8080812fc298b5012fd3d3becb1248',
             'term_id': '1',
             'knowledge_id': "cb1471bd830c49c2b5ff8b833e3057bd",
             'stu_id': '殷烨嵘',
             'stu_response': {'user_id': [], 'item_id': [], 'answer': [], 'b': []},
             'candidate_items': {'item_id': [], 'b': []},
             }

    # pd.DataFrame.from_records([(3,'a'),(4,'h')])
    # pd.DataFrame.from_records([{'id':3,'xx':'a'},{'id':4,'xx':'h'}])
    # pd.DataFrame({'a':[1,4],'b':[3,6]})

    level_response = run_recomm.load_level_response(**param)
    # level_response.to_pickle('level_response.bin')
    # level_response = pd.read_pickle('level_response.bin')

    train_data = level_response.loc[level_response['c_sortorder'] <= 6, :]
    test_data = level_response.loc[level_response['c_sortorder'] >= 5, :]
    candidate_items = test_data.loc[:, ['item_id', 'b']].drop_duplicates('item_id')
    param['candidate_items'] = json.loads(candidate_items.to_json())

    for user_id in list(train_data.loc[:, 'user_id'].unique()):
        stu_response = run_recomm.load_stu_response(user_id, train_data)

        param['user_id'] = user_id
        param['stu_response'] = json.loads(stu_response.to_json())
        print(json.dumps(param))
    # pd.DataFrame().to
    # candidate_items = pd.DataFrame(param['candidate_items'])
    # candidate_items['a'] = 1
    # stu_response = pd.DataFrame(param['stu_response'])
    # stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)

    # stu_response = load_stu_response(param['stu_id'], train_data)
    # stu_acc = stu_response.loc[:, 'answer'].sum() / len(stu_response)
    # candidate_items = load_candidate_items(**param)
    # 从候选集合中剔除已作答过的题目
    # candidate_items.drop(stu_response.index, inplace=True, errors='ignore')


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
    parser.add_argument("-r", "--run", dest="run", choices=['online', 'test_one', 'test_level'], default='online',
                        help=u"运行模式")
    parser.add_argument("-l", "--log", dest="log", choices=['info', 'warning', 'debug', 'error'], default='info',
                        help=u"运行模式")

    return parser


if __name__ == '__main__':

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
