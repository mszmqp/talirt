#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2019/1/17 10:36
KDD Cup 2010 公开的数据集
KDD Cup 2010 Educational Data Mining Challenge

Description: http://pslcdatashop.web.cmu.edu/KDDCup/

Data Download:
    Download site 1: http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp
    Download site 2: http://neuron.csie.ntust.edu.tw/homework/98/NN/KDDCUP2010/Dataset/



"""
import sys
import argparse

__version__ = 1.0

import pandas as pd
import numpy as np
import os


class KddCup2010:
    """
    read the data of Kdd cup 2010,and return pandas.DataFrame
    """

    def __init__(self, data_dir='/Users/zhangzhenhu/Documents/开源数据/kddcup2010/'):
        self.data_dir = data_dir

    @property
    def a56_train(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2005_2006', 'algebra_2005_2006_train.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def a56_test(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2005_2006', 'algebra_2005_2006_master.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def a67_train(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2006_2007_new_20100409', 'algebra_2006_2007_train.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def a67_test(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2006_2007_new_20100409', 'algebra_2006_2007_master.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def a89_train(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2008_2009', 'algebra_2008_2009_train.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def a89_test(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'algebra_2008_2009', 'algebra_2008_2009_test.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read_test(file_path)

    @property
    def ba67_train(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'bridge_to_algebra_2006_2007',
                                     'bridge_to_algebra_2006_2007_train.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def ba67_test(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'bridge_to_algebra_2006_2007',
                                     'bridge_to_algebra_2006_2007_master.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def ba89_train(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'bridge_to_algebra_2008_2009',
                                     'bridge_to_algebra_2008_2009_train.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read(file_path)

    @property
    def ba89_test(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'bridge_to_algebra_2008_2009',
                                     'bridge_to_algebra_2008_2009_test.txt')
        if not os.path.exists(file_path):
            raise ValueError("file not exists %s" % file_path)
        return self.read_test(file_path)

    @staticmethod
    def read(file_path):
        df_data = pd.read_csv(file_path, sep='\t', dtype={'Row': np.int32,
                                                          "Anon Student Id": np.string_,
                                                          "Problem Hierarchy": np.string_,
                                                          "Problem Name": np.string_,
                                                          "Problem View": np.int32,
                                                          "Step Name": np.string_,
                                                          "Step Duration (sec)": np.float64,
                                                          "Correct Step Duration (sec)": np.float64,
                                                          "Error Step Duration (sec)": np.float64,
                                                          "Correct First Attempt": np.int32,
                                                          "Incorrects": np.int32,
                                                          "Hints": np.int32,
                                                          "Corrects": np.int32,
                                                          "KC(SubSkills)": np.string_,
                                                          "Opportunity(SubSkills)": np.string_,
                                                          })
        df_data['item_name'] = df_data['Problem Name'] + ',' + df_data['Step Name']
        # less columns to save memory
        return df_data[
            ['Row', "Anon Student Id", "Problem Hierarchy", "Correct First Attempt", 'item_name']].copy()

    @staticmethod
    def read_test(file_path):
        df_data = pd.read_csv(file_path, sep='\t', dtype={'Row': np.int32,
                                                          "Anon Student Id": np.string_,
                                                          "Problem Hierarchy": np.string_,
                                                          "Problem Name": np.string_,
                                                          "Problem View": np.int32,
                                                          "Step Name": np.string_,
                                                          # "Step Duration (sec)": np.float64,
                                                          # "Correct Step Duration (sec)": np.float64,
                                                          # "Error Step Duration (sec)": np.float64,
                                                          # "Correct First Attempt": np.int32,
                                                          # "Incorrects": np.int32,
                                                          # "Hints": np.int32,
                                                          # "Corrects": np.int32,
                                                          "KC(SubSkills)": np.string_,
                                                          "Opportunity(SubSkills)": np.string_,
                                                          })
        df_data['item_name'] = df_data['Problem Name'] + ',' + df_data['Step Name']
        # less columns to save memory
        return df_data[['Row', "Anon Student Id", "Problem Hierarchy", "Correct First Attempt", 'item_name']].copy()


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
    data = [
        ["Algebra I 2005-2006", "575", "813,661", "809695", "3968"],
        ["Algebra I 2006-2007", "1,840", "2,289,726", "2270384", "19342"],
        ["Bridge to Algebra 2006-2007", "1,146", "3,656,871", "3679199", "7672"],

        ["algebra_2008_2009", "3,310", "", "8,918,055", "508,913"],
        ["bridge_to_algebra_2008_2009", "6,043", "", "20,012,499", "756,387"],
    ]

    columns = ["Data sets", "Students", "Steps", "Train Size", "Test Size", ]

    rst_table(data[:3], columns)
    rst_table(data[3:], columns)


def rst_table(data, columns):
    import pytablewriter
    writer = pytablewriter.RstGridTableWriter()
    writer.table_name = "Final Report"
    writer.headers = columns
    writer.value_matrix = data

    writer.write_table()


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
