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
        return self.read(file_path)

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

    @staticmethod
    def read(file_path):
        df_data = pd.read_csv(file_path, sep='\t', dtype={'Row': np.int32,
                                                          "Anon Student Id": np.string_,
                                                          "Problem Hierarchy": np.string_,
                                                          "Problem View": np.int32,
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
        return df_data


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
    pass


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
