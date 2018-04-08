#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#

# from __future__ import print_function
import sys

import os
import argparse
# $example on:spark_hive$
from os.path import expanduser, join

from pyspark.sql import SparkSession
from pyspark.sql import Row, Column
from pyspark.sql import functions as F
import datetime

reload(sys)
sys.setdefaultencoding('utf-8')


def main(options):
    os.environ['SPARK_HOME'] = '/usr/local/spark-2.2.1-bin-hadoop2.6/'
    # os.environ['PYSPARK_PYTHON'] = './python27/bin/python2'
    builder = SparkSession.builder
    builder.appName("json2hive_%s" % options.output)
    builder.config("spark.executor.instances", 10)
    builder.config("spark.executor.cores", 1)
    builder.config("spark.executor.memory", '3G')
    builder.config("spark.driver.memory", '4G')
    # builder.config("spark.yarn.dist.archives",
    #                'hdfs://iz2ze7u02k402bnxra1j1xz:8020/user/app_bi/tools/python27.jar#python27')
    # 这个配置不生效，用环境变量PYSPARK_PYTHON替代
    # builder.config("spark.pyspark.python", './python27/bin/python2')
    builder.enableHiveSupport()
    spark = builder.getOrCreate()

    spark.catalog.setCurrentDatabase('app_bi')

    df_data = spark.read.json(options.input)
    to_table = spark.table(options.output)

    df_data.select(*to_table.columns).write.insertInto(options.output, overwrite=True)
    spark.stop()


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    # OptionParser 自己的print_help()会导致乱码，这里禁用自带的help参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action="store_true", default=False,
                        dest="debug",
                        help=u"开启debug模式")
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入路径")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"hive table name")
    return parser


if __name__ == '__main__':
    parser = init_option()
    options = parser.parse_args()

    main(options)