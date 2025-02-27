#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2018/4/8 17:02
"""
import sys
import argparse
import json
import os

__version__ = 1.0


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
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-r", "--run", dest="runner", default="",
                        help=u"")
    return parser


def mapper(options):
    map_input_file = os.environ['map_input_file']
    # map_input_file = map_input_file
    p_month = map_input_file.split('/')[-2].split('=')[-1]
    for line in options.input:
        record = line.rstrip('\r\n').split('\1')
        answer_content_json = record.pop(-1)
        columns = ['id', 'student_id', 'course_id', 'course_level_id', 'start_time', 'city_code', 'isdeleted',
                   'create_time', 'modify_time']

        answer_content = json.loads(answer_content_json)

        base_record =  {
                    col: value for col, value in zip(columns, record)
                }
        base_record['isdeleted'] = int(base_record['isdeleted'])
        if base_record['isdeleted'] != 0:
            continue

        base_record['p_month'] = p_month

        for v in answer_content.itervalues():
            v.update(base_record)
            print(json.dumps(v))


def reducer(options):
    pass


def main(options):
    if options.runner == 'mapper':
        mapper(options)

    elif options.runner == 'reducer':

        reducer(options)


if __name__ == "__main__":
    parser = init_option()
    options = parser.parse_args()  # if options.help:
    #     # OptionParser 自己的print_help()会导致乱码
    #     usage = parser.format_help()
    #     print (usage.encode("UTF-8"))
    #     quit()
    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
