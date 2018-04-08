#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/4/8 17:02
"""
import sys
import argparse
import json

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
    for line in options.input:
        record = line.rstrip('\r\n').split('\1')
        # _id, student_id, course_id, course_level_id, start_time, city_code, isdeleted, create_time, modify_time, answer_content_json = record
        answer_content_json = record.pop(-1)

        answer_content = json.loads(answer_content_json)
        for v in answer_content.itervalues():
            print('\t'.join(record + [

                v['answerContent'],
                v['answerStatus'],
                v['levelQuestionId'],
            ]))


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
