#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/4/17 13:06
"""
import sys
import argparse

from pymc3.backends.base import MultiTrace
from pymc3.sampling import _cpu_count
from pymc3.backends.text import Text
from pymc3.backends.sqlite import SQLite


class TextTrace(Text):
    """
    数据是写入到文件的，但是。。。。
    在访问数据时（包括切片操作），会把整个文件读入内存，仍然会导致超内存的问题。
    """
    def __init__(self, name, model=None, vars=None, test_point=None, chain=None):
        super(TextTrace, self).__init__(name, model, vars, test_point)
        self.chain = chain


class SQLiteTrace(SQLite):
    """
    sqlite 限制一张表列的数量最大为2000，可以重新编译调整这个值，但最大不超过32767，坑爹
    http://www.sqlite.org/limits.html
    """
    def __init__(self, name, model=None, vars=None, test_point=None, chain=None):
        super(SQLiteTrace, self).__init__(name, model, vars, test_point)
        self.chain = chain
