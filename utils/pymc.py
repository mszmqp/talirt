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


class TextTrace(Text):
    def __init__(self, name, model=None, vars=None, test_point=None, chain=None, ):
        super(TextTrace,self).__init__(name, model, vars, test_point)
        self.chain = chain
