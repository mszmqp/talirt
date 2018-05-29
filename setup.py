#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
# setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='talirt',
    ext_modules=cythonize('talirt/model/cirt.pyx')
)
