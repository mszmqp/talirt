#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
# setup.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy.distutils.misc_util

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
setup(
    name='talirt',
    ext_modules=
        # cythonize('talirt/model/cirt.pyx', annotate=True),
        cythonize('talirt/model/bktc.pyx', annotate=True),


    include_dirs=include_dirs,
)
