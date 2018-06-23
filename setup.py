#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
# setup.py
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util
import cython_gsl

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
include_dirs.append(cython_gsl.get_include())

ext_modules = [
    # Extension("crandom",
    #           ['talirt/model/crandom.pyx'],
    #           libraries=cython_gsl.get_libraries(),
    #           library_dirs=[cython_gsl.get_library_dir()],
    #           include_dirs=include_dirs),
                Extension("uirt_lib",
                         ['talirt/utils/uirt_lib.pyx'],
                         libraries=cython_gsl.get_libraries(),
                         library_dirs=[cython_gsl.get_library_dir()],
                         include_dirs=include_dirs,
                        ),
               ]

setup(
    name='talirt',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules,annotate=True),
    # cythonize('talirt/model/cirt.pyx', annotate=True),
    # cythonize('talirt/model/bktc.pyx', annotate=True),
    # cythonize(['talirt/model/crandom.pyx','talirt/model/c_utils.pyx'], annotate=True,gdb_debug=True),

    include_dirs=include_dirs,
)
