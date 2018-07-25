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
from setuptools import find_packages

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
include_dirs.append(cython_gsl.get_include())
NAME = "talirt"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]
ext_modules = [
    # Extension("crandom",
    #           ['talirt/model/crandom.pyx'],
    #           libraries=cython_gsl.get_libraries(),
    #           library_dirs=[cython_gsl.get_library_dir()],
    #           include_dirs=include_dirs),
    Extension("uirt_clib",
              ['talirt/utils/uirt_clib.pyx'],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=include_dirs,
              ),
]

setup(
    name=NAME,
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules, annotate=True),
    # cythonize('talirt/model/cirt.pyx', annotate=True),
    # cythonize('talirt/model/bktc.pyx', annotate=True),
    # cythonize(['talirt/model/crandom.pyx','talirt/model/c_utils.pyx'], annotate=True,gdb_debug=True),
    install_requires=["Cython", "cython_gsl", "numpy", "pandas"],
    setup_requires=["Cython", "cython_gsl", "numpy", ],
    include_dirs=include_dirs,

    packages=PACKAGES,
    # version='版本号, 通常在 name.__init__.py 中以 __version__ = "0.0.1" 的形式被定义',
    # description='PyPI首页的一句话短描述',
    # long_description='PyPI首页的正文',
    # url='你的项目主页, 通常是项目的Github主页',
    # download_url='你项目源码的下载链接',
    # license='版权协议名',
)
