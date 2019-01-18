#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
# setup.py
from setuptools import find_packages
from distutils.core import setup
# from Cython.Build import cythonize
from distutils.extension import Extension
# from Cython.Distutils import Extension
# from Cython.Distutils import build_ext
import numpy.distutils.misc_util
import cython_gsl

DESCRIPTION = 'A set of python modules for machine learning and data mining in education'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
include_dirs.append(cython_gsl.get_include())
NAME = "pyedm"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]
extensions = [
    Extension(name="pyedm.model.bkt._bktc",
              sources=['pyedm/model/bkt/_bktc' + ext],
              include_dirs=include_dirs,
              libraries=cython_gsl.get_libraries() + ["m"],
              library_dirs=[cython_gsl.get_library_dir()],
              ),
    Extension(name="pyedm.model.bkt._bkt",
              sources=['pyedm/model/bkt/_bkt' + '.pyx' if USE_CYTHON else '.cpp'],
              include_dirs=include_dirs,
              libraries=cython_gsl.get_libraries() + ["m"],
              library_dirs=[cython_gsl.get_library_dir()],
              ),
    Extension(name="pyedm.model.irt._uirt_clib",
              sources=['pyedm/model/irt/_uirt_clib' + ext],
              libraries=cython_gsl.get_libraries() + ["m"],
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=include_dirs,
              ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, annotate=True)

setup(
    name=NAME,
    # cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    install_requires=["Cython", "numpy", "pandas"],
    setup_requires=["Cython", "cython_gsl", "numpy", ],
    include_dirs=include_dirs,
    packages=PACKAGES,
    version='0.0.1',  # ''版本号, 通常在 name.__init__.py 中以 __version__ = "0.0.1" 的形式被定义',
    description='教育领域数据挖掘工具包',
    # long_description='PyPI首页的正文',
    # url='你的项目主页, 通常是项目的Github主页',
    # download_url='你项目源码的下载链接',
    # license='版权协议名',
    author='zhangzhenhu',
    author_email='acmtiger@gmail.com',
    zip_safe=False,
)

# rm -fr build/ && python3 setup.py build_ext --inplace &&  mv talirt/talirt/model/*.so talirt/model/
