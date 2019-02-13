#!/usr/bin/env bash
#export PATH="/usr/local/opt/llvm/bin:$PATH"
#export LDFLAGS="-L/usr/local/opt/llvm/lib"
#export CPPFLAGS="-I/usr/local/opt/llvm/include"
#
#export CXX=clang
#export CC=clang
rm -fr build/
python3 setup.py build_ext --inplace