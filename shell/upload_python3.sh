#!/usr/bin/env bash
"""
Authors: zhangzhenhu(zhangzhenhu@iwaimai.baidu.com)
Date:    2015/12/20 16:56
"""

__script_dir=$(cd "$(dirname "$0")"; pwd)
HADOOP_PYTHON_BIN=/user/app_bi/tools/python3.jar

cd /usr/local/
NAME=${HOME}/python3.jar
rm -rf $NAME
#tar czvf $NAME *
jar cvf $NAME -C python3/ .
#HADOOP_PYTHON_BIN=`echo ${HADOOP_PYTHON_BIN} | awk -F '#' '{print $1}'`
hadoop fs -rm ${HADOOP_PYTHON_BIN}
hadoop fs -put $NAME ${HADOOP_PYTHON_BIN}