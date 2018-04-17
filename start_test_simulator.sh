#!/usr/bin/env bash
#HADOOP标准文件
__script_dir=$(cd "$(dirname "$0")"; pwd)
#source ${__script_dir}/../../conf/env.sh
export PYSPARK_DRIVER_PYTHON=python
HADOOP_BIN="/usr/local/hadoop-2.6.0//bin/hadoop --config /usr/local/hadoop-2.6.0//etc/hadoop/"
STREAMING_JAR_PATH="/usr/local/hadoop-2.6.0//share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar"
LOCAL_WAREHOUSE="/user/app_bi/warehouse/"
PYTHON_BIN="${PYTHON_HOME}/bin/python"
HADOOP_PYTHON_BIN="hdfs://iz2ze7u02k402bnxra1j1xz:8020/user/app_bi/tools/python27.jar#python"
HADOOP_PYTHON3_BIN="hdfs://iz2ze7u02k402bnxra1j1xz:8020/user/app_bi/tools/python3.jar#python3"
DEFAULT_ENCODE="UTF-8"
SPARK_BIN="${SPARK_HOME}/bin/spark-submit"
HADOOP_HOME="/usr/local/hadoop-2.6.0/"


TASK_NAME="talirt"
AUTHOR="zhangzhenhu"

HDFS_PREFIX="/user/app_bi/test/talirt/test_simulator/"

HADOOP_INPUT="${HDFS_PREFIX}/input_simulator.txt"
HADOOP_OUTPUT="${HDFS_PREFIX}/report"

${HADOOP_BIN} fs -rm ${HDFS_PREFIX}
${HADOOP_BIN} fs -mkdir ${HDFS_PREFIX}

${HADOOP_BIN} fs -put ${__script_dir}/input_simulator.txt ${HADOOP_INPUT}
#${HADOOP_BIN} fs -put ${__script_dir}/train_df.pickle ${HDFS_PREFIX}
#${HADOOP_BIN} fs -put ${__script_dir}/test_df.pickle ${HDFS_PREFIX}




NAME=${__script_dir}/../talirt.jar
rm -rf $NAME
cd ${__script_dir}/../
jar cf $NAME -C talirt/ .
#HADOOP_PYTHON_BIN=`echo ${HADOOP_PYTHON_BIN} | awk -F '#' '{print $1}'`



HADOOP_FILE="${__script_dir}/test_simulator.py#run.py,${__script_dir}/../talirt.jar#talirt"
MAPPER="./python3/bin/python3 run.py -r mapper"
REDUCER="cat"


#map任务个数
MAP_TASK_NUM=50
#reduce任务个数
REDUCE_TASK_NUM=1
#最多同时运行map任务数
MAP_CAPACITY=1000
#最多同时运行reduce任务数
REDUCE_CAPACITY=1000
COMPRESS=false
OUTPUT_COMPRESS=false
PARTITIONER=org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner
#job优先级,VERY_HIGH | HIGH | NORMAL | LOW | VERY_LOW
PRIORITY=VERY_HIGH
MEMORY_LIMIT=1024
#
#    -D parquet.read.support.class=net.iponweb.hadoop.streaming.parquet.GroupReadSupport \
#    -inputformat net.iponweb.hadoop.streaming.parquet.ParquetAsJsonInputFormat \
#     -libjars ${LIB_PATH}/hadoop2-iow-lib-1.20.jar,${LIB_PATH}/parquet-hadoop-bundle-1.8.1.jar\

${HADOOP_BIN} jar ${STREAMING_JAR_PATH} \
    -files ${HADOOP_FILE} \
    -archives "${HADOOP_PYTHON3_BIN}" \
    -D 'mapreduce.map.env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./python3/lib' \
    -D stream.num.map.output.key.fields=1 \
    -D num.key.fields.for.partition=1 \
    -D mapreduce.job.name="${MODULE_NAME}.${TASK_NAME}.${AUTHOR}" \
    -D mapreduce.job.reduces=${REDUCE_TASK_NUM} \
    -D mapreduce.job.maps=${MAP_TASK_NUM} \
    -D mapreduce.job.running.map.limit=${MAP_CAPACITY} \
    -D mapreduce.job.running.reduce.limit=${REDUCE_CAPACITY} \
    -D mapreduce.map.output.compress=${COMPRESS} \
    -D mapreduce.output.fileoutputformat.compress=${OUTPUT_COMPRESS} \
    -D mapreduce.job.priority=${PRIORITY} \
    -D mapreduce.reduce.shuffle.input.buffer.percent=0.1 \
    -D mapreduce.reduce.shuffle.parallelcopies=5 \
    -D mapreduce.job.reduce.slowstart.completedmaps=1 \
    -D mapreduce.map.memory.mb=6096 \
    -D mapreduce.reduce.memory.mb=2096 \
    -D mapreduce.task.timeout=0 \
    -inputformat org.apache.hadoop.mapred.lib.NLineInputFormat \
    -mapper "${MAPPER}"  \
    -reducer "${REDUCER}" \
    -input ${HADOOP_INPUT} \
    -output ${HADOOP_OUTPUT} \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner

if [ $? -ne 0 ]; then
	echo "$(date +"%Y-%m-%d %H:%M:%S") hadoop error!"
	exit 1
fi


