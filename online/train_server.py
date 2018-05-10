#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/5/10 10:09
"""
import sys
import argparse
# import kudu
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import logging
import os
import json
# logger_ch = logging.StreamHandler(stream=sys.stderr)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logger_ch])
# logger = logging.getLogger("recommend")

log_fmt = '%(asctime)s\tproc %(process)s\t%(levelname)s\t%(message)s'
formatter = logging.Formatter(log_fmt)
# 创建TimedRotatingFileHandler对象
log_file_handler = TimedRotatingFileHandler(filename="train_server.log", when='D', interval=1,
                                            backupCount=0)
# log_file_handler.suffix = "-%W.log"
# log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
log_file_handler.setFormatter(formatter)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_server")
logger.addHandler(log_file_handler)

from run_recomm import Recommend, DiskDB
# from confluent_kafka.kafkatest.verifiable_client import VerifiableClient
# from confluent_kafka.kafkatest.verifiable_consumer import VerifiableConsumer
# from confluent_kafka import Consumer, KafkaError
# import re
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np


class Storage:

    def __init__(self, bakend='kudu'):
        self.bakend = bakend
        if bakend == 'kudu':
            import kudu
            self.client_kudu = kudu.connect(host='192.168.23.195', port=7051)
        elif bakend == 'es':
            self.client_es = Elasticsearch(
                ['http://elastic:Y0Vu72W5hIMTBiU@es-cn-mp90i4ycm0007a7ko.elasticsearch.aliyuncs.com:9200/'])

    def get_by_es(self):
        pass

    def get_level_by_kudu(self, param):

        """
        kudu table schema
          sa_city_code           string NOT NULL
          sa_class_id            string NOT NULL
          sa_cuc_id              string NOT NULL
          sa_id                  string NOT NULL
          sa_stu_id              string
          sa_c_id                string
          sa_cl_id               string
          sa_lt_id               string
          sa_qst_id              string
          sa_qst_num             string
          sa_lq_id               string
          sa_start_tm            string
          sa_elapsed_tm          int32
          sa_answer_status       int32
          sa_answer_cont         string
          sa_score               int32
          sa_year                string
          sa_term_id             string
          sa_grd_id              string
          sa_subj_id             string
          sa_lev_id              string
          sa_detail              string
          sa_is_deleted          int32
          sa_is_fixup            int32
          sa_tutorimgurl         string
          sa_studentanswerurl    string
          sa_questiontypestatus  int32
          sa_tutoraudiourl       string
          sa_create_time         string
          sa_knowledge_id        string
          sa_modify_time         string
          sa_tutoraudiotime      int32
          sa_knowledge_id2       string
          sa_baiduurl            string
          sa_serverurl           string
          sa_imgsourcestatus     int32
          sa_tutorcontent        string
          mycat_time             string
          event_op_type          string
          event_timestamp        int64
          etl_time               string
          binlog_pos             int64
        """
        table = self.client_kudu.table('ods_ips_tb_stu_answer')
        scanner = table.scanner()
        preds = [
            table['sa_year'] == param['year'],
            table['sa_city_code'] == param['city_id'],
            table['sa_grd_id'] == param['grade_id'],
            table['sa_term_id'] == param['term_id'],
            table['sa_subj_id'] == param['subject_id'],
            table['sa_lev_id'] == param['level_id'],
        ]
        scanner.add_predicates(preds)
        scanner.open()
        data = scanner.read_all_tuples()
        df_response = self._tuples_2_dataframe(data, table.schema.names)
        scanner.close()
        return df_response

    @staticmethod
    def _tuples_2_dataframe(tuples, names):

        df = pd.DataFrame(tuples, columns=names)
        df = df.loc[:, ['sa_stu_id', 'sa_qst_id', 'sa_answer_status']].rename(
            columns={'sa_stu_id': 'user_id',
                     'sa_qst_id': 'item_id',
                     'sa_answer_status': 'answer',
                     })
        df.loc[df['answer'] == 2, 'answer'] = 0
        df.loc[:, 'a'] = 1
        df.loc[:, 'b'] = 2
        return df

    def get_by_stu(self, stu_id):
        pass

    def get_by_level(self, param):
        if self.bakend == 'kudu':
            return self.get_level_by_kudu(param)


storage = Storage()


def train_level_model(record):
    table = record['event_tablename']
    # if table not in ['']
    is_deleted = record['sa_is_deleted']
    param = {
        'year': record['sa_year'],
        'city_id': record['sa_city_code'],
        'grade_id': record['sa_grd_id'],
        'term_id': record['sa_term_id'],
        'subject_id': record['sa_subj_id'],
        'level_id': record['sa_lev_id'],
        'class_id': record['sa_class_id'],
        'cuc_id': record['sa_cuc_id'],
        'item_id': record['sa_qst_id'],
        'knowledge_id': record['sa_knowledge_id'],
        'knowledge_id2': record['sa_knowledge_id2'],
        'user_id': record['sa_stu_id'],
        'answer': record['sa_answer_status'],

    }
    key = '_'.join([
        param['year'],
        param['city_id'],
        param['grade_id'],
        param['term_id'],
        param['subject_id'],
        param['level_id'],
    ])
    level_response = storage.get_by_level(param)

    rec_obj = Recommend(db=DiskDB(), param=param)
    # print('-' * 10, 'train', '-' * 10, file=sys.stderr)

    ok = rec_obj.train_model(level_response)

    print('train_model', ok, file=sys.stderr)
    print('-' * 10, 'save', '-' * 10, file=sys.stderr)

    rec_obj.save_model()


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"输出文件；默认标准输出设备")
    parser.add_argument("-t", "--train_model", dest="train_type", choices=['student', 'level'], default='level',
                        help=u"输出文件；默认标准输出设备")
    parser.add_argument('--topic', type=str, default='ips.tb_stu_answer')
    parser.add_argument('--group-id', dest='group', default="bidev_curriculum_knowledge_practice")
    parser.add_argument('--broker-list', dest='servers',
                        default='192.168.14.133:9092,192.168.14.192:9092,192.168.14.193:9092')
    parser.add_argument('--session-timeout', type=int, dest='conf_session.timeout.ms', default=6000)
    parser.add_argument('--enable-autocommit', action='store_true', dest='conf_enable.auto.commit', default=False)
    parser.add_argument('--max-messages', type=int, dest='max_messages', default=-1)
    parser.add_argument('--assignment-strategy', dest='conf_partition.assignment.strategy')
    parser.add_argument('--reset-policy', dest='conf_auto.offset.reset', default='earliest')
    parser.add_argument('--consumer.config', dest='consumer_config')
    parser.add_argument('-X', nargs=1, dest='extra_conf', action='append', help='Configuration property', default=[])

    return parser


def run_forever(options):
    consumer = KafkaConsumer(options.topic,
                             group_id=options.group,
                             bootstrap_servers=options.servers.split(','))
    if options.train_type == 'level':
        train_func = train_level_model
    elif options.train_type == 'student':
        train_func = None

    try:
        for message in consumer:
            # message value and key are raw bytes -- decode if necessary!
            # e.g., for unicode: `message.value.decode('utf-8')`
            print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                                 message.offset, message.key,
                                                 message.value))

            train_func(json.loads(message.value))
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
        pass

    logger.info('Closing consumer')
    consumer.close()


def main(options):
    run_forever(options)


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
