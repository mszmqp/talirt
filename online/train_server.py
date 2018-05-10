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
logger = logging.getLogger()
logger.addHandler(log_file_handler)

from run_recomm import Recommend, DiskDB
# from confluent_kafka.kafkatest.verifiable_client import VerifiableClient
# from confluent_kafka.kafkatest.verifiable_consumer import VerifiableConsumer
from confluent_kafka import Consumer, KafkaError
import re


def set_config(conf, args):
    """ Set client config properties using args dict. """
    for n, v in args.items():
        if v is None:
            continue

        if not n.startswith('conf_'):
            # App config, skip
            continue

        # Remove conf_ prefix
        n = n[5:]

        # Handle known Java properties to librdkafka properties.
        if n == 'partition.assignment.strategy':
            # Convert Java class name to config value.
            # "org.apache.kafka.clients.consumer.RangeAssignor" -> "range"
            conf[n] = re.sub(r'org.apache.kafka.clients.consumer.(\w+)Assignor',
                             lambda x: x.group(1).lower(), v)

        elif n == 'enable.idempotence':
            # Ignore idempotence for now, best-effortly.
            sys.stderr.write('%% WARN: Ignoring unsupported %s=%s\n' % (n, v))
        else:
            conf[n] = v


def read_config_file(path):
    """Read (java client) config file and return dict with properties"""
    conf = {}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('#') or len(line) == 0:
                continue

            fi = line.find('=')
            if fi < 1:
                raise Exception('%s: invalid line, no key=value pair: %s' % (path, line))

            k = line[:fi]
            v = line[fi + 1:]

            conf[k] = v

    return conf


class Storage:

    def get_by_es(self):
        pass

    def get_by_kudu(self):
        pass

    def get_by_stu_id(self, stu_id):
        pass

    def get_by_level_id(self, level_id):
        pass


def train_level_model(level_response, param):
    param = {}
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
    parser.add_argument('--topic', action='append', type=str, default='ips.tb_stu_answer')
    parser.add_argument('--group-id', dest='conf_group.id', default="bidev_curriculum_knowledge_practice")
    parser.add_argument('--broker-list', dest='conf_bootstrap.servers',
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
    # from confluent_kafka import Consumer, KafkaError,
    conf = {'broker.version.fallback': '0.8.x.y',
            # Do explicit manual offset stores to avoid race conditions
            # where a message is consumed from librdkafka but not yet handled
            # by the Python code that keeps track of last consumed offset.
            'enable.auto.offset.store': False}
    args = vars(options)
    if args.get('consumer_config', None) is not None:
        args.update(read_config_file(args['consumer_config']))

    args.update([x[0].split('=') for x in args.get('extra_conf', [])])

    set_config(conf, args)
    c = Consumer(conf)
    print(args['topic'])
    c.subscribe(args['topic'].split(','))

    if options.train_type == 'level':
        train_func = train_level_model
    elif options.train_type == 'student':
        train_func = None
    while True:
        msg = c.poll(1.0)

        if msg is None:
            logger.info('time out')
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                logger.error(msg.error())
                break

        print('Received message: {}'.format(msg.value().decode('utf-8')))
        train_func(msg.value())
    c.close()


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
