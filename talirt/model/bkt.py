#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/4/24 14:18
"""
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import time


class Bkt(MultinomialHMM):

    def __init__(self, n_components=2,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=200, tol=1e-5, verbose=False,
                 params="ste", init_params=""):
        MultinomialHMM.__init__(self, n_components,
                                startprob_prior=startprob_prior,
                                transmat_prior=transmat_prior,
                                algorithm=algorithm,
                                random_state=random_state,
                                n_iter=n_iter, tol=tol, verbose=verbose,
                                params=params, init_params=init_params)
        self.n_features = 2
        self.startprob_ = np.array([0.5, 0.5])
        self.transmat_ = np.array([[1 - 1e-8, 1e-8], [0.4, 0.6]])
        self.emissionprob_ = np.array([[0.8, 0.2], [0.2, 0.8]])

    def _do_mstep(self, stats):
        """"""
        super(Bkt, self)._do_mstep(stats)
        # hmmlearn 在计算前后向算法时，计算乘法通过加log改成计算加法，所以这里概率值就不能再是0了。
        #
        self.transmat_[0][1] = 1e-8
        self.transmat_[0][0] = 1 - self.transmat_[0][1]

        if self.transmat_[1][0] <= 0:
            self.transmat_[1][0] = 1e6
            self.transmat_[1][1] = 1 - self.transmat_[1][0]

        # 约束控制
        if self.emissionprob_[0][1] > 0.3:
            self.emissionprob_[0][1] = 0.3

        if self.emissionprob_[0][1] <= 0:
            self.emissionprob_[0][1] = 1e-6

        if self.emissionprob_[1][0] > 0.3:
            self.emissionprob_[1][0] = 0.3

        if self.emissionprob_[1][0] <= 0:
            self.emissionprob_[1][0] = 1e-6

        self.emissionprob_[0][0] = 1 - self.emissionprob_[0][1]
        self.emissionprob_[1][1] = 1 - self.emissionprob_[1][0]

    def model_param(self):
        return {'startprob': self.startprob_,
                'transmat': self.transmat_,
                'emissionprob': self.emissionprob_}


def _cpu_count():
    """Try to guess the number of CPUs in the system.

    We use the number provided by psutil if that is installed.
    If not, we use the number provided by multiprocessing, but assume
    that half of the cpus are only hardware threads and ignore those.
    """
    try:
        import psutil
        cpus = psutil.cpu_count(False)
    except ImportError:
        import multiprocessing
        try:
            cpus = multiprocessing.cpu_count() // 2
        except NotImplementedError:
            cpus = 1
    if cpus is None:
        cpus = 1
    return cpus


def train(x, lengths):
    start = time.time()
    bkt = Bkt()
    bkt.fit(x, lengths)
    print(bkt.monitor_.iter, bkt.monitor_.converged, 'elapsed ', time.time() - start, file=sys.stderr)
    return bkt.model_param()


if __name__ == "__main__":

    """
    https://github.com/myudelson/hmm-scalable
    对于空知识点的处理逻辑是：
    把空知识点看成独立的知识点，但是并不进行hmm训练，而是单纯统计其每个作答状态的比例，用这个比例作为预测值。
    """
    import pandas as pd
    import sys
    from tqdm import tqdm
    from joblib import Parallel, delayed

    skip_rows = 0
    SolverId = "1.2"
    nK = 0  # 知识点数量
    nG = 0  # 学生数量
    nS = 2
    nO = 2
    nZ = 1
    slice_data = {
    }
    stu_set = set()
    print("read data...", file=sys.stderr)
    for line in sys.stdin:
        response, stu, question, skills = line.strip('\r\n').split('\t')
        if not skills or skills == '.' or skills == ' ':
            skip_rows += 1
            continue

        skills = skills.split('~')
        for skill in skills:
            record = slice_data.setdefault(skill, {})
            record.setdefault(stu, []).append(int(response) - 1)
            # record['stu'].append(stu)
            # record['response'].append(int(response) - 1)
            # record['question'].append(question)
            stu_set.add(stu)
    nK = len(slice_data)
    nG = len(stu_set)
    print("SolverId\t%s" % (SolverId))
    print("nK\t" + str(nK))
    print("nG\t" + str(nG))
    print("nS\t" + str(nS))
    print("nO\t" + str(nO))
    print("nZ\t" + str(nZ))
    print("Null skill ratios\txx\txx")

    print('train model...', file=sys.stderr)
    jobs = []
    skills = []
    for skill, data in slice_data.items():
        obs = list(data.values())
        lengths = [len(line) for line in obs]
        x = np.concatenate(obs)
        x = x.reshape((len(x), 1))
        # bkt = Bkt()
        # x = np.array(data['response'])
        # bkt.fit(x.reshape((x.shape[0], 1)))
        # print(len(x),sum(lengths))
        jobs.append(delayed(train)(x, lengths))
        skills.append(skill)
    results = Parallel(n_jobs=_cpu_count() - 1)(jobs)
    index = 0
    for skill, bkt in zip(skills, results):
        print("%d\t%s" % (index, skill))
        print("PI\t%f\t%f" % tuple(bkt['startprob'].flatten()))
        print(
            "A\t%f\t%f\t%f\t%f" % (
                bkt['transmat'][0][0], bkt['transmat'][0][1], bkt['transmat'][1][0], bkt['transmat'][1][1]))
        print(
            "B\t%f\t%f\t%f\t%f" % (
                bkt['emissionprob'][0][0], bkt['emissionprob'][0][1], bkt['emissionprob'][1][0],
                bkt['emissionprob'][1][1]))
        index += 1
