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


class Bkt(MultinomialHMM):

    def __init__(self, n_components=2,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=200, tol=1e-5, verbose=False,
                 params="ste", init_params="ste"):
        MultinomialHMM.__init__(self, n_components,
                                startprob_prior=startprob_prior,
                                transmat_prior=transmat_prior,
                                algorithm=algorithm,
                                random_state=random_state,
                                n_iter=n_iter, tol=tol, verbose=verbose,
                                params=params, init_params=init_params)

    def _do_mstep(self, stats):
        super(Bkt, self)._do_mstep(stats)
        self.transmat_[0][0] = 1
        self.transmat_[0][1] = 0


if __name__ == "__main__":

    import pandas as pd
    import sys
    from tqdm import tqdm

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
    for line in sys.stdin:
        response, stu, question, skills = line.strip('\r\n').split('\t')
        if not skills or skills == '.' or skills == ' ':
            skip_rows += 1
            continue

        skills = skills.split('~')
        for skill in skills:
            record = slice_data.get(skill, {'stu': [],
                                            'response': [],
                                            'question': [], })
            record['stu'].append(stu)
            record['response'].append(int(response))
            record['question'].append(question)
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
    index = 0
    for skill, data in tqdm(slice_data.items(), total=nK):
        bkt = Bkt()
        bkt.fit(np.array(data['response']))
        print("%d\t%s" % (index, skill))
        print("PI\t%f\t%f" % bkt.startprob_)
        print(
            "A\t%f\t%f\t%f\t%f" % (bkt.transmat_[0][0], bkt.transmat_[0][1], bkt.transmat_[1][0], bkt.transmat_[1][1]))
        print(
            "B\t%f\t%f\t%f\t%f" % (
                bkt.emissionprob_[0][0], bkt.emissionprob_[0][1], bkt.emissionprob_[1][0], bkt.emissionprob_[1][1]))
        index += 1
