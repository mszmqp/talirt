# coding=utf-8
from __future__ import print_function
import warnings
from itertools import combinations
import numpy as np
from psy.utils import inverse_logistic, get_nodes_weights
from psy.fa import GPForth, Factor
from psy.settings import X_WEIGHTS, X_NODES
import pymc3 as pm
import pandas as pd
import theano.tensor as tt
from theano.tensor.basic import as_tensor_variable
from sklearn import metrics
import abc
import matplotlib.pyplot as plt
import sys


class BaseCDM(object):
    def __init__(self, Q: pd.DataFrame, response: pd.DataFrame):

        if 'user_id' not in response.columns or 'item_id' not in response.columns or 'answer' not in response.columns:
            raise ValueError("input DataFrame:response have no user_id or item_id  or answer")

        self._response = response[['user_id', 'item_id', 'answer']]

        self._user_ids = self._response['user_id'].unique()
        self._user_count = len(self._user_ids)
        self._item_ids = self._response['item_id'].unique()
        self._item_count = len(self._item_ids)

        self._user_id_loc = {value: index for index, value in enumerate(self._user_ids)}
        self._item_id_loc = {value: index for index, value in enumerate(self._item_ids)}

        self._user_response_locs = []
        self._item_response_locs = []
        for index, row in self._response.iterrows():
            u_loc = self._user_id_loc[row['user_id']]
            q_loc = self._item_id_loc[row['item_id']]
            self._user_response_locs.append(u_loc)
            self._item_response_locs.append(q_loc)
        self.trace = None

    @abc.abstractmethod
    def estimate_em(self):
        raise NotImplemented

    @abc.abstractmethod
    def estimate_mcmc(self):
        raise NotImplemented

    def __str__(self):
        d = self._response['answer'].value_counts()
        return '\n'.join([
            u"用户数量：%d" % self._user_count,
            u"项目数量：%d" % self._item_count,
            u"记录总数：%d" % len(self._response),
            u'正确数量：%d' % d[1],
            u'错误数量：%d' % d[0],
            u'正确比例：%f%%' % (d[1] * 100.0 / d.sum()),
        ])


class DINA(BaseCDM):
    def __init__(self, *args, **kwargs):
        super(DINA, self).__init__(*args, **kwargs)


class RDINA(DINA):
    pass


class HODINA(DINA):
    pass
