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
    pass


class DINA(BaseCDM):
    pass


class RDINA(DINA):
    pass


class HODINA(DINA):
    pass
