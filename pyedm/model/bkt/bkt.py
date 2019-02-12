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
import pandas as pd
import time
import sys
# from scipy.special import logsumexp

# from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
import pyedm.model.bkt._bktc as bktc
from pyedm.model.bkt.bkt_lib import StandardBKT, IRTBKT, TrainHelper

from pyedm.utils import normalize, log_mask_zero, log_normalize, iter_from_X_lengths, logsumexp
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm


class BKTBatch:
    """Base class for Hidden Markov Models.

    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_stat : int
        Number of states in the model.

    startprior : array, shape (n_stat, )
        Initial state occupation prior distribution.

    transitionprior : array, shape (n_stat, n_stat)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_stat, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_stat, n_stat)
        Matrix of transition probabilities between states.
    """

    def __init__(self,
                 start_init=np.array([0.5, 0.5]),
                 transition_init=None,
                 emission_init=np.array([[0.8, 0.2], [0.2, 0.8]]),
                 # random_state=None,
                 n_stat=2,
                 n_obs=2,
                 max_iter=10, tol=1e-2, n_jobs=0, bkt_model="standard", **kwargs):

        self.n_stat = n_stat  # 隐状态的数量
        self.n_obs = n_obs  # 观测状态的数量

        # 0 会 1不会; 0:做对，1:做错
        # if transition_init is None:
        #     transition_init = np.array([[1, 0], [0.4, 0.6]])
        # self.transition_lb = np.array([[1, 0], [0, 0]]).astype(np.float64)
        # self.transition_ub = np.array([[1, 0], [1, 1]]).astype(np.float64)

        # 0 不会，1会 ；0做错，1做对
        if transition_init is None:
            transition_init = np.array([[0.4, 0.6], [0, 1]])
        self.transition_lb = np.array([[0, 0], [0, 1]]).astype(np.float64)
        self.transition_ub = np.array([[1, 1], [0, 1]]).astype(np.float64)

        self.start_init = start_init
        self.transition_init = transition_init
        self.emission_init = emission_init

        self.max_iter = max_iter
        self.tol = tol
        self.njobs = n_jobs
        self.train_cost_time = 0
        self.preprocess_cost_time = 0
        self.predict_cost_time = 0

        # 约束条件
        self.start_lb = np.array([0, 0]).astype(np.float64)
        self.start_ub = np.array([1, 1]).astype(np.float64)

        self.emission_lb = np.array([[0.7, 0], [0, 0.7]]).astype(np.float64)
        self.emission_ub = np.array([[1, 0.3], [0.3, 1]]).astype(np.float64)
        self.model = {}
        self.bkt_model = bkt_model.upper()
        if self.bkt_model not in ['IRT', 'STANDARD']:
            raise ValueError("bkt_model must one of ['IRT', 'STANDARD']")

        self.items_info = None
        self._init_param(**kwargs)

    def _init_param(self, **kwargs):

        start_init = kwargs.get("start_init", None)
        if start_init is not None and isinstance(start_init, np.ndarray):
            self.start_init = start_init

        transition_init = kwargs.get("transition_init", None)
        if transition_init is not None and isinstance(start_init, np.ndarray):
            self.transition_init = transition_init

        emission_init = kwargs.get("emission_init", None)
        if emission_init is not None and isinstance(emission_init, np.ndarray):
            self.emission_init = emission_init

        max_iter = kwargs.get("max_iter", None)
        if max_iter is not None:
            self.max_iter = max_iter

        tol = kwargs.get("tol", None)
        if tol is not None:
            self.tol = tol

        njobs = kwargs.get("n_jobs", None)
        if njobs is not None:
            self.njobs = njobs

        start_lb = kwargs.get("start_lb", None)
        if start_lb is not None and isinstance(start_lb, np.ndarray):
            self.start_lb = start_lb

        start_ub = kwargs.get("start_ub", None)
        if start_ub is not None and isinstance(start_ub, np.ndarray):
            self.start_ub = start_ub

        transition_lb = kwargs.get("transition_lb", None)
        if transition_lb is not None and isinstance(transition_lb, np.ndarray):
            self.transition_lb = transition_lb
        transition_ub = kwargs.get("transition_ub", None)
        if transition_ub is not None and isinstance(transition_ub, np.ndarray):
            self.transition_ub = transition_ub

        emission_lb = kwargs.get("emission_lb", None)
        if emission_lb is not None and isinstance(emission_lb, np.ndarray):
            self.emission_lb = emission_lb

        emission_ub = kwargs.get("emission_ub", None)
        if emission_ub is not None and isinstance(emission_ub, np.ndarray):
            self.emission_ub = emission_ub

    def set_bound_start(self, lb, up):
        self.start_lb = lb.astype(np.float64)
        self.start_ub = up.astype(np.float64)

    def set_bound_transition(self, lb, up):
        self.transition_lb = lb.astype(np.float64)
        self.transition_ub = up.astype(np.float64)

    def set_bound_emission(self, lb, up):
        self.emission_lb = lb.astype(np.float64)
        self.emission_ub = up.astype(np.float64)

    def set_item_info(self, items: pd.DataFrame):
        self.items_info = items

    def score_samples(self, X, lengths=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_obs)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_stat)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        check_is_fitted(self, "start")
        self._check()

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        posteriors = np.zeros((n_samples, self.n_stat))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_(X[i:j])
            logprobij, fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij

            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    def score(self, X, lengths=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_obs)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        check_is_fitted(self, "start")
        self._check()

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij
        return logprob

    def _decode_viterbi(self, start: np.ndarray, transition: np.ndarray, emission: np.ndarray, obs: np.ndarray):
        # n_samples, n_stat = framelogprob.shape

        return bktc._viterbi(
            obs.shape[0], self.n_stat, start,
            transition, emission, obs)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, trace, obs=None, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_obs)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        # check_is_fitted(self, "start")
        # self._check()

        # algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {0!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        # X = check_array(X)
        # n_samples = X.shape[0]
        # logprob = 0
        _model = self.model.get(trace, None)
        if _model is None:
            raise ValueError("未找到模型训练结果")
        state_sequence, state_prob = decoder(_model['start'], _model['transition'], _model['emission'], obs)
        # todo 第一题的预测

        # 下到题目答对的概率
        next_state = np.argmax(_model['transition'][state_sequence[-1], :])
        next_prob = _model['emission'][next_state, 1]
        return next_prob, next_state, state_sequence, state_prob

    # def predict
    def predict_one(self, obs: np.ndarray, trace, group_key=None, algorithm='viterbi'):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_obs)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        next_prob, next_state, state_sequence, state_prob = self.decode(trace=trace, obs=obs, algorithm=algorithm)
        return next_prob, trace, group_key

    def predict_batch(self, response: pd.DataFrame, trace_by=('knowledge',), group_by=('user',), sorted_by=None,
                      njobs=1):

        _start_time = time.time()
        if njobs is None or njobs <= 0:
            for trace_key, group_keys, train_x, lengths in self._split_data(response=response, trace_by=trace_by,
                                                                            group_by=group_by, sorted_by=sorted_by):
                # train_x = check_array(train_x)
                for i, group_key in enumerate(group_keys):
                    obs = train_x[lengths[i, 0]: lengths[i, 1]]
                    next_prob, trace_key, group_key = self.predict_one(obs=obs, trace=trace_key, group_key=group_key)
                    yield next_prob, trace_key, group_key

        else:
            if njobs == 1:
                njobs = os.cpu_count() - 1
            with ThreadPoolExecutor(max_workers=njobs) as tp:
                futures = []
                for trace_key, group_keys, train_x, lengths in self._split_data(response=response, trace_by=trace_by,
                                                                                group_by=group_by, sorted_by=sorted_by):
                    for i, group_key in enumerate(group_keys):
                        obs = train_x[lengths[i, 0]: lengths[i, 1]]
                        futures.append(tp.submit(self.predict_one, obs=obs, trace=trace_key, group_key=group_key))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    next_prob, trace_key, group_key = future.result()
                    yield next_prob, trace_key, group_key

        self.predict_cost_time = time.time() - _start_time

    def predict_proba(self, X, lengths=None):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_obs)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_stat)
            State-membership probabilities for each sample from ``X``.
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    # @staticmethod
    def _split_data(self, response: pd.DataFrame, trace_by=('knowledge',), group_by=('user',), sorted_by=None):
        if isinstance(trace_by, tuple):
            trace_by = list(trace_by)
        if isinstance(group_by, tuple):
            group_by = list(group_by)
        g = response.groupby(trace_by)
        for trace_key, df_data in g:
            train_x = []
            lengths = []
            # pos = 0
            group_keys = []
            for group_key, df_x in df_data.groupby(group_by):
                if sorted_by is not None:
                    # x = df_x.sort_values(sorted_by)['answer'].values.flatten().astype(np.int32)
                    x = df_x.sort_values(sorted_by)  # ['answer'].values.flatten().astype(np.int32)
                else:
                    # x = df_x['answer'].values.flatten().astype(np.int32)
                    x = df_x  # ['answer'].values.flatten().astype(np.int32)

                train_x.append(x)
                # lengths.append((pos, x.shape[0] + pos))
                lengths.append(x.shape[0])
                # pos += x.shape[0]
                group_keys.append(group_key)
            # train_x = np.concatenate(train_x)
            lengths = np.asarray(lengths).astype(np.int32)
            # 观测序列的长度最小是3，不足这个长度无法训练
            if lengths.max() <= 3:
                continue

            train_x = pd.concat(train_x)
            x = train_x['answer'].values.flatten().astype(np.int32)
            if self.bkt_model == "IRT":

                items_id = train_x['item_id'].values.flatten().astype(np.int32)
            else:
                items_id = None

            # yield trace_key, group_keys, train_x, lengths
            yield {
                'trace': trace_key,
                'group_keys': group_keys,
                # 'train_x': train_x,
                'lengths': lengths,
                'items_id': items_id,
                'x': x,
                # '':train_x['item_id'].values.flatten().astype(np.int32),
            }

    def fit_batch_bak(self, response: pd.DataFrame, trace_by=('knowledge',), group_by=('user',), sorted_by=None,
                      **kwargs):
        """

        Parameters
        ----------
        response :  panda.DataFrame
            作答数据,每行是一条作答数据。必有列:['answer',]，可选列['user','knowledge',]。需要包含trace_by和group_by参数指定的列。
        trace_by : tuple or string
            数据按照trace_by进行分组，每一组训练一个独立的bkt模型。
            例如,每个知识点训练一个bkt，trace_by=('knowledge',);每个知识点+学生训练一个bkt，trace_by=('knowledge','user').

        group_by : tuple or string
            每个bkt模型的数据按照group_by切分序列。例如，当trace_by=('knowledge',)时，每个学生是一个序列group_by=('user')。
        lengths
        kwargs : optional
            其它可选参数，支持start_init transition_init emission_init max_iter tol start_lb

        Returns
        -------

        """

        self._init_param(**kwargs)
        # self._check()
        self.model = {}
        _start_time = time.time()
        records = list(self._split_data(response=response, trace_by=trace_by, group_by=group_by, sorted_by=sorted_by))

        # 不并行
        if self.njobs is None or self.njobs <= 0:
            for record in tqdm(records, total=len(records)):
                result = self.fit_one(**record)
                if result is not None:
                    self.model[result['trace']] = result

        else:  # 并行处理
            if self.njobs == 1:
                self.njobs = os.cpu_count()
            with ThreadPoolExecutor(max_workers=self.njobs) as tp:
                futures = [tp.submit(self.fit_one, **record) for record in records]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    # print('xxxxxxx')
                    result = future.result()
                    if result is not None:
                        self.model[result['trace']] = result
        self.train_cost_time = time.time() - _start_time
        return self

    def fit_batch(self, response: pd.DataFrame, trace_by=('knowledge',), group_by=('user',), sorted_by=None,
                  **kwargs):
        """

        Parameters
        ----------
        response :  panda.DataFrame
            作答数据,每行是一条作答数据。必有列:['answer',]，可选列['user','knowledge',]。需要包含trace_by和group_by参数指定的列。
        trace_by : tuple or string
            数据按照trace_by进行分组，每一组训练一个独立的bkt模型。
            例如,每个知识点训练一个bkt，trace_by=('knowledge',);每个知识点+学生训练一个bkt，trace_by=('knowledge','user').

        group_by : tuple or string
            每个bkt模型的数据按照group_by切分序列。例如，当trace_by=('knowledge',)时，每个学生是一个序列group_by=('user')。
        lengths
        kwargs : optional
            其它可选参数，支持start_init transition_init emission_init max_iter tol start_lb

        Returns
        -------

        """
        _start_time = time.time()
        self._init_param(**kwargs)
        # self._check()
        self.model = {}

        records = list(self._split_data(response=response, trace_by=trace_by, group_by=group_by, sorted_by=sorted_by))

        # for record in self._split_data(response=response, trace_by=trace_by, group_by=group_by, sorted_by=sorted_by):
        #     观测序列的长度最小是3，不足这个长度无法训练
        #   if record['lengths'].max() <= 3:
        #     continue

        #   record['x'] = record['train_x']['answer'].values.flatten().astype(np.int32)
        #   record['items_id'] = record['train_x']['item_id'].values.flatten().astype(np.int32)

        #   records.append(record)
        self.preprocess_cost_time = time.time() - _start_time
        _start_time = time.time()
        # 不并行
        if self.njobs is None or self.njobs <= 0:
            self._fit_sequence(records)

        else:  # 并行处理
            if self.njobs == 1:
                self.njobs = os.cpu_count()
            self._fit_parallel(records, self.njobs)
            # with ThreadPoolExecutor(max_workers=self.njobs) as tp:
            #     futures = [tp.submit(self.fit_one, **record) for record in records]
            #     for future in tqdm(as_completed(futures), total=len(futures)):
            #         print('xxxxxxx')
            #         result = future.result()
            #         if result is not None:
            #             self.model[result['trace']] = result
        self.train_cost_time = time.time() - _start_time
        return self

    def _fit_sequence(self, records):
        for record in tqdm(records, total=len(records)):
            result = self.fit_one(**record)
            if result is not None:
                self.model[result['trace']] = result

    def _fit_parallel(self, records, n_jobs):
        results = parallel_fit(
            data=records,
            n_stat=self.n_stat,
            n_obs=self.n_obs,
            start=self.start_init,
            transition=self.transition_init,
            emission=self.emission_init,
            bound={
                'start_lb': self.start_lb,
                'start_ub': self.start_ub,
                'transition_lb': self.transition_lb,
                'transition_ub': self.transition_ub,

            },
            items_info=self.items_info[['slop', 'difficulty', 'guess']].values.astype(np.float64),
            n_jobs=n_jobs
        )
        for result in results:
            self.model[result['trace']] = result

    def _fit(self, kwargs):

        return self.fit_one(**kwargs)

    def fit_one_bak(self, train_x: np.ndarray, lengths: np.ndarray, trace=None):
        """
        这个函数在测试集下0.6秒
        Parameters
        ----------
        train_x
        lengths
        trace

        Returns
        -------

        """
        start = self.start_init.copy()
        transition = self.transition_init.copy()
        emission = self.emission_init.copy()
        log_likelihood = bktc.fit(train_x, lengths, start, transition, emission,
                                  self.start_lb, self.start_ub,
                                  self.transition_lb, self.transition_ub,
                                  self.emission_lb, self.emission_ub,
                                  self.max_iter, self.tol)

        # print("=" * 10, iter, "=" * 10)
        # print('likelihood', log_likelihood, curr_logprob)
        # print("start", start)
        # print("transmat", transition)
        # print("emission", emission)

        # self.start = start
        # self.transition = transition
        # self.emission = emission
        # self.log_likelihood = log_likelihood

        # print("cost time", cost_time)
        # print("iter", iter)
        return trace, start, transition, emission, log_likelihood

    def fit_one(self, x: np.ndarray, lengths: np.ndarray, items_id: np.ndarray=None, trace=None, **kwargs):
        """
        这个函数在测试集下0.3秒
        Parameters
        ----------
        train_x
        lengths
        trace

        Returns
        -------

        """

        bkt = self._get_bkt_object()
        # x_array = train_x['answer'].values.flatten().astype(np.int32)
        if isinstance(bkt, IRTBKT):
            # item_array = train_x['item_id'].values.flatten().astype(np.int32)
            # bkt.set_items(item_array)
            bkt.estimate(x, lengths, items_id)
        else:
            bkt.estimate(x, lengths)
        # print('cost:', time.time() - _t, file=sys.stderr)
        # if trace is None:
        #     return bkt
        return {
            'success': bkt.success,
            'trace': trace,
            'start': bkt.start,
            'transition': bkt.transition,
            'emission': bkt.emission,
            'log_likelihood': bkt.log_likelihood}
        # print(bkt.emission)
        # return trace, bkt.start, bkt.transition, None, bkt.log_likelihood

    def _get_bkt_object(self):

        start = self.start_init
        transition = self.transition_init
        emission = self.emission_init
        if self.bkt_model == "IRT":
            bkt = IRTBKT(self.n_stat, self.n_obs)
            bkt.init(start, transition)
            if self.items_info is None:
                raise ValueError("items_info must set")
            items_array = self.items_info[['slop', 'difficulty', 'guess']].values.astype(np.float64)
            bkt.set_items_param(items_array)
        else:
            bkt = StandardBKT(self.n_stat, self.n_obs)
            bkt.init(start, transition, emission)
            bkt.set_bounded_emission(self.emission_lb, self.emission_ub)

        bkt.set_bounded_start(self.start_lb, self.start_ub)
        bkt.set_bounded_transition(self.transition_lb, self.transition_ub)
        return bkt

    def show(self):
        for key, m in self.model.items():
            print('=' * 10, key, "=" * 10)
            print(m['success'])
            print(m['start'])
            print(m['transition'])
            print(m['emission'])


def test_irtbkt():
    print("read data...", file=sys.stderr)

    file_name = "/Users/zhangzhenhu/Documents/projects/talirt/train_sample.txt"
    df_data = pd.read_csv(file_name, sep='\t', header=None, names=['answer', 'user', 'item', 'knowledge'])
    df_data.loc[:, 'answer'] -= 1
    item_difficulty = df_data.groupby(['item'])[['answer']].mean()
    item_difficulty.rename(columns={'answer': 'difficulty'}, inplace=True)
    item_difficulty.loc[:, 'difficulty'] = (1 - item_difficulty['difficulty']) * 5
    item_difficulty['slop'] = 1.0
    item_difficulty['guess'] = 0
    item_difficulty['item_id'] = np.arange(item_difficulty.shape[0], dtype=np.int32)

    df_data = df_data.join(item_difficulty['item_id'], how='left', on='item')
    # np.ran
    n_stat = 7

    start_init = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    # assert start_init.sum() == 1

    start_lb = np.array([0] * n_stat, dtype=np.float64)
    start_ub = np.array([1] * n_stat, dtype=np.float64)
    transition_init = np.array([
        [0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    transition_lb = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    transition_ub = np.array([
        [1, .5, .5, .5, .5, .5, .5],
        [0, 1, .5, .5, .5, .5, .5],
        [0, 0, 1, .5, .5, .5, .5],
        [0, 0, 0, 1, .5, .5, .5],
        [0, 0, 0, 0, 1, .5, .5],
        [0, 0, 0, 0, 0, 1, .5],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    for _, df_user in df_data.groupby(['knowledge', 'user']):
        train_x = df_user['answer'].values
        items_id = df_user['item_id'].values
        lengths = np.array([train_x.shape[0]], dtype=np.int32)
        bkt = IRTBKT(n_stat=7)
        bkt.init(start_init, transition_init)
        # hmm.set_bounded_start()
        bkt.set_bounded_start(start_lb, start_ub)
        bkt.set_bounded_transition(transition_lb, transition_ub)
        bkt.set_items_param(item_difficulty[['slop', 'difficulty', 'guess']].values)
        print(items_id)
        bkt.set_items(items_id)
        print(train_x, lengths)
        bkt.estimate(train_x.astype(np.int32), lengths.astype(np.int32))
        bkt.show()
        break


if __name__ == "__main__":
    """
    https://github.com/myudelson/hmm-scalable
    对于空知识点的处理逻辑是：
    把空知识点看成独立的知识点，但是并不进行hmm训练，而是单纯统计其每个作答状态的比例，用这个比例作为预测值。
    """
    import pandas as pd
    import sys
    # from tqdm import tqdm
    from joblib import Parallel, delayed

    test_irtbkt()

    quit(0)

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

    file_name = "/Users/zhangzhenhu/Documents/projects/talirt/train_sample.txt"
    df_data = pd.read_csv(file_name, sep='\t', header=None, names=['answer', 'user', 'item', 'knowledge'])
    df_data.loc[:, 'answer'] -= 1

    bkt = BKT()
    bkt.fit_batch(df_data, njobs=1)
    print('cost time', bkt.train_cost_time)
    for key, value in bkt.model.items():
        print("--------", key, "-----------------")
        print("start_prob\n", value['start'])
        print("transmat\n", value['transition'])
        print("emissionprob\n", value['emission'])

    # for next_prob, trace_key, group_key in bkt.predict_batch(df_data, njobs=1):
    #     print(trace_key, group_key, next_prob)

    # print('cost_time', bkt.predict_cost_time)
    # for line in open("/Users/haoweilai/Documents/projects/talirt/train_sample.txt"):
    #     answer, stu, question, skills = line.strip('\r\n').split('\t')
    #     if not skills or skills == '.' or skills == ' ':
    #         skip_rows += 1
    #         continue
    #     # if response==2:
    #     #     response =0
    #     answer = int(answer) - 1
    # skills = skills.split('~')
    # for skill in skills:
    #     record = slice_data.setdefault(skill, {})
    #     record.setdefault(stu, []).append(int(response) - 1)
    # record['stu'].append(stu)
    # record['response'].append(int(response) - 1)
    # record['question'].append(question)
    # stu_set.add(stu)
