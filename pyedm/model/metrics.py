#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(zhangzhenhu1@100tal.com)
Date:    2018/4/12 10:21
"""
import sys
from sklearn import metrics


class Metric(object):

    @classmethod
    def metric_mean_error(cls, y_true, y_proba):
        assert len(y_proba) == len(y_true)
        error = {
            'mse': metrics.mean_squared_error(y_true, y_proba),
            'mae': metrics.mean_absolute_error(y_true, y_proba),
        }

        return error

    @classmethod
    def mean_squared_error(cls, y_true, y_proba):
        assert len(y_proba) == len(y_true)
        return metrics.mean_squared_error(y_true, y_proba)

    @classmethod
    def mean_absolute_error(cls, y_true, y_proba):
        assert len(y_proba) == len(y_true)
        return metrics.mean_absolute_error(y_true, y_proba)

    @classmethod
    def plot_prc(cls, y_true, y_proba):
        import matplotlib.pyplot as plt
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)
        print('=' * 20 + 'precision_recall_curve' + "=" * 20, file=sys.stderr)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        return precision, recall, thresholds

    @classmethod
    def confusion_matrix(cls, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        cm = metrics.confusion_matrix(y_true, y_pred)

        return cm

    @classmethod
    def print_confusion_matrix(cls, matrix, file=sys.stdout):
        print("_\t预假\t预真", file=file)
        print("实假\tTN(%d)\tFP(%d)" % (matrix[0][0], matrix[0][1]), file=file)
        print("实真\tFN(%d)\tTP(%d)" % (matrix[1][0], matrix[1][1]), file=file)

    @classmethod
    def classification_report(cls, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0

        return metrics.classification_report(y_true, y_pred, target_names=[u'答错', u'答对'], digits=8)

    @classmethod
    def accuracy_score(cls, y_true, y_proba, threshold):
        y_pred = y_proba.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        score = metrics.accuracy_score(y_true, y_pred)

        return score

    @classmethod
    def accuracy_score_list(cls, y_true, y_proba):
        scores = []
        for threshold in range(11):
            threshold /= 10
            y_pred = y_proba.copy()
            y_pred[y_pred > threshold] = 1
            y_pred[y_pred <= threshold] = 0
            score = metrics.accuracy_score(y_true, y_pred)
            scores.append((threshold, score))
        return scores
