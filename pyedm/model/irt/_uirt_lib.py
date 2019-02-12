#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2018/6/24 16:46
"""
import numpy as np
import numexpr

def add_bias(X: np.ndarray):
    """
    在输入的array后面增加一列常数1
    Parameters
    ----------
    X

    Returns
    -------

    """
    if X.ndim == 1:
        N = X.size
        X = X.reshape((N, 1))
    else:
        N, _ = X.shape
    return np.column_stack((X, np.ones(N)))


def expo(X, Y, sgn=1):
    """
    Th has shape n p+1
    X has shape N p
    """
    X1 = add_bias(X)
    return np.exp(sgn * X1.dot(Y.T))  # Shape N n


def proba1(Th, X):
    exp_minusXT = expo(Th, X, -1)
    return 1 / (1 + exp_minusXT)


def uirt(theta: np.ndarray, slope: np.ndarray = None, intercept: np.ndarray = None, guess: np.ndarray = None):
    if slope is None:
        z = theta + intercept
    else:

        # theta = add_bias(theta)
        # shape=(x,) column_stack 也能正确处理
        # item = np.column_stack((slop, intercept))
        # z = theta.dot(item.T)
        theta = theta.reshape(theta.size, 1)
        slope = slope.reshape(1, slope.size)
        intercept = intercept.reshape(1, intercept.size)
        # z = numexpr.evaluate("theta * slope + intercept")
        z = theta * slope + intercept

    if guess is None:
        return 1.0 / (1.0 + np.exp(-z))
    else:
        guess = guess.reshape(1, guess.size)
        # return numexpr.evaluate("guess + (1.0 - guess) / (1.0 + exp(-z))")
        return guess + (1.0 - guess) / (1.0 + np.exp(-z))


def log_likelihood(response: np.ndarray, theta: np.ndarray,
                   slope: np.ndarray = None, intercept: np.ndarray = None, guess: np.ndarray = None):
    predict = uirt(theta=theta, slope=slope, intercept=intercept, guess=guess)
    # P = proba1(Th, X)
    # ((1 - Y) * log(1 - P)  # Shape N n
    #     + Y  * log(    P))
    ll = response * np.log(predict) + (1.0 - response) * np.log(1 - predict)
    # 处理作答记录里存在空值的问题，空值进行运算得到的结果也是空值。
    np.nan_to_num(ll, copy=False)
    # 注意没有求和
    return ll


def u2irt_item_jac(response: np.ndarray, theta: np.ndarray,
                   slope: np.ndarray = None, intercept: np.ndarray = None, guess: np.ndarray = None):
    """所有题目的一阶导数，批量计算的，一行一个题目"""

    predict = uirt(theta=theta, slope=slope, intercept=intercept, guess=guess)
    error = response - predict
    if slope is None:
        # 单参数模型
        return error.sum(axis=0)  # shape=(m,)
    if guess is None:  # 双参数模型
        theta = add_bias(theta)
        # 处理作答记录里存在空值的问题，空值进行运算得到的结果也是空值。# np.nan_to_num 会把空值填充为0，
        np.nan_to_num(error, copy=False)
        return error.T.dot(theta)  # shape=(m,2)

    # 三参数模型暂未实现
    raise NotImplemented


def u2irt_item_jac_and_hessian(response: np.ndarray, theta: np.ndarray,
                               slope: np.ndarray = None, intercept: np.ndarray = None, guess: np.ndarray = None):
    """所有题目的二阶导数，批量计算，一行一个题目"""
    predict = uirt(theta=theta, slope=slope, intercept=intercept, guess=guess)
    theta = add_bias(theta)

    error = response - predict
    # 处理作答记录里存在空值的问题，空值进行运算得到的结果也是空值。
    # np.nan_to_num 会把空值填充为0，
    np.nan_to_num(error, copy=False)
    jac = error.T.dot(theta)  # Shape=(m,2)

    # theta.shape=(n,2)
    # todo ????
    XiXiT = np.einsum('ij,ki->ijk', theta, theta.T)  # Shape N 2 2
    Lambda = -predict * (1 - predict)  # Shape=(n,m)
    # Lambda = np.where(np.isnan(response), 0, Lambda)
    return jac, np.tensordot(Lambda.T, XiXiT, axes=1)  # Shape n p+1 p+1
