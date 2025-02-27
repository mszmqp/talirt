//
// Created by 张振虎 on 2018/10/31.
//
#include <iostream>
//#include <gsl/gsl_matrix.h>


#ifndef BKT_HMM_H_
#define BKT_HMM_H_

#include <map>
#include <string>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

#include "utils.h"

//typedef signed char NPAR;



bool issimplexbounded(double *ar, double *lb, double *ub, int size);

void projectsimplexbounded(double *ar, double *lb, double *ub, int size);


/*
 * HMM的估计算法中涉及到缩放因子和多序列问题，
 * 参考论文 https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
 */
class HMM {
public:
    int iter;
    double log_likelihood;
    int n_obs;
    int n_stat;
    int max_iter;
    double *PI;
    double **A;
    double **B;
    bool success;
    std::string msg;


private:
    double *PI_LOW, *PI_UPPER;
    double **A_LOW, **A_UPPER;
    double **B_LOW, **B_UPPER;

    double **fwdlattice;
    double **backlattice;
    double **gammalattice;
//    double **xi_sum;
    double *cn;

    int x_pos;
    int *x_ptr;
    int minimum_obs;


//    EmissionDistribution *ed;

public:
    ///
    /// \param n_stat 隐状态的数量
    /// \param n_obs  观测状态的数量
    HMM(int n_stat = 2, int n_obs = 2);

//    ~HMM();
    virtual ~HMM() = 0;

    /// 初始化设置参数
    /// \param pi 初始概率
    /// \param a 转移概率
    /// \param b 发射概率
    void init(double pi[] = NULL, double a[] = NULL, double b[] = NULL);

    /// 设置初始概率PI的约束
    /// \param lower 对应位置值的下限。一维数组。
    /// \param upper 对应位置值的上限。一维数组。
    void set_bound_pi(double lower[] = NULL, double upper[] = NULL);

    /// 设置转移概率矩阵A的约束
    /// \param lower 对应位置值的下限。用一维连续空间表示二维数组。
    /// \param upper 对应位置值的上限。用一维连续空间表示二维数组。
    void set_bound_a(double lower[] = NULL, double upper[] = NULL);

    /// 设置发射概率矩阵B的约束
    /// \param lower 对应位置值的下限。用一维连续空间表示二维数组。
    /// \param upper 对应位置值的上限。用一维连续空间表示二维数组。
    void set_bound_b(double lower[] = NULL, double upper[] = NULL);

    /// 估计模型参数
    /// \param x 观测值序列。可以是多个不同的观测序列堆在一起，lengths数组记录每个观测序列的长度。len(x)=sum(lengths)
    /// \param lengths 每个观测序列的长度。
    /// \param n_lengths lengths数组的长度，也就是有多少个不同的观测序列。
    /// \param max_iter 最大迭代次数
    /// \param tol 收敛的精度，当两轮迭代似然值的差值小于tol时，结束迭代。
    /// \return
    bool fit(int x[], int lengths[], int n_lengths, int max_iter = 20, double tol = 1e-2);

    /// 获取初始概率的值
    /// \param out 用于保存输出值。一维数组的指针。
    void get_pi(double *out);

    /// 获取转移概率的值
    /// \param out 用于保存输出值。用一维连续空间表示二维数组。
    void get_a(double *out);

    /// 获取发射概率的值
    /// \param out 用于保存输出值。用一维连续空间表示二维数组。
    void get_b(double *out);


    /// 给定观测序列，输出隐状态的分布。实际就是前向算法的结果。
    /// \param out
    /// \param x
    /// \param n_x
    /// \return
    double posterior_distributed(double *out, int *x, int n_x);

    /// viterbi解码算法，求解最优隐状态序列
    /// \param out
    /// \param x
    /// \param n_x
    /// \return
    double viterbi(int *out, int *x, int n_x);

    /// 预测下一个观测值。使用后验概率分布的算法。
    /// \param out [输出] 预测的每个观测状态的概率值
    /// \param x [输入] 已知的观测序列
    /// \param n_x [输入] 已知观测序列的长度
    virtual double predict_by_posterior(double *out, int *x, int n_x);

    /// 预测下一个观测值。使用viterbi的算法。
    /// \param out
    /// \param x
    /// \param n_x
    /// \return
    virtual double predict_by_viterbi(double *out, int *x, int n_x);

    /// 预测首个观测值。使用初始概率计算
    /// \param out
    virtual void predict_first(double *out);

    /// 训练时最小观测序列长度，序列长度小于这个值的观测序列无法进行训练
    /// \param value
    void set_minimum_obs(int value) { this->minimum_obs = MIN(3, value); };

    int get_minimum_obs() { return this->minimum_obs; };

protected:
    /// 前向算法
    /// \param x
    /// \param n_x
    /// \param PI
    /// \param A
    /// \return
    double forward(int x_pos, int n_x, double *PI, double **A);

    /// 后向算法
    /// \param x
    /// \param n_x
    /// \param PI
    /// \param A
    /// \return
    double backward(int x_pos, int n_x, double *PI, double **A);

    /// 计算gamma
    /// \param x
    /// \param n
    /// \param fwdlattice
    /// \param backlattice
    /// \param gamma_sum
    void gamma(int x_pos, int n, double **fwdlattice, double **backlattice, double *gamma_sum);

    void xi(int x_pos, int n, double **fwdlattice, double **backlattice, double **xi_sum);

    virtual double emmit_pdf(int stat, int obs, int t = -1);

    void bounded();

};


#endif //BKT_HMM_H
