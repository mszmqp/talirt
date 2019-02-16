//
// Created by 张振虎 on 2018/10/31.
//
#include <iostream>
//#include <gsl/gsl_matrix.h>


#ifndef BKT_HMM_H
#define BKT_HMM_H

#include <map>
#include <string>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

#include "utils.h"

//typedef signed char NPAR;
typedef unsigned int UINT;


bool issimplexbounded(double *ar, double *lb, double *ub, int size);

void projectsimplexbounded(double *ar, double *lb, double *ub, int size);

class FitBit {
public:
    int *data;
    int *items;
    UINT data_length;
    UINT item_length;
    UINT n_obs;
    UINT n_stat;
    double *PI;
    MatrixView<double> A;
    MatrixView<double> B;

    double **fwdlattice;
    double **backlattice;
    double **gammalattice;
    double *cn;

    double *gamma_sum;
    double *gamma_sum_less;

    double **xi_sum;
    double **gamma_obs_sum;

    double log_likelihood;
    bool success;

    FitBit();

    ~FitBit();

    void set_data(int *x, UINT length, bool copy = false);

    void set_item(int *item, UINT length, bool copy = false);

    /// 必须在set_data之后调用
    /// \param n_stat
    /// \param n_obs
    void init(int n_stat, int n_obs);

    /// 必须在init之后调用
    /// \param ptr
    /// \param copy
    void set_pi(double *ptr, bool copy = false);

    void set_a(double *ptr, bool copy = false);

    void set_b(double *ptr, bool copy = false);

    void reset();

    void _print2d(double **x, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            std::cout << i;
            for (int j = 0; j < cols; ++j) {
                std::cout << " " << x[i][j];

            }
            std::cout << std::endl;
        }
    }

    void print_alpha() {
        std::cout << "alpha" << std::endl;
//        double * cn_cumsum = init1D<double>(this->n_stat);
        double cn_cumsum = 1;
        for (int i = 0; i < this->data_length; ++i) {
            std::cout << i;
            cn_cumsum *= this->cn[i];
            for (int j = 0; j < this->n_stat; ++j) {

                std::cout << " " << this->fwdlattice[i][j] * cn_cumsum;

            }
            std::cout << std::endl;
        }
    };

    void print_beta() {
        std::cout << "beta" << std::endl;
        double cn_cumsum = 1;
        for (int i = this->data_length-1; i >=0; --i) {
            std::cout << i;
            cn_cumsum *= this->cn[i];
//            std::cout << " cn:" <<this->cn[i] << " cn_cumsum:" << cn_cumsum;
            for (int j = 0; j < this->n_stat; ++j) {

//                std::cout << " " << this->backlattice[i][j] << " " << this->backlattice[i][j]*cn_cumsum;
                std::cout <<  " " << this->backlattice[i][j]*cn_cumsum;

            }
            std::cout << std::endl;
        }
    };

    void print_gamma() {
        std::cout << "gamma" << std::endl;
        for (int i = 0; i < this->data_length; ++i) {
            std::cout << i;
            for (int j = 0; j < this->n_stat; ++j) {

                std::cout << " " << this->gammalattice[i][j];

            }
            std::cout << std::endl;
        }
    };
    void print_xi_sum(){
        std::cout << "xi sum" << std::endl;
        print2D<double>(this->xi_sum,this->n_stat,this->n_stat);
    }
    void print_gamma_sum_less(){
        std::cout << "gamma sum less" << std::endl;
        print1D<double>(this->gamma_sum_less,this->n_stat);
    }

private:
    bool free_data;
    bool free_item;
    bool free_pi;
    bool free_a;
    bool free_b;
//    double *PI_ptr;
//    double *A_ptr;
//    double *B_ptr;
};




/*
 * HMM的估计算法中涉及到缩放因子和多序列问题，
 * 参考论文 https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
 */
class HMM {
public:
    int iter;
    double log_likelihood;
    double log_likelihood_0;
    int n_obs;
    int n_stat;
    int max_iter;

    double *PI;
    MatrixView<double> A;
    MatrixView<double> B;

    bool success;
    std::string msg;


private:
    double *PI_LOW, *PI_UPPER;
    double **A_LOW, **A_UPPER;
    double **B_LOW, **B_UPPER;

    int minimum_obs;

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
    double posterior_distributed(FitBit *fb,double *out);

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

    ///
    /// \param x
    /// \param lengths
    /// \param n_lengths
    /// \return
    virtual FitBit **covert2fb(int *x, int *lengths, int n_lengths);
    virtual FitBit *covert2fb(int *x, int length);

    /// 前向算法
    /// \param x
    /// \param n_x
    /// \param PI
    /// \param A
    /// \return
    double forward(int x_pos, int n_x, double *PI, double **A);

    void forward(FitBit *fb);

    /// 后向算法
    /// \param x
    /// \param n_x
    /// \param PI
    /// \param A
    /// \return
    double backward(int x_pos, int n_x, double *PI, double **A);

    void backward(FitBit *fb);

    /// 计算gamma
    /// \param x
    /// \param n
    /// \param fwdlattice
    /// \param backlattice
    /// \param gamma_sum
    void gamma(int x_pos, int n, double **fwdlattice, double **backlattice, double *gamma_sum);

    void gamma(FitBit *fb);

    void xi(int x_pos, int n, double **fwdlattice, double **backlattice, double **xi_sum);

    void xi(FitBit *fb);

    virtual double emmit_pdf(int stat, int obs);

    virtual double emmit_pdf(FitBit *fb, int stat, int t);

    virtual void compute_param(FitBit **fb_list, int length);

    void bounded();
    double compute_loglikehood(FitBit **fb_list, int fb_length);

};


#endif //BKT_HMM_H
