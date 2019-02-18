//
// Created by 张振虎 on 2019/1/22.
//

#ifndef HMM_TRAINHELPER_H
#define HMM_TRAINHELPER_H

//#include "_hmm.h"
#include "bkt.h"

/*
 * 批量训练辅助工具。
*/
class TrainHelper {


private:
    int n_stat;
    int n_obs;
    int model_type;

    double *pi;
    double *pi_lower;
    double *pi_upper;
    double *a;
    double *a_lower;
    double *a_upper;
    double *b;
    double *b_lower;
    double *b_upper;


    double *items_info;
    int item_count;
    bool free_items_info;

    bool free_param;
    bool free_bound_pi;
    bool free_bound_a;
    bool free_bound_b;

public:
    HMM **models;
    int model_count;

    ///
    /// \param n_stat 隐状态的数量
    /// \param n_obs  观测状态的数量
    /// \param model_type  模型的类型，1-标准bkt；2-IRT变种BKT
    TrainHelper(int n_stat = 2, int n_obs = 2, int model_type = 1);

//    ~HMM();
    ~TrainHelper();

    /// 初始化设置参数
    /// \param pi 初始概率
    /// \param a 转移概率
    /// \param b 发射概率
    void init(double *pi = NULL, double *a = NULL, double *b = NULL,bool copy = false);

    /// 设置初始概率PI的约束
    /// \param lower 对应位置值的下限。一维数组。
    /// \param upper 对应位置值的上限。一维数组。
    void set_bound_pi(double *lower = NULL, double *upper = NULL,bool copy = false);

    /// 设置转移概率矩阵A的约束
    /// \param lower 对应位置值的下限。用一维连续空间表示二维数组。
    /// \param upper 对应位置值的上限。用一维连续空间表示二维数组。
    void set_bound_a(double *lower = NULL, double *upper = NULL,bool copy = false);

    /// 设置发射概率矩阵B的约束
    /// \param lower 对应位置值的下限。用一维连续空间表示二维数组。
    /// \param upper 对应位置值的上限。用一维连续空间表示二维数组。
    void set_bound_b(double *lower = NULL, double *upper = NULL,bool copy = false);


    void set_items_info(double *items, int length,bool copy = false);

    /// 估计模型参数
    /// \param x 观测值序列。可以是多个不同的观测序列堆在一起，lengths数组记录每个观测序列的长度。len(x)=sum(lengths)
    /// \param lengths 每个观测序列的长度。
    /// \param n_lengths lengths数组的长度，也就是有多少个不同的观测序列。
    /// \param max_iter 最大迭代次数
    /// \param tol 收敛的精度，当两轮迭代似然值的差值小于tol时，结束迭代。
    /// \return
    void
    fit(int trace[], int group[], int x[], int length, int item[] = NULL, int max_iter = 100,
        double tol = 1e-2);

};


#endif //HMM_TRAINHELPER_H
