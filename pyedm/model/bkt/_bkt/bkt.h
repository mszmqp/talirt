//
// Created by 张振虎 on 2019/1/16.
//

#ifndef HMM_IRTBKT_H_
#define HMM_IRTBKT_H_

#include "utils.h"
#include <math.h>
#include "hmm.h"


class StandardBKT : public HMM {
public:
    StandardBKT(int n_stat = 2, int n_obs = 2) : HMM(n_stat, n_obs) {};

//    double predict_by_posterior(double *out, int *x, int n_x) override;

private:
//    double emmit_pdf(int stat, int obs, int item_pos) override {
////        std::cout << "x_pos " << x_pos << std::endl;
//        return this->B[stat][obs];
//    };
};

//#pragma pack(8)
typedef struct _Item {
    double slop;// discrimination;
    double intercept; // difficulty;
    double guess;
} Item;

double sigmoid(double z);

double irt(double theta, Item *item, int m);


class IRTBKT : public HMM {


private:
    Item *items;
    int items_length;
    int *items_id;
    int items_id_length;

//    void set_bound_b(double lower[] = NULL, double upper[] = NULL);
//
//    void get_b(double *out) {};

public:
    ///
    /// \param n_stat 隐状态的数量
    /// \param n_obs  观测状态的数量
    IRTBKT(int n_stat, int n_obs);

    ~IRTBKT();
    /// 设置题目参数。每个题目三个double类型参数，实际是二维数据(n,3)，n是题目数量。
    /// 但这里需要用一维连续空间保存二维数据，传入的是一维数组空间的地址。
    /// \param items 数组指针，注意，内部不会重新保存一份，所以请确保传入指针在使用期间不要被释放
    /// \param length
    void set_items_info(double *items, int length);

    /// 设置观测序列对应的题目id序列，在训练fit和预测前，都必须先传入对应的题目id序列。
    /// \param items_id 题目id序列的指针。注意，内部不会重新保存一份，所以请确保传入指针在使用期间不要被释放
    /// \param length 序列长度
    void set_obs_items(int *items_id, int length);

    /// 预测下一个观测值。使用后验概率分布的算法。
    /// \param out [输出] 预测的每个观测状态的概率值
    /// \param x [输入] 已知的观测序列
    /// \param n_x [输入] 已知观测序列的长度
    /// \param item_id [输入] 待预测下一题的题目id
    double predict_by_posterior(double *out, int *x, int n_x, int item_id);

    /// 预测下一个观测值。使用viterbi的算法。
    /// \param out  [输出] 预测的每个观测状态的概率值
    /// \param x
    /// \param n_x
    /// \param item_id  [输入] 待预测下一题的题目id
    /// \return
    double predict_by_viterbi(double *out, int *x, int n_x, int item_id);

    /// 预测首个观测值。使用初始概率计算
    /// \param out
    /// \param item_id
    void predict_first(double *out, int item_id);

    void debug();
//    double posterior_distributed(double *out, int *x, int n_x);

private:
    ///
    /// \param x
    /// \param lengths
    /// \param n_lengths
    /// \return
    FitBit **covert2fb(int *x, int *lengths, int n_lengths) override ;
    FitBit *covert2fb(int *x, int length) override ;

//    double emmit_pdf(int stat, int obs, int t) override;
    double emmit_pdf(FitBit *fb, int stat, int t) override;

    double emmit_pdf_ex(int stat, int obs, int item_id);

    //通过私有化&重写的方式使得父类的函数废弃掉
    double predict_by_posterior(double *out, int *x, int n_x) override { return 0;};

    double predict_by_viterbi(double *out, int *x, int n_x) override { return 0;};

    void predict_first(double *out) override {};
};


#endif //HMM_IRTBKT_H
