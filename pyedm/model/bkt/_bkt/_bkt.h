//
// Created by 张振虎 on 2019/1/16.
//

#ifndef HMM_IRTBKT_H
#define HMM_IRTBKT_H

#include "utils.h"
#include <math.h>
#include "_hmm.h"


class StandardBKT : public HMM {
public:
    StandardBKT(int n_stat = 2, int n_obs = 2) : HMM(n_stat, n_obs) {};

    void predict_by_posterior(double *out, int *x, int n_x) override;

private:
    double emmit_pdf(int stat, int obs, int item_pos) override {
//        std::cout << "x_pos " << x_pos << std::endl;
        return this->B[stat][obs];
    };
};

#pragma pack(8)
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

    void set_items_info(double *items, int length);
//    void set_items_info(double items[], int length);

    void set_items(int items_id[], int length);

    void predict_by_posterior(double *out, int *x, int n_x, int item_id);

    void predict_by_viterbi(double *out, int *x, int n_x, int item_id);

    void predict_first(double *out, int item_id);

private:
    double emmit_pdf(int stat, int obs, int t) override;

    double emmit_pdf_ex(int stat, int obs, int item_id);

    void predict_by_posterior(double *out, int *x, int n_x) override {};

    void predict_by_viterbi(int *out, int *x, int n_x) {};

    void predict_first(double *out){};
};


#endif //HMM_IRTBKT_H
