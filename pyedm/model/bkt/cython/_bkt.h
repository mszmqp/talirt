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
    StandardBKT(int n_obs = 2, int n_stat = 2) : HMM(n_obs, n_stat) {};
};




class Item {
public:

    double slop;// discrimination;
    double intercept; // difficulty;
    double guess;

    double sigmoid(double z) {
        return 1.0 / (1 + exp(-z));
    }

    double irt(double theta, int m = 2) {

        double z = 0;
        switch (m) {
            case 1:
                z = theta - intercept;
                break;
            default:
                z = this->slop * theta - intercept;
                break;
        }

        if (m == 3) {
            return sigmoid(z) + guess;

        } else {
            return sigmoid(z);
        }

    }
};


class IRTBKT : public HMM {


private:
    Item *items=NULL;
    int *items_id=NULL;
    void set_bound_b(double lower[] = NULL, double upper[] = NULL);
    void get_b(double *out) {};

public:
    ///
    /// \param n_stat 隐状态的数量
    /// \param n_obs  观测状态的数量
    IRTBKT(int n_stat = 2, int n_obs = 2);

    ~IRTBKT();

    void set_items_param(double items[], int item_size);

    void set_items(int items_id[]);


private:
    double emmit_pdf(int x_pos, int stat, int obs);

};


#endif //HMM_IRTBKT_H
