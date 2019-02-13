//
// Created by 张振虎 on 2019/1/22.
//

#include "TrainHelper.h"


TrainHelper::TrainHelper(int n_stat, int n_obs, int model_type) {
    this->n_obs = n_obs;
    this->n_stat = n_stat;
    this->model_type = model_type;
    this->models = NULL;
    this->pi = NULL;
    this->pi_lower = NULL;
    this->pi_upper = NULL;
    this->a = NULL;
    this->a_lower = NULL;
    this->a_upper = NULL;
    this->b = NULL;
    this->b_lower = NULL;
    this->b_upper = NULL;
    this->item_count = 0;
    this->items_info = 0;
}

TrainHelper::~TrainHelper() {
    delete[] this->models;

}

void TrainHelper::init(double *pi, double *a, double *b) {
    this->pi = pi;
    this->a = a;
    this->b = b;

}

void TrainHelper::set_bound_pi(double *lower, double *upper) {
    this->pi_lower = lower;
    this->pi_upper = upper;
};

void TrainHelper::set_bound_a(double *lower, double *upper) {
    this->a_lower = lower;
    this->a_upper = upper;
};

void TrainHelper::set_bound_b(double *lower, double *upper) {
    this->b_lower = lower;
    this->b_upper = upper;
};

void TrainHelper::set_items_info(double items[], int length) {
    this->items_info = items;
    this->item_count = length;

}

void TrainHelper::fit(int trace[], int group[], int x[], int length, int item[], int max_iter, double tol) {


    int i, j;

    int trace_num; // 一共多少个不同的trace，也就是需要训练多少个模型
    // 统计每个trace id 的数量
    int *trace_ncount = unique_counts<int>(trace, length,trace_num);
//    int trace_count = getArrayLen<int>(trace_ncounts);
    this->model_count = trace_num;

    if (this->model_type == 1) {
//        assert(getArrayLen(train_data) == length * 3);

        this->models = (HMM **) calloc((size_t) length, sizeof(StandardBKT *));
        for (i = 0; i < trace_num; ++i) {
            this->models[i] = new StandardBKT(this->n_stat, this->n_obs);

        }
    } else if (this->model_type == 2) {
//        assert(getArrayLen(train_data) == length * 4);
        assert(item != NULL);
        assert(this->items_info != NULL);
        this->models = (HMM **) calloc((size_t) length, sizeof(IRTBKT *));
        for (i = 0; i < trace_num; ++i) {
            this->models[i] = new IRTBKT(this->n_stat, this->n_obs);

        }
    }

    for (j = 0; j < trace_num; ++j) {
        this->models[j]->init(this->pi, this->a, this->b);
        this->models[j]->set_bound_pi(this->pi_lower, this->pi_upper);
        this->models[j]->set_bound_a(this->a_lower, this->a_upper);
        this->models[j]->set_bound_b(this->b_lower, this->b_upper);
        if (this->model_type == 2) {
            ((IRTBKT *) this->models[j])->set_items_info(this->items_info, this->item_count);
        }
    }


    int trace_pos = 0, cur_x_length = 0;
    int *group_ncount = NULL;
    int *cur_x;
//    std::cout<<"trace_count "<<trace_num<<std::endl;
    int group_num;
    for (i = 0; i < trace_num; ++i) {

        //trace_ncount[i]; // 当前trace的训练数据长度
        group_num = 0; // 当前trace 包含多少个观测序列
        cur_x = x + trace_pos; // 当前训练数据的位置
        cur_x_length = trace_ncount[i]; // 当前训练数据的长度
        group_ncount = unique_counts(group + trace_pos, cur_x_length,group_num); //当前训练数据的分组
//        std::cerr <<
//        if(group_num>1){
//            print1D<int>(group_ncount,group_num);
//
//        }
        if (this->model_type == 2) {

            ((IRTBKT *) this->models[i])->set_obs_items(item + trace_pos, cur_x_length);
        }
        this->models[i]->fit(cur_x, group_ncount, group_num, max_iter, tol);


        trace_pos += trace_ncount[i]; // 下一个trace
        free(group_ncount);
    }

    free(trace_ncount);

}