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
    this->free_param = false;
    this->free_bound_pi = false;
    this->free_bound_a = false;
    this->free_bound_b = false;
    this->free_items_info = false;
}

TrainHelper::~TrainHelper() {
    for(int i=0;i<this->model_count;i++)
    {
        delete this->models[i];
    }
    free(this->models);

    if (free_param) {
        free(this->pi);
        free(this->a);
        free(this->b);

    }
    if (this->free_bound_pi) {
        free(this->pi_lower);
        free(this->pi_upper);
    }
    if (this->free_bound_a) {
        free(this->a_lower);
        free(this->a_upper);
    }
    if (this->free_bound_b) {
        free(this->b_lower);
        free(this->b_upper);
    }
    if (this->free_items_info) {
        free(this->items_info);
    }

}

void TrainHelper::init(double *pi_ptr, double *a_ptr, double *b_ptr, bool copy) {

    assert(this->n_stat > 0 && this->n_obs > 0);

    // 先释放原来的,避免内存泄漏
    if (this->free_param) {
        free(this->pi);
        free(this->a);
        free(this->b);

    }

    if (this->model_type == 1) {
        assert(a_ptr != NULL && a_ptr != NULL && b_ptr != NULL);
    } else if (this->model_type == 2) {
        assert(a_ptr != NULL && a_ptr != NULL);
    }

    if (copy) {
        this->free_param = true;
        if (pi_ptr != NULL) {

            this->pi = Calloc(double, this->n_stat);
            cpy1D<double>(pi_ptr, this->pi, this->n_stat);
        }
        if (a_ptr != NULL) {

            this->a = Calloc(double, this->n_stat * this->n_stat);
            cpy1D<double>(a_ptr, this->a, this->n_stat * this->n_stat);
        }

        if (b_ptr != NULL) {

            this->b = Calloc(double, this->n_stat * this->n_obs);
            cpy1D<double>(b_ptr, this->b, this->n_stat * this->n_obs);
        }

//        this->PI.data = this->I_ptr;

    } else {

        this->free_param = false;

        this->pi = pi_ptr;
        this->a = a_ptr;
        this->b = b_ptr;
    }


}

void TrainHelper::set_bound_pi(double *lower, double *upper, bool copy) {
    assert(lower != NULL && upper != NULL);
    assert(this->n_stat > 0 && this->n_obs > 0);

    // 先释放原来的,避免内存泄漏
    if (free_bound_pi) {
        free(this->pi_lower);
        free(this->pi_upper);
    }
    if (copy) {
        this->free_bound_pi = true;

        this->pi_lower = Calloc(double, this->n_stat);
        cpy1D<double>(lower, this->pi_lower, this->n_stat);

        this->pi_upper = Calloc(double, this->n_stat);
        cpy1D<double>(upper, this->pi_upper, this->n_stat);


    } else {
        this->free_bound_pi = false;
        this->pi_lower = lower;
        this->pi_upper = upper;
    }


};

void TrainHelper::set_bound_a(double *lower, double *upper, bool copy) {
    assert(lower != NULL && upper != NULL);
    assert(this->n_stat > 0 && this->n_obs > 0);
    if (this->free_bound_a) {
        free(this->a_lower);
        free(this->a_upper);
    }

    if (copy) {
        this->free_bound_a = true;
        int size = this->n_stat * this->n_stat;
        this->a_lower = Calloc(double, size);
        cpy1D<double>(lower, this->a_lower, size);

        this->a_upper = Calloc(double, size);
        cpy1D<double>(upper, this->a_upper, size);

    } else {
        this->free_bound_a = false;
        this->a_lower = lower;
        this->a_upper = upper;
    }


};

void TrainHelper::set_bound_b(double *lower, double *upper, bool copy) {
    assert(lower != NULL && upper != NULL);
    assert(this->n_stat > 0 && this->n_obs > 0);
    if (this->free_bound_b) {
        free(this->b_lower);
        free(this->b_upper);
    }


    if (copy) {
        this->free_bound_b = true;
        int size = this->n_stat * this->n_obs;
        this->b_lower = Calloc(double, size);
        cpy1D<double>(lower, this->b_lower, size);

        this->b_upper = Calloc(double, size);
        cpy1D<double>(upper, this->b_upper, size);

    } else {
        this->free_bound_a = false;
        this->b_lower = lower;
        this->b_upper = upper;
    }

};

void TrainHelper::set_items_info(double *items, int length, bool copy) {

    assert(items != NULL && length > 0);
    if (this->free_items_info) {
        free(this->items_info);
    }
    if (copy) {
        this->free_items_info = true;
        this->items_info = Calloc(double, length);
        cpy1D<double>(items, this->items_info, length);


    } else {
        this->free_items_info = false;
        this->items_info = items;
        this->item_count = length;
    }

}

void TrainHelper::fit(int trace[], int group[], int x[], int length, int item[], int max_iter, double tol) {


    int i, j;

    int trace_num; // 一共多少个不同的trace，也就是需要训练多少个模型
    // 统计每个trace id 的数量
    int *trace_ncount = unique_counts<int>(trace, length, trace_num);
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
        group_ncount = unique_counts(group + trace_pos, cur_x_length, group_num); //当前训练数据的分组
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