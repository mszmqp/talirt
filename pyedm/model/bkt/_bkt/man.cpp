//
// Created by 张振虎 on 2018/11/6.
//

#include "bkt.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int test_data() {
    int n_stat = 2, n_obs = 2;


    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};

    double A[] = {1, 0,
                  0.4, 0.6};

    double A_LOW[] = {1, 0, 0, 0};
    double A_UPPER[] = {1, 0, 1, 1};

    double B[] = {0.8, 0.2,
                  0.2, 0.8};
    double B_LOW[] = {0.7, 0, 0, 0.7};
    double B_UPPER[] = {1, 0.3, 0.3, 1};
    StandardBKT hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.set_bound_pi(PI_LOW, PI_UPPER);
    hmm.set_bound_a(A_LOW, A_UPPER);
    hmm.set_bound_b(B_LOW, B_UPPER);

    int x = 0, n_x = 0, i = 0, j = 0;
    int *response = new int[50000];
    int *lengths = new int[50000];
    std::string pre_stu, cur_stu, item, kn, line;

    std::ifstream infile;
    infile.open("/Users/zhangzhenhu/Documents/projects/talirt/train_sample.txt");
    while (getline(infile, line)) {
        std::istringstream iss(line);
        iss >> x >> cur_stu >> item >> kn;
        x -= 1;
        response[i] = x;
        i++;
        if (cur_stu != pre_stu && !pre_stu.empty()) {
            lengths[j] = n_x;
            j++;
            n_x = 0;

        }
        n_x += 1;
        pre_stu = cur_stu;


    }
    lengths[j] = n_x;
    j++;

    hmm.fit(response, lengths, j);


    cout << "PI" << endl;
    print1D(hmm.PI, 2);
    cout << "A" << endl;
//    print2D(hmm.A, n_stat, n_stat);
    cout << "B" << endl;
//    print2D(hmm.B, n_stat, n_obs);

    cout << "predict next observation" << endl;
    double next[2] = {0};
    double start[] = {0.955115, 0.0444221};
    double transition[] = {1, 0, 0.0715796, 0.92842};
    double emission[] = {0.872745, 0.127255, 0.3, 0.7};
    hmm.predict_by_posterior(next, response, lengths[0]);
    print1D(next, 2);

    return 0;
}


void test_viterbi() {

    double A[] = {0.5, 0.2, 0.3,
                  0.3, 0.5, 0.2,
                  0.2, 0.3, 0.5};
    double B[] = {0.5, 0.5,
                  0.4, 0.6,
                  0.7, 0.3};
    double PI[] = {0.2, 0.4, 0.4};
    int obs[3] = {0, 1, 0};
    int out[3] = {0, 0, 0};
    StandardBKT sbkt(3, 2);
    sbkt.init(PI, A, B);
    sbkt.viterbi(out, obs, 3);

    print1D<int>(out, 3);

    double post[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
//    sbkt.posterior_distributed(post, obs, 3);

//    print1D<int>(out,3);
    cout << endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << post[i * 3 + j] << " ";
        }
        cout << endl;
    }


}

void test_toy_data() {

    int x[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0,0,1,1,1};
    int n_x = getArrayLen(x);

    // 0 不会，1会 ；0做错，1做对
//    double A[] = {0.4, 0.6, 0, 1};
//    double A_LOW[] = {0, 0, 0, 1};
//    double A_UPPER[] = {1, 1, 0, 1};
//
//    double B[] = {0.8, 0.2, 0.2, 0.8};
//
//    double B_LOW[] = {0.7, 0, 0, 0.7};
//    double B_UPPER[] = {1, 0.3, 0.3, 1};
//
//    double PI[2] = {0.5, 0.5};
//    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};

    // 0 会 1不会 0做对，1做错
    for (int i = 0; i < n_x; ++i) {
        x[i] = 1-x[i];
    }
    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};

    double A[] = {1, 0, 0.4, 0.6};
    double A_LOW[] = {1, 0, 0, 0};
    double A_UPPER[] = {1, 1, 1, 1};
    double B[] = {0.8, 0.2, 0.2, 0.8};

    double B_LOW[] = {0,0,0,0};
    double B_UPPER[] = {1, 0.3, 0.3, 1};
    // pLo=0.00000000, pT=0.16676161, pS=0.00044059, pG=0.00038573.
    // Overall loglikelihood, actually, goes up from 9.3763477 to 10.4379501 in 3 iterations.



    StandardBKT hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.set_bound_pi(PI_LOW, PI_UPPER);
    hmm.set_bound_a(A_LOW, A_UPPER);
    hmm.set_bound_b(B_LOW, B_UPPER);
//    int x[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
//    int lengths[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3};
    int lengths[] = {20,6};

    hmm.fit(x, lengths, getArrayLen(lengths));
//    printf("%f", hmm.log_likelihood);
    cout << "================" << endl;
    cout << "PI" << endl;
    print1D(hmm.PI, 2);
    cout << "A" << endl;
    hmm.A.print();
    cout << "B" << endl;
    hmm.B.print();

}

int main() {
    test_toy_data();
//    test_viterbi();
    return 0;
    test_data();

    double ar[] = {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                   0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                   1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//    double lb[]={0, 0, 0, 0, 0, 0, 0};
//    double ub[7]={1,1,1,1,1,1,1};
//    print1D(ub,7);
//    projectsimplexbounded(ar,lb,ub,7);






    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};
// 0 会 1不会
//    double A[] = {1, 0, 0.4, 0.6};
//    double A_LOW[] = {1, 0, 0, 0};
//    double A_UPPER[] = {1, 0, 1, 1};
//    double B[] = {0.8, 0.2, 0.2, 0.8};

//    double B_LOW[] = {0.7, 0, 0, 0.7};
//    double B_UPPER[] = {1, 0.3, 0.3, 1};


    // 0 不会，1会 ；0做错，1做对
    double A[] = {0.4, 0.6, 0, 1};
    double A_LOW[] = {0, 0, 0, 1};
    double A_UPPER[] = {1, 1, 0, 1};

    double B[] = {0.8, 0.2, 0.2, 0.8};

    double B_LOW[] = {0.7, 0, 0, 0.7};
    double B_UPPER[] = {1, 0.3, 0.3, 1};

    StandardBKT hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.set_bound_pi(PI_LOW, PI_UPPER);
    hmm.set_bound_a(A_LOW, A_UPPER);
    hmm.set_bound_b(B_LOW, B_UPPER);
//    int x[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
//    int lengths[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3};
    int lengths[] = {4};
    int x[] = {1, 1, 1, 0};
    hmm.fit(x, lengths, getArrayLen(lengths));
    return 0;
}
