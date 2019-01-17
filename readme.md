
# 数据集

kdd cup

http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp

http://neuron.csie.ntust.edu.tw/homework/98/NN/KDDCUP2010/Dataset/



# pymc3的问题
## 内存问题

参数太多或者抽样太多都会导致内存问题。

内存出题的情景有二：
1. 多进程时（njobs>1）， 进程间通信传输数据过大
2. 单进程时（njobs>1），太大也会超过内存限制

解决方法
1. pymc3.sample(discard_tuned_samples=False),必须设置为discard_tuned_samples=False
2. 修改pymc3/sampling.py:664

```python
def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=None):
                 ...
                 
                 
                 
    try:
        step.tune = bool(tune)
        for i in range(draws):
            if i == tune:
                step = stop_tuning(step)
            if step.generates_stats:
                point, states = step.step(point)
                if strace.supports_sampler_stats:
                    if i>= tune: # 增加行
                        strace.record(point, states)
                else:
                    if i>=tune: # 增加行
                        strace.record(point)
            else:
                point = step.step(point)
                if i>=tune: # 增加行
                    strace.record(point)
            yield strace                 
                 
                 
```

## 性能问题

pymc3效率比较差，而且还伴随内存问题。主要导致原因是其每次抽样的样本都要记录下来，这样有两个不良影响：
1. 效率差，每次记录都要消耗时间
2. 占用内存大，在变量较多时，记录太多消耗内存巨大。

解决方法：

自己实现一个backends，抛弃前期burn-in的样本，有效样本只保留累计值和数量，
最后求个平均值就行了，毕竟一般我们也只是要抽样的平均值（期望）


## 缺失值问题

学生答题有缺失值，对于观测变量缺失值的情况，pymc3是支持缺失值的http://docs.pymc.io/notebooks/getting_started中有一段

	Missing values are handled transparently by passing a MaskedArray or a pandas.
	DataFrame with NaN values to the observed argument when creating an observed stochastic random variable. 
	Behind the scenes, another random variable, disasters.missing_values is created to model the missing values.
	 All we need to do to handle the missing values is ensure we sample this random variable as well.

	Unfortunately because they are discrete variables and thus have no meaningful gradient, 
	we cannot use NUTS for sampling switchpoint or the missing disaster observations. 
	Instead, we will sample using a Metroplis step method, which implements adaptive Metropolis-Hastings, 
	because it is designed to handle discrete values. 
	PyMC3 automatically assigns the correct sampling algorithms.


**但是实验发现，这么搞抽样会非常慢**

## burn-in 数量

sampler默认是nuts，经实验 burn-in 1000和10000没区别不大。

# 贝叶斯知识追踪

### Standard knowledge tracing

A.T.Corbett, and J.R.Anderson. (1995). 
Knowledge tracing: Modeling the acquisition of procedural knowledge. 
User Modeling and User-Adapted Interaction, 4,253278. 3, 5, 30, 33, 38, 40, 50, 51, 65, 78, 81, 84, 129

[knowledge-tracing-modeling-the-acquisition-of-proc.pdf](https://slideheaven.com/queue/knowledge-tracing-modeling-the-acquisition-of-procedural-knowledge.html)


###BKT- EM (Expectation Maximization) [7]

K.Chang, J.Beck, J.Mostow, and A.Corbett.(2006). A bayes net toolkit for student modeling in intelligent tutoring systems. 
In Proceedings of International Conference on Intelligent Tutoring Systems (ITS 2006), 
104113, Springer. 31, 32, 33, 36, 37, 38, 40, 50, 51, 54, 65, 78, 81, 92

[A bayes net toolkit for student modeling in intelligent tutoring systems](https://www.cs.cmu.edu/~listen/pdfs/ChangBeckMostowCorbett.2006.ITS.BNT-SM.pdf)

### BKT-BF (Brute Force) [2]


R.S.Baker, A.T.Corbett, and V.Aleven. (2008). More accurate student modeling through contextual estimation of 
slip and guess probabilities in bayesian knowledge tracing. 
In Proceedings of the 9th International Conference on Intelligent Tutoring Systems, 
ITS 08, 406415, Springer- Verlag, Berlin, Heidelberg. 32, 33, 36,37, 38, 40, 50, 51, 54, 65, 78, 81, 92


### BKT-PPS (Prior Per Student) 
[Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing 2010](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.1943&rep=rep1&type=pdf)


[Using HMMs and bagged decision trees to leverage rich features of user and skill from an intelligent tutoring system dataset](http://pslcdatashop.org/KDDCup/workshop/papers/pardos_heffernan_KDD_Cup_2010_article.pdf)

Z.A.Pardos, and N.T.Heffernan.(2010). 
In Proceedings of the KDD Cup 2010 Workshop on Improving Cognitive Models with Educational Data Mining,
Washington, DC, USA. ix, 25, 34, 36, 37, 40, 153, 154




### BKT-CGS (Contextual Guess and Slip) 


R.S.J.Baker, A.T.Corbett, S.M.Gowda, A.Z.Wagner, B.A.MacLaren, L.R.Kauffman, A.P.Mitchell, and S.Giguere.(2010). 
Contextual slip and prediction of student performance after use of an intelligent tutor. 
In the 18th International Conference on User Modeling, Adaption and Personalization (UMAP 2010), 
Lecture Notes in Computer Science, 5263, Springer. 35, 36


W.J.Hawkins, N.T.Heffernan, S.Ryan, and J.D.Baker.(2011) 
Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities.Springer-Verlag Berlin Heidelberg.

[Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities](http://www.upenn.edu/learninganalytics/ryanbaker/paper_143.pdf)


Ensembling predictions of student knowledge within intelligent tutoring systems. (2011)

R.S.J.Baker,Z.Pardos, S.Gowda, B.Nooraei and H.Heffernan.. 
In J. Konstan, R. Conejo, J. Marzo & N. Oliver, eds., 
User Modeling, Adaption and Personalization, 
vol. 6787 of Lecture Notes in Computer Science, 1324, Springer Berlin / Heidelberg. 32, 36, 37, 52


Factorization techniques for predicting student performance. (2011). 

N.Thai-Nghe, L.Drumond, T.Horvath, A.Krohn-Grimberghe, A.Nanopoulos, and L.Schmidt-Thieme. 
In O.C. Santos and J.G. Boticario,eds., 
Educational Recommender Systems and Technologies: Practices and Challenges (ERSAT 2011),IGI Global. 7


Multi-relational factorization models for predicting student performance. (2011). 

N.Thai-Nghe, L.Drumond, T.Horvath, and L.Schmidt-Thieme. 
In Proceedings of the KDD 2011 Workshop on Knowledge Discovery in Educational Data (KDDinED 2011). 
Held as part of the 17th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.


Personalized forecasting student performance(2011)
N.Thai-Nghe, T.Horvath, and L.Schmidt-Thieme. 
In Proceedings of the 11th IEEE International Conference on Advanced Learning Technologies (ICALT 2011), IEEE Computer Society,Athens, GA, USA. 7

Matrix and tensor factorization for predicting student performance.(2011)

N.Thai-Nghe, L.Drumond, T.Horvath, A.Nanopoulos, and L.Schmidt- Thieme. 
In Proceedings of the 3rd International Conference on Computer Supported Education (CSEDU 2011). Best Student Paper Award, 69 78, Noordwijkerhout, the Netherlands. 7, 8


Factorization models for forecasting student performance.(2011). 
N.Thai-Nghe, T.Horvath, and L.Schmidt-Thieme.  In Pechenizkiy, M., Calders, T., Conati, C.,Ventura, S., Romero , C., and Stamper, J. (Eds.). Proceed- ings of the 4th International Conference on Educational Data Mining (EDM 2011), 11 20, Eindhoven, theNetherlands. 8


Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities(2011)

W.J.Hawkins, N.T.Heffernan, S.Ryan, and J.D.Baker
.Springer-Verlag Berlin Heidelberg.


Comparison of methods to trace multiple subskills: Is LR-DBN best?(2012)

Y.Xu and J.Mostow. 
In Proceedings of the 5th international conference on educational data mining (pp. 4148)



### PC-BKT (EP)

[Predicting Students’ Performance on Intelligent Tutoring System - Personalized Clustered BKT (PC-BKT) Model.pdf](https://www.amrita.edu/system/files/publications/predicting-students-performance-on-intelligen-tutoring-system-personalized-clustered-bkt-pc-bkt-model.pdf)


## papers

[Ensembling Models of Student Knowledge in Educational Software.pdf](http://www.upenn.edu/learninganalytics/ryanbaker/PBGH-SIGKDDExp.pdf)


[Limits to Accuracy- How Well Can We Do at Student Modeling.pdf](http://www.educationaldatamining.org/EDM2013/papers/rn_paper_04.pdf)

[Feature Engineering and Classifier Ensemble for KDD Cup 2010.pdf](http://pslcdatashop.org/KDDCup/workshop/papers/kdd2010ntu.pdf)

[A Spectral Learning Approach to Knowledge Tracing.pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.353.4762&rep=rep1&type=pdf)

[Structured Knowledge Tracing Models for Student Assessment on Coursera.pdf](https://www.cs.cmu.edu/~zhitingh/data/lats16structured.pdf)

[Introducing Item Difficulty to the Knowledge Tracing Model.pdf](https://web.cs.wpi.edu/~nth/pubs_and_grants/papers/2011/UMAP/Pardos%20Introducing%20Item%20Difficulty.pdf)

[On the Performance Characteristics of Latent-Factor and Knowledge Tracing Models.pdf](https://graphics.ethz.ch/~sobarbar/papers/Kli15/Kli15a.pdf)

[Revisiting and Extending the Item Difficulty Effect Model .pdf](http://ceur-ws.org/Vol-1009/0106.pdf)

[基于贝叶斯知识跟踪模型的慕课学生评价.pdf](http://net.pku.edu.cn/dlib/MOOCStudy/MOOCPaper/ours/201410-201.pdf)


[Individualized Bayesian Knowledge Tracing Models.pdf](https://www.cs.cmu.edu/~ggordon/yudelson-koedinger-gordon-individualized-bayesian-knowledge-tracing.pdf)


[Applications of Bayesian Knowledge Tracing to the Curation of Education videos.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-98.pdf)



C++ 版本的BKT
https://github.com/myudelson/hmm-scalable

实验数据
http://pslcdatashop.web.cmu.edu/KDDCup/

国际教育领域数据挖掘协会

http://educationaldatamining.org/resources/
https://github.com/IEDMS

卡内基梅隆大学的贝叶斯学生模型工具

http://www.cs.cmu.edu/~listen/BNT-SM/




**Feature-Aware Student knowledge Tracing**


This is the repository of FAST, an efficient toolkit for modeling time-changing student performance 
([González-Brenes, Huang, Brusilovsky et al, 2014] 
(http://educationaldatamining.org/EDM2014/uploads/procs2014/long%20papers/84_EDM-2014-Full.pdf)). 
FAST is alterantive to the [BNT-SM toolkit] (http://www.cs.cmu.edu/~listen/BNT-SM/), 
a toolkit that requires the researcher to design a different different Bayes Net for each feature set they want to prototype. 
The FAST toolkit is up to 300x faster than BNT-SM, and much simpler to use.

We presented the model in the 7th International Conference on Educational Data Mining (2014) 
(see [slides] (http://www.cs.cmu.edu/~joseg/files/fast_presentation.pdf) ), where it was selected as one the top 5 paper submissions.

https://github.com/ml-smores/fast

## WCRP

WCRP is a Weighted Chinese Restaurant Process model for inferring skill labels in Bayesian Knowledge Tracing.

Check out the paper for more information.

https://github.com/robert-lindsey/WCRP


## All of Knowledge Tracing

A research project on bayesian knowledge tracing. You can see the paper in progress here.

https://github.com/swirepe/AllKT




# 深度学习知识追踪

论文
http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf
https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf

## Deep Knowledge Tracing Implementation
This repository contains our implementation of the Deep Knowledge Tracing (DKT) model which we used in our paper.

DKT is a recurrent neural network model designed to predict students' performance. 
The authors of the DKT paper used a Long-Term Short-Term (LSTM) network in the paper but only published code for a simple recurrent neural network. 
In our paper we compare various enhanced flavors of Bayesian Knowledge Tracing (BKT) to DKT. 
To ensure a fair comparison to DKT, we implemented our own LSTM variant of DKT. This repository contains our implementation.

https://github.com/mmkhajah/dkt

https://github.com/lccasagrande/Deep-Knowledge-Tracing

https://github.com/davidoj/deepknowledgetracingTF


## Going Deeper with Deep Knowledge Tracing - EDM-2016

Source code and data sets for Going Deeper with Deep Knowledge Tracing

https://github.com/siyuanzhao/2016-EDM

## DKT+

It is the DKT+ model implemented in python3 and tensorflow1.2

This is the repository for the code in the paper DKT+: Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization

https://github.com/ckyeungac/deep-knowledge-tracing-plus