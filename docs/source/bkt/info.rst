=============================
BKT 相关资料
=============================


论文 :download:`Bayesian Knowledge Tracing, Logistic Models, and Beyond: An Overview of Learner Modeling Techniques <../../pdf/用户模型/Bayesian Knowledge Tracing, Logistic Models, and Beyond- An Overview of Learner Modeling Techniques.pdf>`
对学习建模的相关技术（贝叶斯知识追踪、基于逻辑函数的模型）进行了汇总和总结，非常值得一看。



论文集合
=============================



.. [#KT] [1995 KT] Knowledge tracing: Modeling the acquisition of procedural knowledge

    A.T.Corbett, and J.R.Anderson. (1995).
    User Modeling and User-Adapted Interaction, 4,253278. 3, 5, 30, 33, 38, 40, 50, 51, 65, 78, 81, 84, 129

    `下载地址 <https://slideheaven.com/queue/knowledge-tracing-modeling-the-acquisition-of-procedural-knowledge.html>`_

    首次提出的BKT模型，最原始版本。
    其实就是标准的HMM模型进行建模。在一个知识点下，学生作答序列是一个隐马尔科夫序列。
    其中学生掌握、未掌握当前知识点作为HMM的隐状态，作答结果（对、错）是观测序列，学生的学习率（从不会到会的概率）是转移概率矩阵。

    缺点：
        建模的时知识点而不是学生，也就是说这些参数描述的当前知识点的特征，与学生无关，与题目无关。

    适用场景：
        知识点的题目是固定的，作答顺序也是固定的。对知识点进行建模。

.. [#] [2006 BKT-EM (Expectation Maximization)] A bayes net toolkit for student modeling in intelligent tutoring systems.

    K.Chang, J.Beck, J.Mostow, and A.Corbett.(2006).

    In Proceedings of International Conference on Intelligent Tutoring Systems (ITS 2006),
    104113, Springer. 31, 32, 33, 36, 37, 38, 40, 50, 51, 54, 65, 78, 81, 92

    `下载地址 <https://www.cs.cmu.edu/~listen/pdfs/ChangBeckMostowCorbett.2006.ITS.BNT-SM.pdf>`_

    本文描述了一种对"学生在学习知识技能过程中能力状态变化"的建模方法。
    动态贝叶斯网络（DBN）提供了一种强有力的方式来表示和推理时间序列数据中的不确定性，因此非常适合模拟学生的知识。许多通用的贝叶斯网络包已经实施和分发;然而，构建DBN通常涉及复杂的编码工作。为了解决这个问题，我们引入了一个名为BNT-SM的工具。 BNT-SM输入由研究人员假设的贝叶斯网络模型的数据集和紧凑XML规范，以描述学生知识和观察到的行为之间的因果关系。 BNT-SM使用贝叶斯网络工具箱生成并执行代码来训练和测试模型[1]。与它输出的BNT代码相比，BNT-SM将使用DBN所需的代码行数减少了5倍。除了支持更灵活的模型外，我们还说明了如何使用BNT-SM来模拟知识跟踪（ KT）[2]，一种用于学生建模的既定技术。
    训练有素的DBN比原始KT代码（曲线下面积= 0.610> 0.568）更好地建模和预测学生表现，因为它估计参数的方式不同。

    本文主要介绍了如何利用BNT-SM工具包实现BKT。



.. [#]  [2008 BKT-BF (Brute Force)] More accurate student modeling through contextual estimation of slip and guess probabilities in bayesian knowledge tracing


    R.S.Baker, A.T.Corbett, and V.Aleven. (2008). More accurate student modeling through contextual estimation of
    slip and guess probabilities in bayesian knowledge tracing.
    In Proceedings of the 9th International Conference on Intelligent Tutoring Systems,
    ITS 08, 406415, Springer- Verlag, Berlin, Heidelberg. 32, 33, 36,37, 38, 40, 50, 51, 54, 65, 78, 81, 92

    本文介绍了一种利用上下文信息估计Guess和Slip参数，提高BKT效果的方法。 （PS：个人感觉实际应用价值不大）

.. [#] [2010 BKT-CGS (Contextual Guess and Slip)] Contextual slip and prediction of student performance after use of an intelligent tutor


    R.S.J.Baker, A.T.Corbett, S.M.Gowda, A.Z.Wagner, B.A.MacLaren, L.R.Kauffman, A.P.Mitchell, and S.Giguere.(2010).
    Contextual slip and prediction of student performance after use of an intelligent tutor.
    In the 18th International Conference on User Modeling, Adaption and Personalization (UMAP 2010),
    Lecture Notes in Computer Science, 5263, Springer. 35, 36

    同上。


.. [#BKT-PPS] [2010 BKT-PPS] Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing 2010

    `互联网下载地址 <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.1943&rep=rep1&type=pdf>`_

    BKT-PPS (Prior Per Student) 每个学生独立的先验，初始概率矩阵实现每个学生独立设置。

    Pardos和Heffernan 通过根据一组启发式分配来个性化掌握p(L\ :sub:`0`\ )\ :sup:`k` 的初始概率：

    - 随机
    - 通过使用基于第一学生响应正确性的两个预设值，
    - 通过使用总体百分比正确来随机选择。

    在作者考虑的问题集的很大一部分中，“先验每个学生”模型比传统BKT更适合。



.. [#KT-IDEM] [2011 KT-IDEM] Introducing Item Difficulty to the Knowledge Tracing Model

    Pardos ZA, Heffernan NT (2011)

    User Modeling, Adaption and Personalization pp 243–254

    `互联网下载地址 <https://web.cs.wpi.edu/~nth/pubs_and_grants/papers/2011/UMAP/Pardos%20Introducing%20Item%20Difficulty.pdf>`_


    KT-IDEM 在原始版本上，引入题目难度。

    发射矩阵中guess和slip参数可以看做是和题目相关，guess越大，说明题目越简单；反之，表示题目较难；slip同理。
    在HMM链中，每个时刻t，都对应着一道题目，这样的话，为每个题目训练独立的guess和slip参数，就可以实现题目维度上的区别对待。
    在标准的HMM中，每个时刻t的发射概率参数都是一样的，在这里就变成不一样的了，并且是和题目绑定。
    要想训练这样的模型，要求有很多个学生的作答序列才可以，并且最好每个学生的作答序列是同样的题目集合（保证每个题目都有很多人作答），
    但不必作答顺序一致。

.. [#] [2012 ] The Impact on Individualizing Student Models on Necessary Practice Opportunities.

    In: Yacef, K., Za ̈ıane, O.R., Hershkovitz, A., Yudelson, M., Stamper, J.C. (eds.)
    Proceedings of the 5th International Conference on Edu- cational Data Mining (EDM 2012), pp. 118-125. (2012)


.. [#] [2013] Individualized Bayesian Knowledge Tracing Models

    `本地下载地址 <../../pdf/用户模型/Bayesin knowledge tracing/Individualized Bayesian Knowledge Tracing Models.pdf>`_
    `互联网下载地址 <https://www.cs.cmu.edu/~ggordon/yudelson-koedinger-gordon-individualized-bayesian-knowledge-tracing.pdf>`_

    关于个性化BKT模型的先前工作（例如 KT_ ， BKT-PPS_ ）和 [7]_ ）描述了定义和学习学生特定参数的完全不同的方法，以及报告根本不同的性能测量。
    在本文中，我们讨论了以更系统的方式引入学生特定参数的问题。 我们以增量方式构建多个个性化BKT模型（批量添加学生特定参数）并检查每个添加对模型交叉验证性能的影响。

    我们发现，与先验学生知识相对应的BKT参数仅为BKT模型提供了边际交叉验证性能改进。 同时，学生特定的学习参数速度导致模型预测精度的显着提高。


    我们的目标是统一和扩展个性化BKT模型的先前工作。
    我们构建了个性化BKT模型的四种变体，改变了学生特定参数的数量。并且我们根据看不见的数据的预测准确性对构建的模型进行排序。

    1. Standard BKT model,
    2. Individualized BKT with student-specific p(L0),
    3. Individualized BKT with student-specific p(T),
    4. Individualized BKT with student-specific p(L0) and p(T).

    正如我们所展示的那样，我们对个性化BKT模型的实施能够切实提高预测学生在智能辅导系统中工作成功率的准确性。
    一个有趣的发现是，添加学生特定的学习概率（pLearn）比增加学生特定的初始掌握概率（pInit）更有利于模型的准确性。
    在基于逻辑回归（例如，项目反应理论）的学习实践模型的替代领域中，初始掌握概率的类比是学生熟练度，其被认为对于模型表现是关键的。
    在这些模型中，个性化学习率是否优于个性化熟练度。


.. [#] [2014 PC-BKT (EP)] Predicting Students’ Performance on Intelligent Tutoring System - Personalized Clustered BKT (PC-BKT) Model


    `下载地址 <https://www.amrita.edu/system/files/publications/predicting-students-performance-on-intelligen-tutoring-system-personalized-clustered-bkt-pc-bkt-model.pdf>`_

    智能辅导系统（ITS）是传统学习方法的补充，用于个性化学习目的，从探索简单的例子到理解错综复杂的问题。
    贝叶斯知识追踪（BKT）模型是一种用于学生建模的既定方法。最近对BKT模型的改进是BKT-PPS（先前每个学生），其为每个学生介绍了先前学习的内容。
    虽然这种方法与其他方法相比证明了改进的预测结果，但是有几个方面限制了它的用途; （a）对于学生来说，先前的学习对所有技能都是通用的，但实际上，每种技能都有所不同
    （b）不同的学生具有不同的学习能力;因此，这些学生不能被视为同质群体。

    在本文中，我们的目标是使用一个名为PC-BKT（个性化和集群）的增强型BKT模型来改进学生表现的预测，该模型为每个学生和技能提供单独的先验，
    并根据不断变化的学习能力对学生进行动态聚类。我们使用超过240,000个日志数据评估ASSISTments智能辅导数据集中未来表现的预测，
    并表明我们的模型在一般和冷启动问题中都提高了学生预测的准确性。

    存在不同的BKT变体，包括:BKT-EM（期望最大化）[7]，BKT-BF（暴力）[2]，BKT-PPS（先前每学生）[14]，BKT-CGS（语境猜测和滑动）[3]。
    虽然已经进行了几项实证研究来衡量哪种类型的学生模型更好地预测未来的表现，无论是在互动学习环境内外，不同学生模型的相对表现的结果在研究之间是相当不稳定的[4] 。


    我们回顾了某些预测模型，如潜在因子模型[18] [19]，多关系矩阵分解模型[20]，个性化预测模型[21]和张量分解模型[22] [23]。虽然这些方法与其他方法相比可以改善预测结果，但有几个方面限制了它们的用途;例如，不同的学生具有不同的学习能力，因此将所有学生视为一组是没有帮助的。

    在本文中，我们旨在使用个性化BKT模型来改进学生表现的预测，该模型支持基于不断变化的学习能力动态地聚集学生。
    我们为BKT模型引入了一个分解模型，以提供比传统模型更准确的性能。动态聚类有助于更好地处理冷启动问题。
    我们使用超过240,000个日志数据评估了ASSISTments智能辅导数据集中的预测。

    .. note::
        通过给学生聚类，把学生分组，同一组学生具有相同的参数。非常复杂，但也提供了一种思路。



.. [#] [2011 ] Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities.Springer-Verlag Berlin Heidelberg.

    W.J.Hawkins, N.T.Heffernan, S.Ryan, and J.D.Baker.(2011)
    [Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities](http://www.upenn.edu/learninganalytics/ryanbaker/paper_143.pdf)


.. [#] [2011] Ensembling predictions of student knowledge within intelligent tutoring systems. (2011)

    R.S.J.Baker,Z.Pardos, S.Gowda, B.Nooraei and H.Heffernan..
    In J. Konstan, R. Conejo, J. Marzo & N. Oliver, eds.,
    User Modeling, Adaption and Personalization,
    vol. 6787 of Lecture Notes in Computer Science, 1324, Springer Berlin / Heidelberg. 32, 36, 37, 52


.. [#] [2011] Factorization techniques for predicting student performance. (2011).

    N.Thai-Nghe, L.Drumond, T.Horvath, A.Krohn-Grimberghe, A.Nanopoulos, and L.Schmidt-Thieme.
    In O.C. Santos and J.G. Boticario,eds.,
    Educational Recommender Systems and Technologies: Practices and Challenges (ERSAT 2011),IGI Global. 7


.. [#] [2011] Multi-relational factorization models for predicting student performance. (2011).

    N.Thai-Nghe, L.Drumond, T.Horvath, and L.Schmidt-Thieme.
    In Proceedings of the KDD 2011 Workshop on Knowledge Discovery in Educational Data (KDDinED 2011).
    Held as part of the 17th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.


.. [#] [2011] Personalized forecasting student performance(2011)
    N.Thai-Nghe, T.Horvath, and L.Schmidt-Thieme.
    In Proceedings of the 11th IEEE International Conference on Advanced Learning Technologies (ICALT 2011), IEEE Computer Society,Athens, GA, USA. 7

.. [#] [2011] Matrix and tensor factorization for predicting student performance.(2011)

    N.Thai-Nghe, L.Drumond, T.Horvath, A.Nanopoulos, and L.Schmidt- Thieme.
    In Proceedings of the 3rd International Conference on Computer Supported Education (CSEDU 2011). Best Student Paper Award, 69 78, Noordwijkerhout, the Netherlands. 7, 8


.. [#] [2011] Factorization models for forecasting student performance.(2011).
    N.Thai-Nghe, T.Horvath, and L.Schmidt-Thieme.  In Pechenizkiy, M., Calders, T., Conati, C.,Ventura, S., Romero , C., and Stamper, J. (Eds.). Proceed- ings of the 4th International Conference on Educational Data Mining (EDM 2011), 11 20, Eindhoven, theNetherlands. 8


.. [#] [2011] Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities(2011)

    W.J.Hawkins, N.T.Heffernan, S.Ryan, and J.D.Baker
    .Springer-Verlag Berlin Heidelberg.


.. [#] [2012] Comparison of methods to trace multiple subskills: Is LR-DBN best?(2012)

    Y.Xu and J.Mostow.
    In Proceedings of the 5th international conference on educational data mining (pp. 4148)




其他
------------

Using HMMs and bagged decision trees to leverage rich features of user and skill from an intelligent tutoring system dataset](http://pslcdatashop.org/KDDCup/workshop/papers/pardos_heffernan_KDD_Cup_2010_article.pdf)

    Z.A.Pardos, and N.T.Heffernan.(2010).
    In Proceedings of the KDD Cup 2010 Workshop on Improving Cognitive Models with Educational Data Mining,
    Washington, DC, USA. ix, 25, 34, 36, 37, 40, 153, 154



[Ensembling Models of Student Knowledge in Educational Software.pdf](http://www.upenn.edu/learninganalytics/ryanbaker/PBGH-SIGKDDExp.pdf)


[Limits to Accuracy- How Well Can We Do at Student Modeling.pdf](http://www.educationaldatamining.org/EDM2013/papers/rn_paper_04.pdf)

[Feature Engineering and Classifier Ensemble for KDD Cup 2010.pdf](http://pslcdatashop.org/KDDCup/workshop/papers/kdd2010ntu.pdf)

[A Spectral Learning Approach to Knowledge Tracing.pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.353.4762&rep=rep1&type=pdf)

[Structured Knowledge Tracing Models for Student Assessment on Coursera.pdf](https://www.cs.cmu.edu/~zhitingh/data/lats16structured.pdf)


[On the Performance Characteristics of Latent-Factor and Knowledge Tracing Models.pdf](https://graphics.ethz.ch/~sobarbar/papers/Kli15/Kli15a.pdf)

[Revisiting and Extending the Item Difficulty Effect Model .pdf](http://ceur-ws.org/Vol-1009/0106.pdf)

[基于贝叶斯知识跟踪模型的慕课学生评价.pdf](http://net.pku.edu.cn/dlib/MOOCStudy/MOOCPaper/ours/201410-201.pdf)



[Applications of Bayesian Knowledge Tracing to the Curation of Education videos.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-98.pdf)


开源代码
=========================

C++ 版本的BKT
https://github.com/myudelson/hmm-scalable





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

WCRP
-------------------------------------
WCRP is a Weighted Chinese Restaurant Process model for inferring skill labels in Bayesian Knowledge Tracing.

Check out the paper for more information.

https://github.com/robert-lindsey/WCRP



All of Knowledge Tracing
--------------------------------------

A research project on bayesian knowledge tracing. You can see the paper in progress here.

https://github.com/swirepe/AllKT


开源实验数据
=========================
http://pslcdatashop.web.cmu.edu/KDDCup/


深度学习知识追踪
=============================

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