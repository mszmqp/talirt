========================
贝叶斯知识追踪
========================


论文
========================

* Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge
    BKT首次提出，基础版

* Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing
    引入人的先验掌握概率，(Prior Per Student,PPS)

* KT-IDEM: Introducing Item Difficulty to the Knowledge Tracing Model
    引入题目难度


* Individualized Bayesian Knowledge Tracing Models
    参数估计


基础版本
========================

:论文: Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge


其实就是标准的HMM模型进行建模。在一个知识点下，学生作答序列是一个隐马尔科夫序列。
其中学生掌握、未掌握当前知识点作为HMM的隐状态，作答结果（对、错）是观测序列，学生的学习率（从不会到会的概率）是转移概率矩阵。

缺点：
    建模的时知识点而不是学生，也就是说这些参数描述的当前知识点的特征，与学生无关，与题目无关。

适用场景：
    知识点的题目是固定的，作答顺序也是固定的。对知识点进行建模。


人的个性化
========================

个性化先验
-------------------------------------------------------

:论文: Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing





引入题目难度
========================

:论文: Introducing Item Difficulty to the Knowledge Tracing Model

发射矩阵中guess和slip参数可以看做是和题目相关，guess越大，说明题目越简单；反之，表示题目较难；slip同理。
在HMM链中，每个时刻t，都对应着一道题目，这样的话，为每个题目训练独立的guess和slip参数，就可以实现题目维度上的区别对待。
在标准的HMM中，每个时刻t的发射概率参数都是一样的，在这里就变成不一样的了，并且是和题目绑定。
要想训练这样的模型，要求有很多个学生的作答序列才可以，并且最好每个学生的作答序列是同样的题目集合（保证每个题目都有很多人作答），
但不必作答顺序一致。


参数估计
============================

:论文: Individualized Bayesian Knowledge Tracing Models




数据集
============================


KDD cup 2010