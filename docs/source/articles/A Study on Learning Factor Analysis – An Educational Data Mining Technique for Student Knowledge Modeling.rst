
================================================================================================================
Learning Factor Analysis 的研究
================================================================================================================

A Study on Learning Factor Analysis – An Educational Data Mining Technique for Student Knowledge Modeling



1. Introduction 导言
========================

Educational Data Mining is an inter-disciplinary field utilizes methods from machine learning, cognitive science, data mining, statistics, and psychometrics. The main aim of EDM is to construct computational models and tools to discover knowledge by mining data taken from educational settings. The increase of e-learning resources such as interactive learning environments, learning management systems (LMS), intelligent tutoring systems (ITS), and hypermedia systems, as well as the establishment of school databases of student test scores, has created large repositories of data that can be explored by EDM researchers to understand how students learn and find out models to improve their performance.

教育数据挖掘是一个跨学科领域，利用机器学习，认知科学，数据挖掘，统计学和心理测量学等方法。
EDM的主要目的是，通过挖掘从教育环境中获取的数据来发现知识来构建计算模型和工具。
互动学习环境，学习管理系统（LMS），智能辅导系统（ITS）和超媒体系统等电子学习资源的增加，以及学生考试成绩的学校数据库的建立，
创建了大量的数据存储库EDM研究人员可以探索这一点，以了解学生如何学习并找出改善其表现的模型。

Baker [1] has classified the methods in EDM as: prediction, clustering, relationship mining, distillation of data for human judgment and discovery with models. These methods are used by the researchers [1][2] to find solutions for the following goals:

Baker [1]将EDM中的方法分类为：预测，聚类，关系挖掘，人类判断数据的升华和模型发现。研究人员[1] [2]使用这些方法找到以下目标的解决方案：

1. Predicting students‟ future learning behavior by creating student models that incorporate detailed information about students‟ knowledge, meta-cognition, motivation, and attitudes.
2. Discovering or improving domain models that characterize the content to be learned and optimal instructional sequences.
3. Studying the effects of different kinds of pedagogical support that can be provided by learning software.
4. Advancing scientific knowledge about learning and learners through building computational models that incorporate models of the student, the software‟s pedagogy and the domain.


1.通过创建包含有关学生详细信息的学生模型来预测学生“未来的学习行为”知识，元认知，动机和态度。
2.发现或改进表征要学习的内容和最佳教学序列的领域模型。
3.研究可以通过学习软件提供的各种教学支持的效果。
4.通过构建包含学生模型，软件教学法和领域的计算模型，推进关于学习和学习者的科学知识。

The application areas [3] of EDM are: 1) User modeling 2) User grouping or Profiling 3) Domain modeling and 4) trend analysis.
These application areas utilize EDM methods to find solutions.
User modeling [3] encompasses what a learner knows, what the user experience is like,
what a learner‟s behavior and motivation are, and how satisfied users are with online learning.
User models are used to customize and adapt the system behaviors' to users specific needs so that the systems "say" the "right" thing at the "right" time in the "right" way [4].
This paper concerns with applying EDM method Learning factor Analysis (LFA) for User knowledge Modeling.

EDM的应用领域[3]是：1）用户建模2）用户分组或分析3）领域建模和4）趋势分析。这些应用领域利用EDM方法找到解决方案。
用户建模[3]包括学习者知道什么，用户体验是什么样的，学习者的行为和动机是什么，以及用户对在线学习的满意度。
用户模型用于根据用户的特定需求定制和调整系统行为，以便系统在“正确”的时间以“正确”的方式说出“正确”的东西[4]。
本文涉及将EDM方法学习因子分析（LFA）应用于用户知识建模。

This paper is organized as follows: section 2 lists the related works done in this research area;
section 3 explains LFA method used in this research;
section 4 describes methodology used, section 5 discusses the results and section 6 concludes the work.

本文的结构如下：第2节列出了本研究领域的相关工作;第3节解释了本研究中使用的LFA方法;
第4节描述了使用的方法，第5节讨论了结果，第6节总结了工作。


2. Literature Review 文献评论
================================

A number of studies have been conducted in EDM to find the effect of using the discovered methods on student modeling.
This section provides an overview of related works done by other EDM researchers.
在EDM中已经进行了许多研究以发现使用所发现的方法对学生建模的影响。本节概述了其他EDM研究人员所做的相关工作。

Newell and Rosenbloom[5] found a power relationship between the error rate of performance and the amount of practice.

Newell和Rosenbloom [5]发现了绩效错误率与实践量之间的权力关系.

Corbett and Anderson [6] discovered a popular method for estimating students‟ knowledge is knowledge tracing model,
an approach that uses a Bayesian-network-based model for estimating the probability that a student knows a skill based on observations of him or her attempting to perform the skill.

Corbett和Anderson [6]发现了一种评估学生的常用方法“知识是知识追踪模型，一种使用贝叶斯网络的方法基于对他或她试图执行技能的观察来估计学生知道技能的概率的基于模型。

Baker et.al [7] have proposed a new way to contextually estimate the probability that a student obtained a correct answer by guessing,
or an incorrect answer by slipping, within Bayesian Knowledge Tracing.

Baker et.al [7]提出了一种新的方法来上下文估计学生通过在贝叶斯知识追踪中猜测或通过滑动得到错误答案来获得正确答案的概率。

Koedinger et. al [8]demonstrated that a tutor unit, redesigned based on data-driven cognitive model improvements, helped students reach mastery more efficiently.
It produced better learning on the problem-decomposition planning skills that were the focus of the cognitive model improvements.

Koedinger等, [8]证明，基于数据驱动的认知模型改进重新设计的导师单元帮助学生更有效地掌握。它更好地学习了问题分解规划技能，这是认知模型改进的重点。

Stamper and Koedinger [9], presented a data- driven method for researchers to use data from educational technologies to identify and validate improvements
in a cognitive model which used Knowledge or skill components equivalent to latent variables in a logistic regression model called the Additive Factors Model (AFM)
Stamper和Koedinger [9]提出了一种数据驱动方法，供研究人员使用来自教育技术的数据来识别和验证认知模型的改进，认知模型在逻辑回归模型中使用等同于潜在变量的知识或技能组件，称为加性因子模型（AFM）。

Brent et. al [10] used learning curves to analyze a large volume of user data to explore the feasibility of using them as a reliable method for fine tuning adaptive educational system.
布伦特等。 al [10]使用学习曲线分析大量用户数据，探索将其用作微调自适应教育系统的可靠方法的可行性。

Feng et. al[11], addressed the assessment challenge in the ASSISTment system,
which is a web-based tutoring system that serves as an e-learning and e-assessment environment.
They presented that the on line assessment system did a better job of predicting student knowledge by considering how much tutoring assistance was needed,
how fast a student solves a problem and how many attempts were needed to finish a problem.

冯等人[11]，解决了ASSISTment系统中的评估挑战，这是一个基于网络的辅导系统，可作为电子学习和电子评估环境。
他们表示，在线评估系统通过考虑需要多少辅导帮助，学生解决问题的速度以及完成问题需要多少尝试来更好地预测学生的知识。

Saranya et. al [12] proposed system regards the student‟s holistic performance by mining student data and Institutional data.
Naive Bayes classification algorithm is used for classifying students into three classes – Elite, Average and Poor.

Saranya等。 [12]提出的系统通过挖掘学生数据和机构数据来考虑学生的整体表现。朴素贝叶斯分类算法用于将学生分为三类 - 精英，平均和差。

Koedinger, K.R..,[13] Professor, Human Computer Interaction Institute, Carnegie Mellon University has done lot to this EDM research. He developed cognitive models and used students interaction log taken from the Cognitive Tutors,
analyzed for the betterment of student learning process Better assessment models always result with quality education.

Koedinger, K.R..,[13]教授，Human Computer Interaction Institute, 卡内基梅隆大学人机交互研究所为这项EDM研究做了大量工作。
他开发了认知模型并使用了从认知导师那里获取的学生交互日志，分析了学生学习过程的改进。更好的评估模型总是带来优质教育。

Assessing student‟s ability and performance with EDM methods in e-learning environment for math education in school level in India has not been identified in our literature review.
Our method is a novel approach in providing quality math education with assessments indicating the knowledge level of a student in each lesson.

在我们的文献综述中，尚未确定在印度学校数学教育的电子学习环境中使用EDM方法评估学生的能力和表现。
我们的方法是一种提供高质量数学教育的新方法，其评估表明每节课中学生的知识水平。


3. Learning Factor Analysis 学习因素分析
=========================================================

User modeling or student modeling identifies what a learner knows, what the learner experience is like,
what a learner's behavior and motivation are, and how satisfied users are with e-learning.
Item Response Theory and Rash model [20] is Psychometric Methods to measure students' ability.
They lack in providing results that are easy to interpret by the users.

用户建模或学生建模识别学习者知道什么，学习者体验是什么样的，学习者的行为和动机是什么，以及用户对电子学习的满意程度。
**项目反应理论和Rash模型[20]是衡量学生能力的心理测量方法。他们缺乏提供易于用户解释的结果。**

This paper deals with identifying learners'  knowledge level (knowledge modeling) using LFA in an e-learning environment.
本文涉及在电子学习环境中使用LFA识别学习者的知识水平（知识建模）。

LFA is an EDM method for evaluating cognitive models and analysing student-tutor log data. LFA uses three components:

LFA是一种用于评估认知模型和分析学生 - 导师日志数据的EDM方法。 LFA使用三个组件：

1) Statistical model – multiple logistic regression model is used to quantify the skills.

1）统计模型 - 多元逻辑回归模型用于量化技能。

2) Human expertise- difficulty factors (concepts or KCs) defined by the subject experts (teachers):
a set of factors that make a problem-solving step more difficult for a student.

2）由专家（教师）定义的人类专业知识 - 难度因素（概念或KCs）：使学生解决问题的一系列因素变得更加困难。

3) A* search – a combinatorial search for model selection.

3）A* 搜索 - 模型选择的组合搜索。

A good cognitive model for a tutor uses a set of production rules or skills which specify how students solve problems.
The tutor should estimate the skills learnt by each student when they practice with the tutor.
The power law [5] defines the relationship between the error rate of performance and the amount of practice,
depicted by equation (1).
This shows that the error rate decreases according to a power function as the amount of practice increase.

一个良好的导师认知模型使用一套生产规则或技能来指定学生如何解决问题。
导师应该估计每个学生在与导师练习时学到的技能。幂律[5]定义了性能误差率与实践量之间的关系，由等式（1）描述。
这表明随着实践量的增加，误差率根据幂函数而减小。

.. math::

    Y=aX^b \tag{1}


Where
Y = the error rate 错误率

X = the number of opportunities to practice a skill 练习这个知识点的作答次数

a = the error rate on the first trial, reflecting the intrinsic difficulty of a skill .
第一次尝试时的错误率，反映了技能的内在难度。

b = the learning rate, reflecting how easy a skill is to learn. 学习率，反映了技能学习的容易程度。


While the power law model applies to individual skills, it does not include student effects.
In order to accommodate student effects for a cognitive model that has multiple rules, and that contains multiple students,
the power law model is extended to a multiple logistic regression model (equation 2)[24].

虽然幂律模型适用于技能个体，但它不包括学生效应。为了适应具有多个规则且包含多个学生的认知模型的学生效应，将幂律模型扩展到多重逻辑回归模型（等式2）[24]。


.. math::

    ln[ \frac{P_{ijt}}{1-P_{ijt}} ] = \sum \alpha_i X_i + \sum \beta_j Y_j + \sum \gamma_j Y_j T_{jt}   \tag{2}


Where :math:`P_{ijt}` is the probability of getting a step in a tutoring question right by the ith student's t th
opportunity to practice the jth KC;


X = the covariates for students;

Y = the covariates for skills(knowledge components);

T = the number of practice opportunities student i has had on knowledge component j;

α = the coefficient for each student, that is, the student intercept;

β = the coefficient for each knowledge component, that is, the knowledge component intercept;

γ = the coefficient for the interaction between a knowledge component and its opportunities, that is, the learning curve slope.

The model says that the log odds of :math:`P_{ijt}` is proportional to the overall “smarts” of
that student (:math:`\alpha_i`) plus the “easiness” of that KC (:math:`\beta_j`) plus the amount gained (:math:`\gamma_j`) for each practice opportunity.
This model can show the learning growth of students at any current or past moment.

该模型可以显示当前或过去时刻学生的学习成长情况。


A difficulty factor refers specifically to a property of the problem that causes student difficulties.
The tutor considered for this research has metric measures as lesson 1 which requires 5 skills
(conversion, division,multiplication, addition, and result).
These are the factors (KCs) in this tutor (Table 1) to be learnt by the students in solving the steps.
Each step has a KC assigned to it for this study.

难度因素具体指的是导致学生困难的问题的属性。考虑本研究的导师将度量指标作为第1课，
需要5种技能（转换，除法，乘法，加法和结果）。
这些是学生在解决这些步骤时要学习的导师（表1）中的因素（KCs）。每个步骤都为此研究分配了一个KC。

Table 1. Factors for the Metric measures and their values

==============  ================
Factor Names     Factor Values
==============  ================
Converion        Correct formula, Incorrect
Addition         Correct, Wrong
Multiplication   Correct, Wrong
Division         Correct, Wrong
Result           Correct, Wrong
==============  ================

The combinatorial search will select a model within the logistic regression model space.
Difficulty factors are incorporated into an existing cognitive model through a model operator called Binary Split,
which splits a skill a skill with a factor value, and a skill without the factor value. For example,
splitting production Measurement by factor conversion leads to two productions:
Measurement with the factor value Correct formula and Measurement with the factor value Incorrect.
A* search is the combinatorial search algorithm [25] in LFA.
It starts from an initial node, iteratively creates new adjoining nodes, explores them to reach a goal node.
To limit the search space, it employs a heuristic to rank each node and visits the nodes in order of this heuristic estimate.
In this study, the initial node is the existing cognitive model.
Its adjoining nodes are the new models created by splitting the model on the difficulty factors.
We do not specify a model to be the goal state because the structure of the best model is unknown.
For this paper 25 node expansions per search is defined as the stopping criterion.
AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) are two estimators used as heuristics in the search.

组合搜索将在逻辑回归模型空间中选择模型。通过称为二元分裂的模型运算符将难度因子结合到现有的认知模型中，
该运算符将技能与具有因子值的技能和不具有因子值的技能分开。例如，通过因子转换分割生产测量导致两个生产：使用因子值进行测量正确公式和使用因子值进行测量不正确。

A* 搜索是LFA中的组合搜索算法[25]。它从一个初始节点开始，迭代地创建新的相邻节点，探索它们以到达目标节点。
为了限制搜索空间，它采用启发式方法对每个节点进行排名，并按照此启发式估计的顺序访问节点。在这项研究中，初始节点是现有的认知模型。
它的相邻节点是通过在难度因子上划分模型而创建的新模型。我们没有将模型指定为目标状态，因为最佳模型的结构是未知的。
对于本文，每次搜索的25个节点扩展被定义为停止标准。 AIC（Akaike信息准则）和BIC（贝叶斯信息准则）是在搜索中用作启发式的两个估计器。

.. math::

    AIC = -2*\text{log-likelihood} + 2*\text{number of parameters} \tag{3}

.. math::
    BIC = -2*\text{log-likelihood} + \text{number of parameters} * \text{number of observations} \tag{4}


Where log-likelihood measures the fit, and the number of parameters,
which is the number of covariates in equation 2, measures the complexity.
Lower AIC & BIC scores, mean a better balance between model fit and complexity.



4. Methodology 方法
=========================


In this paper the LFA methodology is illustrated using data obtained from the Metric measures lesson of Mensuration Tutor MathsTutor[18] .
Our dataset consist of 2,247 transactions involving 60 students, 32 unique steps and 5 Skills (KCs) in students exercise log.
All the students were solving 9 problems 5 in mental problem category, 3 in simple and one in big.
Total steps involved are 32. While solving exercise problem a student can ask for a hint in solving a step.
Each data point is a correct or incorrect student action corresponding to a single skill execution.
Student actions are coded as correct or incorrect and categorized in terms of “knowledge components” (KCs) needed to perform that action.
Each step the student performs is related to a KC and is recorded as an “opportunity” for the student to show mastery of that KC.
This lesson has 5 skills (conversion, division, multiplication, addition, and result) correspond to the skill needed in a step.
Each step has a KC assigned to it for this study.

在本文中，使用从Mensuration Tutor MathsTutor [18]的Metric测量课程获得的数据来说明LFA方法。
我们的数据集包括2,247个交易，涉及60名学生，32个独特步骤和5个技能（KCs）学生运动日志。
所有学生在心理问题类别中解决了9个问题，其中3个是简单问题，1个是大问题。
所涉及的总步骤是32.在解决运动问题时，学生可以要求提示解决一个步骤。
每个数据点是对应于单个技能执行的正确或不正确的学生动作。
学生行为被编码为正确或不正确，并根据执行该行动所需的“知识组件”（KCs）进行分类。
学生执行的每个步骤都与KC相关，并被记录为学生掌握该KC的“机会”。
本课程有5项技能（转换，除法，乘法，加法和结果）对应于步骤中所需的技能。
每个步骤都为此研究分配了一个KC。

The table 2 shows a sample data with columns:
表2显示了一个带有列的样本数据：

Student- name of the student;

学生的学生姓名;

Step – problem 1 Step1; Success – Whether the student did that step correctly or not in the first attempt. 1- success and 0-failure;

步骤 - 问题1 Step1;成功 - 学生是否在第一次尝试中正确执行了该步骤。 1-成功和0失败;

Skill – Knowledge component used in that step;
技能 - 该步骤中使用的知识组件;

Opportunities – Number of times the skill is used by the same student computed from the first and fourth column.
机会 - 从第一和第四列计算的同一学生使用技能的次数。

Table 2. The sample data




