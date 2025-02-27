
属性层级模型（AHM）
=================

 :math:`AHM` **需要先确定属性间的层级关系，即属性层级结构(AHS)** 。

分类方法 :math:`A` （ :math:`AHM-A` ）：将需要判别的一个观察反应模式与所有的期望反应模式逐个比较，求取观察反应模式与每一个期望反应模式相似的概率，最后将观察反应模式判归于有最大相似概率的期望反应模式，实际上也就是将拥有该观察反应模式的被试判定为有最大相似概率的期望反应模式所对应的属性掌握模式。
将观察反应模式 :math:`X_{i}` 与每个期望反应模式 :math:`V_{r}`
逐个元素进行比较。记 :math:`D_{ir}=V_{r}-X_{i}`
产生一个以 :math:`\left\{ -1,0,+1 \right\}` 为元素的 :math:`M` 维向量，

当 :math:`D_{ir}` 中元素 :math:`d_{j}=0` 时表示被试 :math:`i` 在项目 :math:`j` 上既无失误也无猜测；

当 :math:`d_{j}=1` 时，表示被试 :math:`i` 在项目 :math:`m` 上产生 :math:`(1->0)` 失误，其概率为 :math:`1-P_{m}(\theta_{r})` ，
 :math:`\theta_{r}` 为第 :math:`r` 个理想反应模式的能力估计值，
 :math:`P_{m}(\theta_{r})` 表示能力值为 :math:`\theta_{r}` 的期望被试答对第 :math:`m` 题的概率；

当 :math:`d_{j}=-1` 时，表示被试i在项目k上产生 :math:`(1->0)` 猜测，其概率为 :math:`P_{k}(\theta_{r})` ，
 :math:`P_{k}(\theta_{r})` 表示能力为 :math:`\theta_{r}` 的被试答对第 :math:`k` 题的概率。

 :math:`A` 方法考察失误和猜测的情形，对于第 :math:`r` 个理想反应模式，计算发生失误和猜测的似然为：

.. math::

    P_{ rExpected }(\theta )=\prod _{ k=1 }^{ K }{ { P }_{ m }({ \theta  }_{ r }) } \prod _{ m=1 }^{ L }{ \left[ 1-{ P }_{ m }({ \theta  }_{ r }) \right]  }

表示发生了 :math:`K` 个从 :math:`(0->1)` 的猜测， :math:`L` 个从 :math:`(1->0)` 的失误。
**计算 :math:`X_{i}` 在每个理想反应模式下的似然值，并把 :math:`X` 归到使 :math:`P_{ rExpected }(\theta )` 值最大的类中，即归到似然值最大的类中。**
\*这里计算的能力值是理想反应模式下的能力值

方法 :math:`B(AHM-B)` ：认为观察反应模式 :math:`X_{i}` 的知识状态，要么为 :math:`X_{i}` 包含的理想反应模式对应的知识状态，要么取 :math:`X` 不包含的理想反应模式中似然值最大的理想反应模式对应的知识状态，似然值的计算方法仅取 :math:`A` 中的失误部分，即

.. math::

    \prod _{ m=1 }^{ L }{ \left[ 1-{ P }_{ m }({ \theta  }_{ r }) \right] }

最小化广义距离判别法（ :math:`GDD` ）：

.. math::

    d({ X }_{ i },{ V }_{ r })=\sum _{ j=1 }^{ M }{ d({ X }_{ ij },{ V }_{ rj }) } =\sum _{ j=1 }^{ M }{ \left| { X }_{ ij }-{ V }_{ rj } \right|  } { P }_{ j }({ \theta  }_{ i })^{ { X }_{ ij } }(1-{ P }_{ j }({ \theta  }_{ i }))^{ 1-{ X }_{ ij } }

**该方法以观察反应模式估计的能力值为基准** ，
计算每个项目答对的概率 :math:`{ P }_{ j }({ \theta  }_{ i })` 与答错概率 :math:`1-{ P }_{ j }({ \theta  }_{ i })` ，
然后与理想反应模式进行匹配，
理想反应模式与观察反应模式元素相同，定义距离为 :math:`0` ；
理想反应模式为 :math:`0` ，观察反应模式为 :math:`1` ，定义距离为 :math:`{ P }_{ j }({ \theta  }_{ i })` ，反之，定义距离为 :math:`1-{ P }_{ j }({ \theta  }_{ i })` ，
使得距离最小的理想反应模式对应的知识状态为被试的知识状态

\*AHM的分类方法还可采用神经网络

