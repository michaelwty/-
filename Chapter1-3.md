1.1. 基本术语
样本与数据
数据集 data set：样本的集合
示例instance（或样本 sample）：每个事件或对象的描述
属性attribute（或特征 feature）：反映事件或对象在某方面的表现或者性质的事项
属性值attribute value ：属性上的取值
属性空间attribute space：属性张成的空间
特征向量 feature vector： 每个样本在属性空间中的坐标
维数 dimensionality: 属性空间的维数
数学表达

​ 令$ D={\boldsymbol{x_1},\boldsymbol{x_2},\ldots,\boldsymbol{x_m}} $，表示$ m $个示例的数据集，每个实例由$ d $个属性描述，则每个实例：$ \boldsymbol{x_i}=(x_{i1};x_{i2};\ldots;x_{id}) $

预测与聚类
学习 learning（或训练training）： 从数据中学得模型的过程

训练数据 training data： 训练过程使用的数据

训练样本 training sample：训练数据中的一个样本

训练集 training set：训练样本的集合

假设 hypothesis：学得模型对应的某种潜在规律，这种前在规律成为“真相”或“真实”（ground-truth）

学习器learner：模型

预测

预测 prediction：用模型预言样本的“结果”

标签 label：“结果”，定性结果

样例 example：拥有标签的样本

标签空间 label space：标签的

分类 classification： 若预测的是离散值，此类学习任务为分类

二分类 binary classification：两个类型的分类
通常称其中一个类为“正类” positive class，另一类为“负类” negative class
多分类 multi-class classification
回归 regression：若预测值是连续值，此类学习任务为回归

测试 testing：学得模型，使用模型进行预测的过程

​ **数学表达：**对训练集$ {(\boldsymbol{x_1},y_1),(\boldsymbol{x_2},y_2),\ldots,(\boldsymbol{x_m},y_m)} $进行学习，建立一个从输入空间$\mathcal{X}$到标签空间$\mathcal{Y}$

的映射$f:\mathcal{X}\mapsto\mathcal{Y}$ 。二分类中，通常$\mathcal{Y}={-1, +1}$或${0, 1}$；对于多分类任务，$|\mathcal{Y}|>2$；对回归任务，$\mathcal{Y}=\mathbb{R}$，$\mathbb{R}$为实数。测试：$y=f(x)$。

分类

聚类 clustering：将对象分为若干组
簇 cluster：每一组为一个簇
根据标记信息的有无，分类任务分为：监督学习(supervised learning)和无监督学习(unsupervised learning)
泛化

泛化 generalization：学得的模型嫩南瓜适用于新样本的能力
尽管训练集通常只是样本空间的一个很小的采样，我们仍希望它能很好地反映出样本空间的特性，否则就很难期望在训练集上学得的模型能再整个样本空间上都工作得很好。
通常假设样本空间中全体样本服从一个未知“分布”（distribution）$\mathcal{D}$，我们得到的每个样本都是独立地从这个分布上采样获得的，即“独立同分布”（independent and identically distributed，i.i.d.）。一般而言，训练样本越多，我们得到的关于$\mathcal{D}$的信息越多，这样就越有可能通过学习获得具有强泛化能力的模型。
1.2. 假设空间
归纳与演绎是科学推理的两大基本手段。前者是从特殊到一般的“泛化”；后者是从一般到特殊的“特化”过程。

“从样例学习”显然是一种归纳的过程，所以称为“归纳学习”（inductive learning）。

归纳学习分为狭义和广义，广义的归纳学习相当于从样例中学习，狭义的归纳学习则要求从训练数据中学得概念（concept），因此亦称为“概念学习”或“概念形成”。

学习过程是从“假设空间”中学习得到一个“假设”。学习过程是基于有限样本训练集，可能存在一个假设集与训练集一致，我们称之为“版本空间”。

1.3. 归纳偏好
学习得到的模型对应假设空间中的一个假设，但是版本空间带我们带来麻烦。假设集对应的模型对同一个新样本会有不同的输出。

由于我们必须从多个假设中选择一个假设，机器学习算法在学习过程中对某种类型假设的偏好，成为“归纳偏好”（inductive bias）。

归纳偏好是一种选择的启发式或价值观。

“奥卡姆剃刀”是一种常见的、基本的原则，即“若有多个假设与观察一致，则选择最简单的那个”。

“天下没有免费的午餐”（NFL）定理：所有学习算法的期望性能跟随机胡猜差不多。NFL定理最重要的寓意是，让我们清楚地知道，脱离具体问题，空泛讨论“什么学习算法更好”毫无意义。

1.4. 发展历程与应用现状
略

# 线性模型

- 基本形式

$$f(x)=w_{1}x_{1}+w_{2}x_{2}+\dots+w_{d}x_{d}+b$$

- 向量形式

$$f(x)=w^{T}x+b$$

# 单元线性回归

$$f(x) = wx_{i}+b,使得f(x_i) \approx y_i$$

性能度量：

$$(w^*,b^*)=\underset{(w,b)}{arg min}\sum_{i=1}^{m}(f(x_i)-y_i)^2\\=\underset{(w,b)}{arg min}\sum_{i=1}^{m}(y_i-wx_i-b)^2$$

$$只需针对w和b分别求骗到即可得到最优解$$

# 多元线性回归

$$f(x_i)=w^Tx_i +b使得f(x_i)\approx y_i, 其中x_i=(x_{i1};x_{i2};\dots;x_{id})$$

$$数据集可表示为\\X=\begin{pmatrix}        x_1 & x_{12} & \dots &x_{1d} & 1 \\       x_{21} & x_{22} & \dots & x_{2d} & 1 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ x_{m1} & x_{m2} & \dots & x_{md} & 1  \end{pmatrix}=\begin{pmatrix}x_1^T &1 \\ x_2^T & 1 \\ \vdots & \vdots \\x_m^T & 1\end{pmatrix}.$$

$$类似使用最小二乘法，线性预测使得如下指标最小 \\ w^*=\underset{w}{argmin}(y-Xw)^T(y-Xw)$$

# 广义线性模型

现实中不可能每次都能用线性模型进行拟合，需要对输出做空间的非线性映射，便可得到广义的线性模型，从线性到非线性。

$$y=g^{-1}(w^{T}x+b)$$

# 线性判别分析（LDA）

- 给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离；在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别。（监督降维技术）

[](https://www.notion.so/a8ba13664a654924bb49045f6495b32e#89fead99ae2d41d9862ea7df69e5ae04)

# 多分类学习

- 多分类学习可在二分类基础上进行。将原问题拆分为多个二分类任务，然后每个二分类训练一个分类器，然后再进行集成获得最终的多分类结果。
- 最典型的拆分策略有三种：

     1. OvO One vs. One

     2. OvR One vs. Rest 

     3. MvM Many vs. Many