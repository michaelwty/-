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