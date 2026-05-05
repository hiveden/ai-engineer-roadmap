# 主 PPT (06集成学习.pptx)

## Slide 1

**标题**: 集成学习

## Slide 2

集成学习思想
随机森林算法
Adaboost算法
GBDT
XGBoost

## Slide 3

知道集成学习是什么？
了解集成学习的分类
理解bagging集成的思想
理解boosting集成的思想

## Slide 4

**标题**: 集成学习
集成学习是机器学习中的一种思想，它通过多个模型的组合形成一个精度更高的模型，参与组合的模型成为弱学习器（弱学习器）。训练时，使用训练集依次训练出这些弱学习器，对未知的样本进行预测时，使用这些弱学习器联合进行预测。

_备注_: 大法师得分

## Slide 5

**标题**: 集成学习

## Slide 6

**标题**: 集成学习分类
Bagging：随机森林
Boosting：Adaboost、GBDT、XGBoost、LightGBM
自助聚合  装袋法                                           增强法
学习器之间相互独立                                                   学习器之间需要通信
学习器需要获得前一个学习器的输出

## Slide 7

**标题**: 集成学习 – Bagging思想
Bagging思想图
有放回的抽样(bootstrap抽样)产生不同的训练集，从而训练不同的学习器 （每个模型的训练集不相同）
通过平权投票、多数表决的方式决定预测结果
弱学习器可以并行训练
平权投票

_备注_: Bootstrap  用现有的数据进行深入挖掘

## Slide 8

**标题**: 集成学习 – Bagging思想
Bagging思想图
目标：把右图的圈和方块进行分类
1 采样不同数据集
2 训练分类器
3 平权投票，获取最终结果
测试样本X

_备注_: 每个样本所属的类别 圈圈还是方块，由3个分类器平权投票决定     三个分类器的训练样本不同。

## Slide 9

**标题**: 集成学习 – Boosting思想
每一个训练器重点关注前一个训练器不足的地方进行训练
通过加权投票的方式，得出预测结果
串行的训练方式
初步结果                        修补的内容                             再次修补的内容
Boosting思想图
全部样本        全部样本       全部样本
通信                            通信

## Slide 10

**标题**: 集成学习 – Boosting思想
Boosting思想生活中的举例
随着学习的积累从弱到强
每新加入一个弱学习器，整体能力就会得到提升
代表算法：Adaboost，GBDT，XGBoost，LightGBM
滚球兽→亚古兽→暴龙兽→机械暴龙兽→战斗暴龙兽

## Slide 11

**标题**: Bagging&Boosting对比
 | bagging | boosting
数据采样 | 对数据进行有放回的采样训练 | 全部样本，根据前一轮学习结果调整数据的重要性
投票方式 | 所有学习器平权投票 | 对学习器进行加权投票
学习顺序 | 并行的，每个学习器没有依赖关系 | 串行，学习有先后顺序

## Slide 12

1 集成学习是什么？
多个弱学习器组合成一个更强大的学习器，解决单一预测，进步一得到更好性能。
2 bagging思想
有放回的抽样
平权投票、多数表决的方式决定预测结果
并行训练
3 boosting思想
重点关注前一个训练器不足的地方进行训练
加权投票的方式
串行的训练方式

## Slide 13

A  一个精度比较高的分类器
B 从数据集中随机采样出多个子集，每个子集训练一个弱分类器，然后将这些弱分类器组合成一个强分类器
C 从特征集中随机选择多个特征，使用这些特征训练一个弱分类器，然后将多个弱分类器组合成一个强分类器
D  以上都不是
正确答案：B
C是随机森林的定义
1、Bagging 是什么意思，下面说法正确的是（）

## Slide 14

集成学习思想
随机森林算法
Adaboost算法
GBDT
XGBoost

## Slide 15

理解随机森林的构建方法
知道随机森林的API
能够使用随机森林完成分类任务

## Slide 16

**标题**: 随机森林算法    树多了，即为森林
随机森林是基于 Bagging 思想实现的一种集成学习算法，采用决策树模型作为每一个弱学习器。
训练：
有放回的产生随机训练样本  （什么是有放回抽样？）
随机挑选 n 个特征（n 小于总特征数量, 随机森林的要求）
预测：平权投票，多数表决输出预测结果
平权投票

_备注_: 什么是有放回抽样？

## Slide 17

**标题**: 随机森林步骤

_备注_: 也能做回归   各弱分类器的平均预测值  CART回归树

## Slide 18

**标题**: 随机森林算法 – 概念
思考题1：为什么要随机抽样训练集？
思考题2：为什么要有放回地随机抽样？

## Slide 19

**标题**: 随机森林算法 – 概念
思考题1：为什么要随机抽样训练集？   确保各学习器训练集“有差异”
如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样。
思考题2：为什么要有放回地随机抽样？  确保各学习器训练集“有交集”
如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是“有偏的”，也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。
综上：弱学习器的训练样本既有交集也有差异数据，更容易发挥投票表决效果
有放回抽样：
10个学生各自从整本教材中随机抄了80%的内容（有重复）来复习。→ 每个人都学过大部分核心概念，只是例题略有不同。→ 考试时大家水平差不多，取平均值有意义。
无放回抽样：
10个学生各自从整本教材中随机拿走了10%的内容（无重复）来复习。→ 每个人仅关注一小部分概念，互不覆盖。→ 考试时，每人只答各自学过的一部分题， 取均值无意义。

## Slide 20

**标题**: 随机森林算法 – API

## Slide 21

**标题**: 随机森林算法 – API

## Slide 22

**标题**: 泰坦尼克号案例
import pandas as pdfrom sklearn.model_selection import train_test_splitfrom sklearn.tree import DecisionTreeClassifierfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.model_selection import GridSearchCVdef dm01_随机森林():    # 1 获取数据集    titan = pd.read_csv(“./data/titanic/train.csv”)    # 2 确定特征值和目标值    x = titan[[“Pclass”, “Age”, “Sex”]].copy()    y = titan[“Survived”]    # 3-1 处理数据-处理缺失值    x[‘Age’].fillna(value=titan[“Age”].mean(), inplace=True)    print(x.head())    # 3-2 one-hot编码    x = pd.get_dummies(x)       # 4 数据集划分    x_train, x_test, y_train, y_test = \        train_test_split(x, y, random_state=22, test_size=0.2)

## Slide 23

**标题**: 泰坦尼克号案例
# 5-1 使用决策树进行模型训练和评估    dtc = DecisionTreeClassifier()    dtc.fit(x_train, y_train)    dtc_y_pred = dtc.predict(x_test)    accuracy = dtc.score(x_test, y_test)    print('单一决策树accuracy-->\n', accuracy)    # 5-2 随机森林进行模型训练和评估    rfc = RandomForestClassifier(max_depth=6, random_state=9)    rfc.fit(x_train, y_train)    rfc_y_pred = rfc.predict(x_test)    accuracy = rfc.score(x_test, y_test)    print('随机森林进accuracy-->\n', accuracy)    # 5-3 随机森林 交叉验证网格搜索 进行模型训练和评估    estimator = RandomForestClassifier()    param = {"n_estimators": [40, 50, 60, 70], "max_depth": [2, 4, 6, 8, 10], "random_state":[9]}    grid_search = GridSearchCV(estimator, param_grid=param, cv=2)    grid_search.fit(x_train, y_train)    accuracy = grid_search.score(x_test, y_test)    print("随机森林网格搜索accuracy:", accuracy)    print(grid_search.best_estimator_)

## Slide 24

1 随机森林概念
bagging思想的代表算法，bagging+决策树
2 随机森林构建过程
1随机选数据、2随机选特征，3训练弱学习器、4重复1-3训练n个、5平权投票
3 随机森林API sklearn.ensemble.RandomForestClassifier()

## Slide 25

正确答案： B→A→D→C。
1、请对下列随机森林的构建方法进行排序：
A）重复采样，构建出多颗决策树
B）随机选取部分样本，并随机选取部分特征交给其中一颗决策树训练
C）如果是分类场景则采用平权投票的方式决定最终随机森林的预测结果，
如果是回归场景则采用简单平均法获取最终结果
D）将相同的测试数据交给所有构建出来的决策树进行及结果预测

## Slide 26

集成学习思想
随机森林算法--bagging
Boosting算法之    Adaboost算法     分类问题
Boosting算法之 GBDT                     回归问题
Boosting算法之 XGBoost                回归问题

## Slide 27

理解adaboost算法的思想
知道adaboost的构建过程
实践泰坦尼克号生存预测案例

## Slide 28

全部样本        全部样本       全部样本
通信                             通信
初步结果                       修正的结论               再次修正的结论

## Slide 29

**标题**: Adaboost 类比： 珠宝真伪鉴定
初步结果                       修正的结论               再次修正的结论
全部样本        全部样本       全部样本
通信                             通信

## Slide 30

**标题**: Adaboost算法
Adaptive Boosting(自适应提升)基于 Boosting思想实现的一种集成学习算法核心思想是通过逐步提高那些被前一步分类错误的样本的权重来训练一个强分类器。
1.训练第一个弱学习器 （性别分类）
2.调整数据分布
身高
体重
身高
体重
身高
体重
身高
体重
Y =
线性分类器
让下一个分类在训练的时候
多关注一下错样本

## Slide 31

**标题**: Adaboost算法
3.训练第二个弱学习器
4.再次调整数据分布
体重
身高
体重
身高
体重
身高
体重
蓝色区域： 圈圈       红色区域： 叉叉

## Slide 32

**标题**: Adaboost算法
5.依次训练学习器，调整数据分布
6.整体过程实现
AdaBoost 通过迭代训练一系列弱分类器，每一轮根据前一轮的表现，动态调整样本权重——分错的样本被“加重”，分对的被“减轻”，让后续分类器更关注难例。
每个模型的发言权
蓝色区域： 圈圈       红色区域： 叉叉

_备注_: 基于前面的分类器进一步训练。    前面的分类器，贡献就是 不同样本 权重

## Slide 33

**标题**: Adaboost算法推导
2 根据新权重的样本集 训练第 2 个学习器
根据预测结果找一个错误率最小的分裂点
计算、更新：分类错误率  模型权重  样本权重
3 迭代训练在前一个学习器的基础上，根据新的样本权重训练当前学习器
- 直到训练出 m 个弱学习器
1 初始化训练数据权重相等，训练第 1 个学习器
如果有 100 个样本，则每个样本的初始化权重为：1/100
根据预测结果找一个错误率最小的分裂点（定义分割线）
计算、更新：分类错误率  模型权重  样本权重
模型                   模型                   模型

_备注_: 模型权重 ，用于
样本权重  用于

## Slide 34

**标题**: Adaboost算法推导
模型                   模型                   模型

## Slide 35

基分类器必须是同一种，比如都是逻辑回归  或都是决策树。   否则称为 “异构集成”

## Slide 36

**标题**: Adaboost算法 – 构建过程举例
已知训练数据见下面表格，假设弱分类器由 x 产生， 即用若干个决策树
预测结果使该分类器在训练数据集上的分类误差率最低，试用 Adaboost 算法学习一个强分类器。
特征
标签
正例  负例

## Slide 37

**标题**: Adaboost算法-构建第1个弱分类器      - 简化  一维数据
特征
权重
标签
标签： 左  1  右  -1
依次类推

## Slide 38

**标题**: Adaboost算法 –构建第1个弱分类器
该模型的发言权
Zt 发生变更
初始值 0.1  x  e^alpha
重新归一化
特征
权重更新！！
标签
发现总权重不再为1

_备注_: 注意  第一到第十的样本  编号id从0-9

## Slide 39

**标题**: Adaboost算法-构建第2个弱学习器
特征
权重更新！！
标签

## Slide 40

**标题**: Adaboost算法 –构建第2个弱学习器


=
} =
3、4、5

_备注_: 注意  第一到第十的样本  编号id从0-9

## Slide 41

**标题**: Adaboost算法 – 构建第3个弱学习器

## Slide 42

最终强学习器
每个学习器都基于
前一个学习器的权重信息

## Slide 43

**标题**: 案例AdaBoost实战葡萄酒数据
需求
已知葡萄酒数据，根据数据进行葡萄酒分类
API： myada = AdaBoostClassifier(base_estimator=mytree, n_estimators=500, learning_rate=0.1)
参1: 弱分类器(事先定义好的决策树对象)
参2: 弱分类器个数
参3: 学习率，作用于模型权重
用于增大或缩小每个权重的贡献
学习率：  0~ + ∞
从而影响了样本权重的更新快慢

_备注_: 集成算法, 若为SAMME .r 表示输出 软标签 也就是 概率

## Slide 44

**标题**: 案例AdaBoost实战葡萄酒数据
# AdaBoost实战葡萄酒数据import pandas as pdfrom sklearn.preprocessing import LabelEncoderfrom sklearn.model_selection import train_test_splitfrom sklearn.tree import DecisionTreeClassifierfrom sklearn.ensemble import AdaBoostClassifier     # 集成学习from sklearn.metrics import accuracy_scoredef dm01_adaboost():    # 1 读数据到内存 df_wine    df_wine = pd.read_csv('./data/wine0501.csv')    # df_wine.info()    # 2 特征处理    # 2-2 Adaboost一般做二分类 去掉一类(1,2,3)    df_wine = df_wine[df_wine['Class label'] != 1]    # 2-3 准备特征值和目标值 Alcohol酒精含量 Hue颜色    x = df_wine[['Alcohol', 'Hue']].values    y = df_wine['Class label']    # 2-4 类别转化y (2,3)=>(0,1)    y = LabelEncoder().fit_transform(y)    # print('y-->\n', y )    # 2-5 划分数据    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
思路分析
# 1 读数据到内存# 2 特征处理# 2-1 修改列名# 2-2 Adaboost一般做二分类，去掉过多的类别 (比如有类别 1,2,3 )# 2-4 类别转化 (2,3)=>(0,1)
注意  （0,1）标签与 （-1,1）标签均可接受
# 2-5 划分数据
# 3 实例化单决策树 实例化Adaboost-由500颗树组成# 4 单决策树训练和评估# 5 AdaBoost训练和评估
如果基学习器太强，则集成学习不一定能得到更好结果。

## Slide 45

**标题**: 案例AdaBoost实战葡萄酒数据
# 3 实例化单决策树 实例化Adaboost-由500颗树组成    mytree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)    myada = AdaBoostClassifier(base_estimator=mytree, n_estimators=500, learning_rate=0.1, random_state=0)    # 4 单决策树训练和评估    mytree.fit(X_train, y_train)    myscore = mytree.score(X_test, y_test)    print('myscore-->', myscore)    # 5 AdaBoost训练和评估    myada.fit(X_train, y_train)    myscore = myada.score(X_test, y_test)    print('myscore-->', myscore)

## Slide 46

1 Adaboost概念
通过逐步提高被分类错误的样本的权重来训练一个强分类器。提升的思想
2 Adaboost构建过程
1 初始化数据权重，来训练第1个弱学习器。找最小的错误率计算模型权重，再更新模数据权重。
2 根据更新的数据集权重，来训练第2个弱学习器，再找最小的错误率计算模型权重，再更新模数据权重。
3 依次重复第2步，训练n个弱学习器。组合起来进行预测。结果大于0为正类、结果小于0为负类

## Slide 47

1、下列关于Adaboost的说法正确的是的是？（多选）
A）Adaboost算法一般用来做二分类，特别在视觉领域应用较多
B）AdaBoost算法不能提高精度
C）AdaBoost算法API函数可以配置学习率参数，学习率参数作用于每一颗树的数据权重更新上
D）AdaBoost算法使用的树深度不要过深，否则容易过拟合
答案：AD

## Slide 48

集成学习思想
随机森林算法
Adaboost算法
GBDT
XGBoost

## Slide 49

能说出残差提升树的概念
能说出残差提升树的基本构建过程
能说出梯度提升树GBDT的基本构建过程

## Slide 50

**标题**: 提升树 （Boosting Decision Tree ）  残差树 –做回归
思想
通过拟合残差的思想来进行性能提升
残差：真实值 - 预测值
生活中的例子
预测某人的年龄为100岁
第1次预测：对100岁预测，预测成80岁；100 – 80 = 20（残差）
第2次预测：上一轮残差20岁作为目标值，预测成16岁；20 – 16 = 4 （残差）
第3次预测：上一轮的残差4岁作为目标值，预测成3.2岁；4 – 3.2 = 0.8（残差）
若三次预测的结果串联起来： 80 + 16 + 3.2 = 99.2
通过拟合残差可将多个弱学习器组成一个强学习器，这就是提升树的最朴素思想
商品
一共82元
50元                   20元                 10元
举例2： 每个小朋友，看到（输入）同样的商品，回归目标分别是82元、32元、12元
模型之间的信息传递  = 前一个模型的残差
通信
通信

## Slide 51


_备注_: 本页讲完，直接将补充材料

## Slide 52


_备注_: 本页讲完，直接将补充材料

## Slide 53


_备注_: 本页讲完，直接将补充材料

## Slide 54

**标题**: 梯度提升树 （Gradient Boosting Decision Tree）
GBDT 利用梯度下降的近似方法，利用损失函数的负梯度作为提升树算法中的残差近似值。
假设：
前一轮迭代得到的强学习器是：fi-1(x)
损失函数为平方损失是：L (  y,  f​i−1(x)  )
本轮迭代的目标是找到一个弱学习器：输出预测值 hi(x)
让本轮的损失最小化: Loss  = L ( y,   fi−1(x)) + hi(x)  )
5.则要拟合单个样本的负梯度为:
即：对于平方损失
GBDT 拟合的负梯度就是残差。
对于任一样本：
对于任一模型参数：

## Slide 55

**标题**: 梯度提升树   ---- 基础学习器：
对于整个训练集：给出标签的均值，作为基准值。
执行标准 ：
当模型预测值为何值时，会使得第一个弱学习器 在所有训练样本上
预测的平方误差最小，即：求损失函数对 f(xi) 的导数，并令导数为0.
商品
一共82元
50元                   20元                 10元
作为一个基准值
不能差太多 ！！

_备注_: 本页讲完，直接将补充材料

## Slide 56


_备注_: 本页讲完，直接将补充材料

## Slide 57

**标题**: 梯度提升树 –案例
已知：
基础学习器(CART树)：  给出标签的均值
当模型预测值为何值时，会使得第一个弱学习器的平方误差最小，即：求损失函数对 f(xi) 的导数，并令导数为0.

## Slide 58

**标题**: 梯度提升树 – 例子2
2 构建第1个弱学习器，根据负梯度的计算方法得到下表：
3 当 6.5 作为切分点时，平方损失最小，此时得到第1棵决策树
当1.5为切分点：拟合负梯度-1.75, -1.61, -1.40, -0.91, … , 1.74
左子树：1个样本 -1.75， 右子树9个样本：-1.61，-1.40，-0.91…
右子树均值为：((-1.61) + (-1.40)+(-0.91)+(-0.51)+(-0.26)+1.59 +1.39 + 1.69 + 1.74 )/9=0.19；左子树均值为：- 1.75
计算平方损失：左子树0 + 右子树：(-1.61-0.19)*(-1.61-0.19)  + (-1.40-0.19)* (-1.40-0.19) + (-0.91-0.19)* (-0.91-0.19) + (-0.51-0.19)*(-0.51-0.19) +(-0.26-0.19)*(-0.26-0.19) +(1.59-0.19) *(1.59-0.19) + (1.39-0.19) *(1.39-0.19) + (1.69-0.19)*(1.69-0.19)  + (1.74-0.19) * (1.74-0.19)  =15.72308
就是分割后，两个类别的均值作为预测值
也是新的目标值
开始切分

## Slide 59

**标题**: 梯度提升树 – 例子3
构建第2个弱学习器
以3.5 作为切分点时，平方损失最小，此时得到第2棵决策树
就是分割后，
两个类别的均值作为预测值
新的负梯度
新的目标值

## Slide 60

**标题**: 梯度提升树 – 例子4
构建第3个弱学习器
以6.5 作为切分点时，平方损失最小，此时得到第3棵决策树

## Slide 61

**标题**: 梯度提升树 – 例子5
构建最终弱学习器
以把x=6样本为例：输入到最终学习器中进行判别，结果 ：7.31 + (-1.07) + 0.22 + 0.15 = 6.61

## Slide 62

**标题**: 梯度提升树的构建流程
1 初始化弱学习器（目标值的均值作为预测值）
2 迭代构建学习器，每一个学习器拟合上一个学习器的负梯度
3 直到达到指定的学习器个数
4 当输入未知样本时，将所有弱学习器的输出结果组合起来作为强学习器的输出

## Slide 63

**标题**: 梯度提升树 – 案例泰坦尼克号生存预测
# 导入库import pandas as pdfrom sklearn.model_selection import train_test_splitfrom sklearn.tree import DecisionTreeClassifierfrom sklearn.ensemble import GradientBoostingClassifierfrom sklearn.metrics import classification_reportfrom sklearn.model_selection import GridSearchCVdef dm01_gbdtapi():    # 1 读数据到内存    taitan_df = pd.read_csv('./data/titanic/train.csv')    # taitan_df.info()    # print('taitan_df.describe()-->\n', taitan_df.describe())    # 2 数据基本处理准备    # 2-1 x y    x = taitan_df[['Pclass', 'Age', 'Sex']].copy()    y = taitan_df['Survived'].copy()    # 2-2 缺失值处理    x['Age'].fillna(x['Age'].mean(), inplace=True)    # 2-3 pclass离散型数据需one-hot编码    x = pd.get_dummies(x)

## Slide 64

**标题**: 梯度提升树 – 案例泰坦尼克号生存预测
# 2-4 数据集划分    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)    # 3 GBDT 训练和评估    estimator = GradientBoostingClassifier()    estimator.fit(x_train, y_train)    mysorce = estimator.score(x_test, y_test)    print("gbdt mysorce-->1", mysorce)    # 4 GBDT 网格搜索交叉验证    estimator = GradientBoostingClassifier()    param = {"n_estimators": [100, 110, 120, 130], "max_depth": [2, 3, 4], "random_state": [9]}    estimator = GridSearchCV(estimator, param_grid=param, cv=3)    estimator.fit(x_train, y_train)    mysorce = estimator.score(x_test, y_test)    print("gbdt mysorce-->2", mysorce)    print(estimator.best_estimator_)    pass

## Slide 65

1 提升树？
每一个弱学习器通过拟合残差来构建强学习器
2 梯度提升树
每一个弱学习器通过拟合负梯度来构建强学习器

## Slide 66

1、下列关于GBDT的说法正确的是？（多选）
A）它使用的弱学习器是决策树
B）它使用了Boosting的思想
C）去拟合每次弱学习器学习后的负梯度信息
D）GBDT可以解决回归问题
答案：ABCD。

## Slide 67

集成学习思想
随机森林算法
Adaboost算法
GBDT
XGBoost

## Slide 68

知道XGBoost算法的思想
理解XGBoost目标函数
了解XGBoost的算法API
实现红酒品质预测案例

## Slide 69

**标题**: XGBoost (Extreme Gradient Boosting)
2、在损失函数中加入正则化项，
提高对未知的测试数据的泛化性能 。
极端梯度提升树，集成学习方法的王牌，在数据挖掘比赛中，大部分获胜者用了XGBoost。
Xgb的构建思想：
1、构建模型的方法是最小化训练数据的损失函数:
训练的模型复杂度较高，易过拟合。

## Slide 70

**标题**: XGBoost
XGBoost（Extreme Gradient Boosting）是对GBDT的改进，并且在损失函数中加入了正则化项
正则化项用来降低模型的复杂度

## Slide 71

**标题**: XGBoost
假设我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分：
多个特征

## Slide 72

**标题**: XGBoost
训练出tree1和tree2，类似之前gbdt的原理，两棵树的结论累加起来便是最终的结论
树tree1的复杂度表示为
多个特征

## Slide 73

**标题**: XGBoost
进行 t 次迭代的学习模型的目标函数如下为：
直接对目标函数求解比较困难，通过泰勒展开将目标函数换一种近似的表示方式

## Slide 74

**标题**: XGBoost（复习）
泰勒展开
将一个函数在某一点处展开成无限项的多项式表达式       （ 用             的一些列表达式，等价于                     的值）
一阶泰勒展开
二阶泰勒展开

## Slide 75

**标题**: XGBoost提升树 – 目标函数推导2 – 泰勒展开3
目标函数对 yi(t-1) 进行泰勒二阶展开，得到如下近似表示的公式：
观察目标函数，发现以下两项表示t-1个弱学习器构成学习器的目标函数，都是常数，我们可以将其去掉：
其中gi 和 hi 的分别为损失函数的一阶导、二阶导：

## Slide 76

**标题**: XGBoost
从样本角度转为按照叶子节点输出角度，优化损失函数
举个栗子：请计算10样本在叶子结点上的输出表示
上式中：
gi 表示每个样本的一阶导，hi 表示每个样本的二阶导
ft(xi) 表示样本的预测值
T 表示叶子结点的数目
||w||2 由叶子结点值组成向量的模

## Slide 77

**标题**: XGBoost
gift(xi) 表示样本的预测值，表示为：
hift2(xi) 转换从叶子结点的问题，表示为：
λ||w||2 由于本身就是从叶子角度来看，表示为：
目标函数中的各项可以做以下转换：

## Slide 78

**标题**: XGBoost提升树 – 目标函数推导3 – 转化为叶子节点输出角度3
令：
Gi 表示所有样本的一阶导之和
Hi 表示所有样本的二阶导之和
最终：

## Slide 79

**标题**: XGBoost提升树 – 目标函数推导4 – 目标函数最优解1
求损失函数最小值
对 w 求导并令其等于 0，可得到 w 的最优值
最优w，带入公式可求目标函数的最小值：

## Slide 80

**标题**: XGBoost
目标函数最终为：
该公式也叫做打分函数 (scoring function)，从损失函数、树的复杂度两个角度来衡量一棵树的优劣。当我们构建树时，可以用来选择树的划分点，具体操作如下式所示：

## Slide 81

**标题**: XGBoost
根据上一页PPT中计算的gain值：
对树中的每个叶子结点尝试进行分裂
计算分裂前 - 分裂后的分数：
如果gain > 0，则分裂之后树的损失更小，会考虑此次分裂
如果gain< 0，说明分裂后的分数比分裂前的分数大，此时不建议分裂
当触发以下条件时停止分裂：
达到最大深度
叶子结点数量低于某个阈值
所有的结点在分裂不能降低损失
等等...

## Slide 82

**标题**: XGBoost算法API
XGB的安装和使用
XGB的编码风格
在sklean机器学习库中没有集成xgb。想要使用xgb，需要手工安装
pip3 install xgboost
可以在xgb的官网上查看最新版本：https://xgboost.readthedocs.io/en/latest/
支持非sklearn方式，也即是自己的风格
支持sklearn方式，调用方式保持sklearn的形式

## Slide 83

**标题**: xgb案例：红酒品质分类
已知
数据集共包含 11 个特征，共计 3269 条数据. 我们通过训练模型来预测红酒的品质, 品质共有 6个类别，
分别使用数字:0、 1、2、3、4、5 来表示
需求：对红酒品质进行多分类
分析：
从数据可知 1、目标是多分类    2、数据存在样本不均衡问题

## Slide 84

**标题**: 案例：红酒品质分类
import joblibimport numpy as npimport pandas as pdimport xgboost as xgbfrom collections import Counterfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import classification_reportfrom sklearn.model_selection import StratifiedKFold   # 基本数据处理def dm01_realdata():    # 1 加载训练集    data = pd.read_csv('./data/红酒品质分类.csv')    x = data.iloc[:, :-1]    y = data.iloc[:, -1] - 3    # 2 数据集划分    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)    # 3 数据存储    pd.concat([x_train, y_train], axis=1).to_csv('data/红酒品质分类-train.csv')    pd.concat([x_test, y_test], axis=1).to_csv('data/红酒品质分类-test.csv')

## Slide 85

**标题**: xgb案例：红酒品质分类
基本程
def dm02_训练模型():    # 1 加载数据集    train_data = pd.read_csv('./data/红酒品质分类-train.csv')    test_data = pd.read_csv('./data/红酒品质分类-test.csv')    # 2 准备数据 训练集测试集    x_train = train_data.iloc[:, :-1]    y_train = train_data.iloc[:, -1]    x_test = test_data.iloc[:, :-1]    y_test = test_data.iloc[:, -1]    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)    # 3 xgb模型训练    estimator = xgb.XGBClassifier(n_estimators=100, objective='multi:softmax',                      eval_metric='merror', eta=0.1, use_label_encoder=False, random_state=22)    estimator.fit(x_train, y_train)    # 4 xgb模型评估    y_pred = estimator.predict(x_test)    print( classification_report(y_true=y_test, y_pred=y_pred))    # 5 模型保存    joblib.dump(estimator, './data/mymodelxgboost.pth')

## Slide 86

**标题**: xgb案例：红酒品质分类
from sklearn.utils import class_weightdef dm03_训练模型():    # 1 加载数据集    train_data = pd.read_csv(‘./data/红酒品质分类-train.csv’)    test_data = pd.read_csv(‘./data/红酒品质分类-test.csv’)    # 2 准备数据 训练集测试集    x_train = train_data.iloc[:, :-1]    y_train = train_data.iloc[:, -1]    x_test = test_data.iloc[:, :-1]    y_test = test_data.iloc[:, -1]    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)    # 2-2 样本不均衡问题处理    classes_weights = class_weight.compute_sample_weight(class_weight=‘balanced’, y=y_train)    # 3 xgb模型训练    estimator = xgb.XGBClassifier(n_estimators=100, objective=‘multi:softmax’,                                  eval_metric=‘merror’, eta=0.1, use_label_encoder=False, random_state=22)    # 训练的时候，指定样本的权重    estimator.fit(x_train, y_train, sample_weight=classes_weights)    # 4 xgb模型评估    y_pred = estimator.predict(x_test)    print(classification_report(y_true=y_test, y_pred=y_pred))

## Slide 87

**标题**: xgb案例：红酒品质分类
from sklearn.model_selection import StratifiedKFoldfrom sklearn.model_selection import GridSearchCVdef dm04_交叉验证网格搜索():    # 1 加载数据集    train_data = pd.read_csv('./data/红酒品质分类-train.csv')    test_data = pd.read_csv('./data/红酒品质分类-test.csv')    # 2 准备数据 训练集测试集    x_train = train_data.iloc[:, :-1]    y_train = train_data.iloc[:, -1]    x_test = test_data.iloc[:, :-1]    y_test = test_data.iloc[:, -1]    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)    # 3 交叉验证时,采用分层抽取    spliter = StratifiedKFold(n_splits=5, shuffle=True)
# 4 模型训练    # 4-1定义超参数    param_grid = {‘max_depth’: np.arange(3, 5, 1),                  ‘n_estimators’: np.arange(50, 150, 50),                  ‘eta’: np.arange(0.1, 1, 0.3)}    # 4-2 实例化xgb    estimator = xgb.XGBClassifier(n_estimators=100,                                  objective=‘multi:softmax’,                                  eval_metric=‘merror’,                                  eta=0.1,                                  use_label_encoder=False,                                  random_state=22)    # 4-2 实例化cv工具    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=spliter)       # 4-3 训练模型    estimator.fit(x_train, y_train)    # 5 模型评估    y_pred = estimator.predict(x_test)    print(classification_report(y_true=y_test, y_pred=y_pred))    print(‘estimator.best_estimator_-->’, estimator.best_estimator_)    print(‘estimator.best_params_-->’, estimator.best_params_)

## Slide 88

1 xgboost的目标函数
2 xgboost的模型复杂度
3 xgboost API
XGBClassifier(n_estimators, max_depth, learning_rate, objective)

## Slide 89

1、下列关于XGBoost的描述错误的是？
A）它是极限梯度提升树（Extreme Gradient Boosting）的缩写
B）它在数据挖掘方面拥有更好的性能
C）Xgbboost使用了正则化
D）xgboost算法不可以使用线性模型进行集成
答案：D
2、下列关于XGBoost损失函数的正则化项描述错误的是？
A）它使用的是CART回归树作为弱学习器
B）它的正则化项只包含一棵树的结果
C）它的正则化项由树的叶子节点的个数以及L2正则化项组成
D）模型可以通过超参数来调整正则化项对模型的惩罚力度
答案：B

## Slide 90

答案：A
答案：D
3、下列关于XGBoost损失函数的描述错误的是？
A）它的第T棵树的损失与第T-1棵树无关
B）在求第T棵树的结构时可将前T-1棵树的结构作为常数
C）它使用了二阶泰勒展开式去近似目标函数
D）最终得出的损失函数值越小代表模型的效果越好
4、下列关于XGBoost树的描述错误的是？
A）它可以使用打分函数确定某个节点是否能够继续分裂
B）它可以使用打分函数确定某个特征的最佳分割点
C）最大树深度和最小叶子节点样本数可以用来调节树结构
D）超参数gamma的大小对树结构没有影响

## Slide 91


## Slide 92

商品
一共82元
50元                   20元                 10元
提升树（残差树）
举例：
权重树
举例：

---

# 补充材料 (06集成学习补充材料.pptx)

## Slide 1

**标题**: 集成学习补充材料
闻璋正

## Slide 2

**标题**: 回归树的流程图
左侧均值 6.24                                                右侧均值8.91
平方和损失1.86                                          平方和损失0.0719
二叉树

## Slide 3

**标题**: GBDT（Gradient Boosting Decision Tree）

## Slide 4

**标题**: GBDT（Gradient Boosting Decision Tree）
训练过程：
y = 7 + ri
CART回归树们，给我修正值 ！！
年级  时长
因为首次预测不要太差
平方和误差最小

_备注_: 训练时 初始化为均值，也就是基础模型预测为均值  因为 平方和误差最小

## Slide 5

最佳切分点评判标准：（残差与均值之间）平方误差
用来衡量： 在整体上残差均值 是否贴近 各个残差值。
当前切分的平方误差：
左：(-2 +2)² = 0
右： (-1 -0.5)² + (0−0.5)² + (1−0.5)² + (2−0.5)² = 2.25 + 0.25 + 0.25 + 2.25 = 5
总平方误差 = 5
回归残差
第一棵树
分裂开始 ！！
比如以年级x=1.5来切分
当前尝试结果：分为2个群体，
课时的修正值分别是 -2 和 0.5
评判标准值 = 5
（对修正值大致猜测）
本树回归目标

## Slide 6

当前尝试结果：分为2个群体，
课时的修正值分别是 -1.5 和 1
比如以年级x=2.5来切分
其余年级 x=3.5、4.5来切分
请同学尝试  ！！
评判标准： （残差与均值之间）平方误差
左：(-2 +1.5)² + (-1 +1.5)² = 0.25 + 0.25 = 0.5
右：(0−1)² + (1−1)² + (2−1)² = 1 + 0 + 1 = 2
总平方误差 = 2.5
（对修正值大致猜测）
评判标准值 = 2.5

## Slide 7

**标题**: 寻找最佳切分点
年级 x                 1.5     2.5     3.5     4.5
平方误差：          5      2.5     2.5      5
当前尝试结果：分为2个群体，
课时的修正值分别是 -1.5 和 1
（对修正值大致猜测）
第一棵决策树修正完成 ！！
对于1、2年级，预测每天学习时长的修正值 = – 1.5
对于3、4、5年级，预测每天学习时长的修正值 = + 1

## Slide 8

**标题**: 效果：
当前尝试结果：分为2个群体，
课时的修正值分别是 -1.5 和 1
（对修正值大致猜测）
修正后的最新预测
5.5
5.5
8
8
8
修正后的
最新预测
初始猜测值 = 7
第一次修正值
（决策树残差拟合）
=
+
更接近标准答案
推理时刻：

## Slide 9

第二棵决策树
回归残差
分裂开始 ！！
重新开始：
按年级 x= 1.5、2.5、3.5、4.5来切分
本树回归目标
原
通过评判标准： （残差与均值之间）平方误差
判断最佳切分， 得到第二棵树的最佳切分点为 x = 3.5

## Slide 10

按年级 x= 3.5切分
-0.33
-0.33
-0.33
0.5
0.5
均值：-0.33
均值：0.5
修正后的最新预测
修正后的
最新预测
初始猜测值 = 7
第一次修正值
（第一棵决策树残差拟合）
=
+
第二次修正值
（第二棵决策树残差拟合）
+
5.17
5.17
7.67
8.5
8.5
上一次预测值
第二棵决策树修正完成 ！！
对于1、2、3年级，预测每天学习时长的修正值 = – 0.33
对于4、5年级，预测每天学习时长的修正值 = + 0.5
<3.5        >3.5
推理时刻：

## Slide 11

实际效果：
均值
第一棵树
拟合残差
给出修正
第二棵树
拟合残差
给出修正
更多棵树
最终预测
5.03
5.98
7.02
8.01
8.99
（第n棵决策树残差拟合）
基础模型
均值
均值， 各棵决策树的残差值
都会被模型记录用于模型推理。
推理时刻：

## Slide 12

推理时刻  x = 4.2 ：
均值
第一棵树
拟合残差
给出修正
第二棵树
拟合残差
给出修正
更多棵树
最终预测
5.03
5.98
7.02
8.01
8.99
（第n棵决策树残差拟合）
基础模型
均值
x=4.2时 ，根据决策树切分情况
给出对应的修正值
y = 7 + 1  + 0.5 + …… = 8.01
<2.5        >2.5
<3.5        >3.5

## Slide 13

在GBDT中，每棵树预测的是前一棵树的预测值的残差
为什么是预测残差 ？
GBDT 本质：
所以GBDT回归树的
预测实际上是负梯度
注意 学习率

_备注_: 类比  梯度更新公式

## Slide 14

在GDBT中，每棵树预测的是前一棵树的预测值的残差
为什么是预测残差 ？
负梯度怎么计算 ？？
所以GBDT回归树的
预测实际上是负梯度

## Slide 15

总结：

## Slide 16

假设我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分：
XGBoost（ Extreme Gradient Boosting ）

## Slide 17

利用不同的特征训练出tree1和tree2，两棵树的结论累加起来，从而更接近最终的结论
类似之前gbdt的原理（基础值+残差），两棵树的结论累加起来更接近 最终的结论。
XGBoost（ Extreme Gradient Boosting ）

## Slide 18

**标题**: XGBoost（ Extreme Gradient Boosting ）
但是问题是XGBoost的loss 更为复杂：
XGBoost的loss          =    损失           +             正则项
（不一定是平方损失）  （限制模型复杂度）
也希望将该loss降到最低。但是存在正则项，难以求负梯度，从而优化困难。
对策：化繁为简
利用泰勒展开公式来近似该loss表达式。

_备注_: 防止过拟合

## Slide 19

**标题**: XGBoost
泰勒展开
将一个函数在某一点处展开成无限项的多项式表达式       （ 用             的一些列表达式，等价于                     的值）
一阶泰勒展开
二阶泰勒展开

## Slide 20

可有可无
预测值： Wi
即叶子得分（权重）
f_t : 当前树的预测值
增益越高，损失函数降低越多

## Slide 21

特征x1       特征x2
计算gi  （一阶导 ）和  hi  （二阶导）
根据目标函数（loss）： 简化后
当损失函数是平方损失的时候，
有：
标签      随机猜测
系数2

## Slide 22

因此，计算出残差与 一阶导gi  和 二阶导hi
第一棵决策树进行修正
分裂开始 ！！
要找一个最佳特征以及分裂点（比如“天赋” ）。
需要一个评判标准。
复杂推导，可证：
可以利用  增益Gain 来寻找合适的分割线。
同时，损失函数实现最小化
x系数2
GBDT用的是平方和损失 ！！
弱学习器之间的通信

## Slide 23

略 复杂推导1：

## Slide 24

略 复杂推导2：

## Slide 25

左节点
样本统计
右节点
样本统计
分裂前父节点
样本统计
GR  和 HR 是各分支的样本的g 以及h 的求和值
先考察天赋特征：

## Slide 26

w1 ？             w2  w3 ？
学习率设定了更新的步伐
给出每个节点的回归值
之前GBDT 是残差均值
（分子无平方）
(得分）                     (得分)
(得分）

_备注_: 之前GBDT 是残差均值

## Slide 27

当前的局面：
第二棵决策树进行修正
换一个特征
分裂开始 ！！

## Slide 28

换特征 ：

## Slide 29

w2 ？             w1  w3 ？
计算权重：
进一步逼近
更多的决策树，可以重复利用特征构建。也可添加新特征。

## Slide 30
