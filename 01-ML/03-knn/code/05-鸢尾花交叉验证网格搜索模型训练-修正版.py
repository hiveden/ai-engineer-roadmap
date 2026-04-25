"""
交叉验证解释：
    概述：交叉验证是一种更完善、可信度更高的一种模型评估方法。
    思路：把数据集份N份，每次都去1份当做测试集，其他当做训练集，然后在给模型评分
    然后再用下一份当做测试集，其他当做训练集（皇帝轮流坐，今天到你家）。最后计算所有的
    均值，当模型最终评分。
    思路是：
        把数据分成N份，例如数据分成4份， 4折交叉验证
            第1次：把第1份数据作为验证集（测试集），其他当做训练集，训练模型，模型预测，获取：准确率-》准确率1
            第2次：把第2份数据作为验证集（测试集），其他当做训练集，训练模型，模型预测，获取：准确率-》准确率2
            第3次：把第3份数据作为验证集（测试集），其他当做训练集，训练模型，模型预测，获取：准确率-》准确率3
            第4次：把第4份数据作为验证集（测试集），其他当做训练集，训练模型，模型预测，获取：准确率-》准确率4
            然后计算上诉4次的准确率的平均值，作为模型最终准确率
            假设第4次最好（准确率最高），则：用全部数据（训练集+测试集）训练模型，再用第4次的测试集对模型进行测试
        目的：
        为了让模型最终结果更加准确
        好处：相比单一切分的训练集和测试集获取的评分，经过交叉验证可信度更高

网格搜索：
    概述：机器学习内置API ，一般结合交叉验证使用，通过API实现找到最优超参数
    目的：网络搜素+交叉验证 ： 让模型变得更强
"""

from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.model_selection import train_test_split, GridSearchCV #  训练和测试数据集分割   寻找最优超参（网格搜索+交叉验证）
from sklearn.preprocessing import StandardScaler         #  数据预处理  数据标准化
from sklearn.neighbors import KNeighborsClassifier         #模型， KNN分类器
from sklearn.metrics import accuracy_score         # 模型评估，计算模型的准确率
from sklearn.pipeline import Pipeline

#  复制到数据标准化

# 加载鸢尾花数据集
mydataset = load_iris()

# 数据预处理，切分数据集，比例 4:1
# 参1：数据集           参2：目标值            参3：测试集所占比例（随机抽）           参4：随机数种子
# 返回值：训练集特征，测试集特征，训练集标签，测试集标签
x_train, x_test, y_train, y_test = train_test_split(mydataset.data, mydataset.target, test_size=0.3, random_state=20)
x_train_, x_test_ = x_train.copy(), x_test.copy()

# 定义创建标准化对象
transfer = StandardScaler()

''' 标准做法， 利用管道工具 
    在交叉验证的每一折内部，独立进行标准化，
'''
pipe = Pipeline([
    ('std', transfer),            #  标准化工具
    ('knn2', KNeighborsClassifier())       #  分类器定义  不写超参，通过检索决定
])

# 参数搜索空间
param_dict = {  'knn2__n_neighbors': [1,3,5,7,9,11] }
        #  knn2__n_neighbors knn2是你的模型名字， 后面必须2个下划线，之后n_neighbors就是该模型固有参数

# 3. 使用网格搜索 (此时传入管道)
# 4.3 创建 网格搜索对象，  寻找最优参数，  （交叉验证+网格搜索）
# 参1：分类器模型对象 （管道）    参2：超参的可能出现的值     参3：每一组进行交叉验证的次数 cross validation
# 返回值： 经过处理后的模型对象
estimator = GridSearchCV(pipe, param_dict, cv=4)
estimator.fit(x_train, y_train)  # 此处直接传

# 4.5 打印最优参数组合
print(estimator.best_score_)     # 最优准确率             0.9333333333333333
print(estimator.best_params_)    # 最优参数组合           {'n_neighbors': 7}
print(estimator.best_estimator_) # 获得最优模型， 用用于后期推理   KNeighborsClassifier(n_neighbors=7)

# 具体的交叉验证中间结果， 可不查看
#每种参数组合都经过 4 次验证（split0 ～ split3）, 每次交叉验证有6个值（k从1~11），然后计算平均、标准差等统计量。
print(estimator.cv_results_)
print('='*20)

# 确定了最佳模型，  开启正式训练流程 ：
# 利用所有数据进行训练， 先对数据集进行标准化
# 数据集预处理-数据标准化     fit + transform  记录均值方差   针对训练集
x_train_ = transfer.fit_transform(x_train_)

# 让测试集的均值和方法, 转换测试集数据;   仅仅transform  针对测试集
x_test_ = transfer.transform(x_test_)

# 5 模型评估
# 5.1 “重新定义”最佳 模型  最优超参
estimator = KNeighborsClassifier(n_neighbors=7)
#或者 estimator = estimator.best_estimator_

# 5.2 使用所有训练集数据重新进行训练
estimator.fit(x_train_, y_train)
# 5.3 模型推理
y_predict = estimator.predict(x_test_)
# 5.4 模型评估
myscore = accuracy_score(y_test, y_predict)    # 用了全部训练集，因此准确率会更高。
print(myscore)