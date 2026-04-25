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



#  todo 步骤:
#  todo 1加载数据
mydataset = load_iris()

#  todo 2数据预处理(清洗）
#数据集分割
# 参数1：数据集中的特征列（4列）      参数2：标签（150列，1行）
# 参3：测试集所占比例20%（随机抽）           参4：随机数种子
# 150样本 ： 训练集为120个 测试集为30个
x_train, x_test, y_train, y_test = train_test_split(mydataset.data, mydataset.target, test_size=0.3, random_state=22, stratify=mydataset.target)

#  todo 3特征工程(提取，归一化标准化）
# 实例化标准化
transfer = StandardScaler()
# 先对训练集计算均值方差（并且保存这俩值），再进行标准化
x_train = transfer.fit_transform(x_train)

# 对于测试集，使用训练集的均值方差进行标准化
x_test = transfer.transform(x_test)

# 定义模型  不输入任何参数，通过网格搜索决定超参
model = KNeighborsClassifier()
# 定义参数搜索空间
param_dict = {  'n_neighbors': [1,3,5,7,9,11] }

# 创建网格搜索对象，  寻找最优参数，  （交叉验证+网格搜索）
# 参数1：模型对象    参2：超参的名称以及他们可能出现的值（字典格式）
# 参3：每一组进行交叉验证的次数 cross validation
estimator = GridSearchCV(model, param_dict, cv=4)

# 使用网格搜索对象进行训练
estimator.fit(x_train, y_train)

# 打印最优参数组合，以及最优准确率
print(estimator.best_score_)  # 最优准确率     0.9711538461538463
print(estimator.best_params_) # 最优参数组合   {'n_neighbors': 7}
print(estimator.best_estimator_) # 获得最优模型， 用用于后期推理  KNeighborsClassifier(n_neighbors=7)

# 如果希望查看中间结果（每一次训练的准确率值）
print(estimator.cv_results_)

# 用全部数据训练最优的模型
#estimator_ = KNeighborsClassifier(n_neighbors=7)
estimator_ = estimator.best_estimator_

estimator_.fit(x_train, y_train)

# 推理训练好的最优模型，并评估
y_predict = estimator_.predict(x_test)
myscore = accuracy_score(y_test, y_predict)

print(myscore)










