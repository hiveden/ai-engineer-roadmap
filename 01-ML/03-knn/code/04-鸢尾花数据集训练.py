'''
鸢尾花数据集的KNN模型训练 以及 评估
步骤:
1加载数据
2数据预处理(清洗）
3特征工程(提取，归一化标准化）
4模型训练
5模型评估
6模型预测
'''

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score    #  模型评估工具

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

#  todo 4模型训练
# 实例化KNN分类器模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

#  todo 5模型评估
y_predict = model.predict(x_test)
print('预测结果：', y_predict)
print('标签：', y_test)

# 给模型评估打分   输入测试集的标签 和 测试集的预测结果
myscore1 = accuracy_score(y_test, y_predict)
myscore2 = model.score(x_test, y_test)   # 模型评估方法2，输入 测试集特征和标签

print('准确率1：', myscore1)
print('准确率2：', myscore2)

#  todo 6模型预测
x_ceshi = [[3, 5, 4, 2]]   # 创建预测数据
x_ceshi = transfer.transform(x_ceshi)   # 对测试数据进行标准化
y_predict = model.predict(x_ceshi)
print('预测结果：', y_predict)

y_predict_probability = model.predict_proba(x_ceshi)
print('预测结果的概率分布：', y_predict_probability)        #  90%      5%       5%


