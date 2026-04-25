''' 案例 手写数字识别
模型的训练和保存

步骤
1 加载数据
2 数据的预处理.
    a 抽取特征 和 标签列
    b 归一化
    c 拆分数据集
4 模型训练
5 模型评估
6 模型保存
'''
import matplotlib.pyplot as plt  #画图库
import pandas as pd              #pandas库 处理表格数据
from sklearn.model_selection import train_test_split   #训练集和测试集切分
from sklearn.neighbors import KNeighborsClassifier   # KNN分类器
import joblib                    # 模型保存
from collections import Counter  # 数据分布的统计信息
from sklearn.metrics import accuracy_score         # 模型评估，计算模型的准确率



# 数据读取，定义文件路径
path=r'D:\AI_itheima\class\20260423_Shanghai_Fundamentals_of_Machine_Learning\code\datasets'
df = pd.read_csv(path+r'\手写数字识别.csv')

#print(df)  # [42000 rows x 785 columns]  42000个样本， 784个特征， 1个标签

# 提取特征和标签   index location
x = df.iloc[: ,  1:]    # 42000， 784列
y = df.iloc[:,0]

# 归一化
x = x/255.0   # 就是归一化公式  x-min/(max-min)  ， min = 0

# 拆分数据集为训练集和测试集
# 参1：特征数据集     参2：标签数据集     参3：测试集所占比例（随机抽）     参4：随机数种子
# 参5：是否打乱数据集  参6 ： stratify  是否按照标签比例进行训练集和测试集的切分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, \
           random_state=42, shuffle=True, stratify=y)
# 模型定义
estimator = KNeighborsClassifier(n_neighbors=5)

# 模型训练
estimator.fit(x_train, y_train)

# 模型评估
y_predict = estimator.predict(x_test)
print('准确率1：', accuracy_score(y_test, y_predict))

#方法二
print('准确率2：', estimator.score(x_test, y_test))

# 保存模型
# 参1 ：模型对象    参2 ：保存的文件名
joblib.dump(estimator, '手写数字识别.pth')