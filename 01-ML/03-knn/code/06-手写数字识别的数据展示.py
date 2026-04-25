''' 案例 手写数字识别 图片的读取和展示
像素值成为特征
顺便学习 pandas dataframe库的使用
'''

import matplotlib.pyplot as plt  #画图库
import pandas as pd              #pandas库 处理表格数据
from sklearn.model_selection import train_test_split   #训练集和测试集切分
from sklearn.neighbors import KNeighborsClassifier   # KNN分类器
import joblib                    # 模型保存
from collections import Counter  # 数据分布的统计信息


# 数据读取，定义文件路径
path=r'D:\AI_itheima\class\20260423_Shanghai_Fundamentals_of_Machine_Learning\code\datasets'
df = pd.read_csv(path+r'\手写数字识别.csv')

#print(df)  # [42000 rows x 785 columns]  42000个样本， 784个特征， 1个标签

# 提取特征和标签   index location
x = df.iloc[: ,  1:]    # 42000， 784列
y = df.iloc[:,0]

print(y, Counter(y))
#Counter({1: 4684, 7: 4401, 3: 4351, 9: 4188, 2: 4177, 6: 4137, 0: 4132, 4: 4072, 8: 4063, 5: 3795})

# 将某个样本 （784列）  -->  28x28的图像，并进行展示
idx = 10000
# 将第10000行的数据的值提取出来，然后转换成28x28的图像
data = x.iloc[idx].values.reshape(28,28)

print(data.shape,type(data))   #  (28, 28) <class 'numpy.ndarray'>

# 展示图像
plt.imshow(data, cmap='gray')  # 参1 ：图像数据  参2：颜色模式
plt.show()

