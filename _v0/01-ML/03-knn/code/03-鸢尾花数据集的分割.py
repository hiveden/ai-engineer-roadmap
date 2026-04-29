'''
将数据集分割为训练集和测试集
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split    # 数据集切分工具


# 加载数据集
mydataset = load_iris()

# 数据集切分
# 参数1：数据集中的特征列（4列）      参数2：标签（150列，1行）
# 参3：测试集所占比例20%（随机抽）           参4：随机数种子
# 150样本 ： 训练集为120个 测试集为30个
x_train, x_test, y_train, y_test = train_test_split(mydataset.data, \
               mydataset.target, test_size=0.2, random_state=10, stratify=mydataset.target)
#  stratify = mydataset.target
# 依据全部数据集标签中各个类别的占比进行切分，保证切分完的训练集和测试集的标签占比一致

#               150     50     50     50

#        训练集          45     45      30
#        测试集          5      5       20

print(x_train.shape)  #  120,  4
print(x_test.shape)   #   30,  4
print(y_train.shape)  #       120
print(y_test.shape)   #        30