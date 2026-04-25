'''将鸢尾花数据集进行可视化，画散点图'''

import seaborn as sns   #  绘制散点图的工具
import matplotlib.pyplot as plt     # 画图工具
import pandas as pd
from sklearn.datasets import load_iris    # 鸢尾花数据集

# 加载数据集
mydataset = load_iris()

print(type(mydataset.data)) #  <class 'numpy.ndarray'>

#需要将数据从numpy 数组转为 pandas 表格形式， 才能用seaborn 画图
# 参1 数据集（numpy array）  参2 列名
data = pd.DataFrame(mydataset.data, columns=mydataset.feature_names)

#  对表格形式的数据新增一列label
data['label'] = mydataset.target  #  0 ,1 , 2
print(data)

# 通过seaborn画散点图
# 参1 数据表格  参2 x轴列名   参3 y轴列名    参4 色调（按类别分组，并对每一组上色）   参5：是否显示拟合线
#sns.lmplot(data=data, x='sepal length (cm)', y='sepal width (cm)', hue='label', fit_reg=False)
sns.lmplot(data=data, x='petal length (cm)', y='petal width (cm)', hue='label', fit_reg=False)

# 设置图的标题并显示
plt.title('Iris data')
plt.show()