#  展示鸢尾花数据集的内容

#导入数据集
from sklearn.datasets import load_iris

# 加载数据集
mydataset = load_iris()  # 字典格式
#print(mydataset, type(mydataset))

# 查看所有的键
print(mydataset.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])



# 展示数据集内容  特征和标签
print(mydataset.data, len(mydataset.data))   # 150个样本 4个特征

# 展示标签，以及 标签的名称
print(mydataset.target, mydataset.target_names)

# 展示4列特征的名称
print(mydataset.feature_names)

# 展示数据集的描述
print(mydataset.DESCR)



