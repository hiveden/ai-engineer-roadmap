# MNIST 手写数字识别 - 数据探索
# 数据集格式：CSV，42000 行 × 785 列
# 每行 = 1 张图（28×28 像素）+ 1 个标签
# 第 0 列是 label（0-9），其余 784 列是像素值（0-255）
#
# ⚠️ 注意：原始课程代码用的是 Windows 绝对路径 `D:\AI_itheima\...`
# macOS 跑请把数据集放在项目根 `assets/source-materials/` 或类似位置后改 path

import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


# 数据路径（按本机情况调整）
path = '../../../../assets/source-materials/手写数字识别.csv'   # 假设的相对路径
# Windows 原版：r'D:\AI_itheima\class\20260423_Shanghai_Fundamentals_of_Machine_Learning\code\datasets\手写数字识别.csv'

# 用 pandas 读 CSV
df = pd.read_csv(path)
# print(df)   # [42000 rows x 785 columns] - 42000 样本 × 784 像素 + 1 label

# iloc = integer location，按位置切片
# x = 所有行，第 1 列到末尾（784 个像素）
# y = 所有行，第 0 列（label）
x = df.iloc[:, 1:]      # shape (42000, 784)
y = df.iloc[:, 0]       # shape (42000,)

# 看类别分布（10 类，0~9）
print(y, Counter(y))
# Counter({1: 4684, 7: 4401, 3: 4351, ...})  - 类别基本平衡，每类 4000+ 样本

# 把某一行（784 像素的扁平向量）还原成 28×28 二维图像
idx = 10000
data = x.iloc[idx].values.reshape(28, 28)   # .values 把 pandas Series 转 numpy ndarray

print(data.shape, type(data))   # (28, 28) <class 'numpy.ndarray'>

# 用 matplotlib 显示灰度图
plt.imshow(data, cmap='gray')   # cmap='gray' = 灰度模式（默认是 viridis 彩色）
plt.show()
