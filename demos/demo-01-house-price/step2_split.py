"""
Step 2 — 把数据分成训练集和测试集
问自己：为什么要分？如果不分会怎样？

代码知识点：
──────────────────────────────────────────────
from sklearn.model_selection import train_test_split
  — sklearn 提供的数据拆分工具函数

X = df.drop("Price", axis=1)
  — 从 DataFrame 删掉 "Price" 列，剩下的就是输入特征
  — axis=1 表示按列操作（axis=0 是按行）
  — 不会修改原始 df，返回一个新的 DataFrame
  — ML 惯例：大写 X = 输入特征矩阵

y = df["Price"]
  — 取出 Price 列作为输出目标
  — ML 惯例：小写 y = 要预测的目标值
  — 返回一个 Series（一列数据，带索引）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  — 把 X 和 y 同时拆成训练集和测试集
  — test_size=0.2 → 20% 做测试，80% 做训练
  — random_state=42 → 随机种子，保证每次运行拆出来的数据一样（可复现）
  — 返回 4 个值：训练输入、测试输入、训练输出、测试输出
  — 类比：函数返回一个元组，Python 支持解构赋值（类似 JS 的 const [a, b] = fn()）

为什么要分？
  — 训练集：模型从这些数据里学规律
  — 测试集：模型没见过的数据，用来验证它学得好不好
  — 如果不分，模型可能"背答案"——在训练数据上表现完美，遇到新数据就翻车（过拟合）
  — 类比：考试不能用原题考，否则分辨不出真会还是死记硬背

X_train.head(3)
  — 看训练集前 3 行

y_train.head(3)
  — 看这 3 行对应的真实房价
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

raw = fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target

# 输入 = 8 个特征列，输出 = Price
X = df.drop("Price", axis=1)
y = df["Price"]

# 80% 训练，20% 测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"全部数据：{len(df)} 条")
print(f"训练集：  {len(X_train)} 条（拿来学规律）")
print(f"测试集：  {len(X_test)} 条（拿来验证学得好不好）")

print(f"\n=== 训练集前 3 行（输入）===")
print(X_train.head(3))
print(f"\n=== 对应的房价（输出）===")
print(y_train.head(3))
