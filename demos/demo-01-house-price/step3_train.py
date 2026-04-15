"""
Step 3 — 训练：让模型从训练数据里学规律
问自己：模型学到了什么？这些数字是什么意思？

代码知识点：
──────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
  — 导入线性回归模型类
  — sklearn 里每个算法都是一个类，用法都一样：创建 → fit → predict

model = LinearRegression()
  — 创建一个线性回归模型实例
  — 此时模型是空的，什么都没学

model.fit(X_train, y_train)
  — 训练。把训练数据喂给模型，让它学规律
  — X_train = 输入特征（8 列数字），y_train = 对应的真实房价
  — fit 之后，模型内部就存了一组权重（每个特征的影响力）
  — 类比：你给一个新员工看 16512 个历史案例，让他总结出规律

model.coef_
  — 训练完之后的权重数组，长度 = 特征数（8 个）
  — 每个值表示：这个特征每增加 1，房价变化多少
  — 正数 = 正向影响（特征越大房价越高），负数 = 反向
  — 例：MedInc（收入）权重 +0.45 → 收入中位数每涨 1 万，房价涨 0.45 × 10万 = 4.5 万

model.intercept_
  — 偏置（截距），所有特征都为 0 时的"底价"
  — 实际意义不大（所有特征不可能全为 0），但数学上需要

zip(X.columns, model.coef_)
  — 把列名和权重一一配对
  — zip 是 Python 内置函数，把两个列表按位置合并成元组
  — 类比 JS：X.columns.map((name, i) => [name, model.coef_[i]])

f"{weight:+.4f}"
  — 格式化输出：+ 强制显示正负号，.4f 保留 4 位小数
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

raw = fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 两行代码：创建模型，喂数据
model = LinearRegression()
model.fit(X_train, y_train)

print("=== 模型学到的东西 ===")
print("每个特征对房价的影响力（权重）：")
for name, weight in zip(X.columns, model.coef_):
    direction = "↑ 正向" if weight > 0 else "↓ 反向"
    print(f"  {name:>12s}  {weight:+.4f}  ({direction})")

print(f"\n基础价：{model.intercept_:+.4f}")
print("\n意思是：房价 = 基础价 + 每个特征 × 对应权重")
