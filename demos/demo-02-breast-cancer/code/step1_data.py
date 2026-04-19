"""
Step 1 — 看数据长什么样
问自己：每一行是什么？每一列是什么？target 列是什么？

数据集：乳腺癌诊断（sklearn 内置）
  — 569 条数据，每条是一个肿瘤的检测结果
  — 30 个特征（肿瘤的各种测量值：半径、纹理、周长、面积等）
  — 目标：0 = 恶性，1 = 良性
  — 这是一个二分类问题：给一组检测数据，判断"恶性还是良性"

对比 demo-01：
  — demo-01（回归）：输出是一个数字（房价 2.35）
  — demo-02（分类）：输出是一个类别（0 恶性 / 1 良性）

代码知识点：
──────────────────────────────────────────────
load_breast_cancer()
  — 和 fetch_california_housing() 用法一样，返回 Bunch 对象
  — .data → 特征矩阵 (569, 30)
  — .target → 目标数组 (569,)，值只有 0 和 1
  — .target_names → ['malignant', 'benign']（恶性、良性）
  — .feature_names → 30 个特征的名字

df.shape
  — 返回 (行数, 列数) 的元组

df["Diagnosis"].value_counts()
  — 统计某列中每个值出现了多少次
  — 类比 SQL：SELECT Diagnosis, COUNT(*) FROM df GROUP BY Diagnosis
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target  # 0=恶性, 1=良性

print("=== 前 5 行（只显示前 6 列 + 诊断结果）===")
print(df.iloc[:5, list(range(6)) + [-1]])

print(f"\n总共 {df.shape[0]} 行，{df.shape[1]} 列")

print("\n=== 诊断分布 ===")
counts = df["Diagnosis"].value_counts()
print(f"  良性(1)：{counts[1]} 例")
print(f"  恶性(0)：{counts[0]} 例")
