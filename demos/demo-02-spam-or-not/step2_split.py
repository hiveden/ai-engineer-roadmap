"""
Step 2 — 切分数据
和 demo-01 完全一样的流程：特征/标签分离 → 训练集/测试集拆分

代码知识点：
──────────────────────────────────────────────
和 demo-01 的 step2 一模一样，只是数据换了。
ML 的流程是通用的，不管什么数据集都是这几步。
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"全部数据：{len(df)} 条")
print(f"训练集：  {len(X_train)} 条")
print(f"测试集：  {len(X_test)} 条")

print(f"\n=== 训练集标签分布 ===")
counts = y_train.value_counts()
print(f"  良性(1)：{counts[1]} 例")
print(f"  恶性(0)：{counts[0]} 例")
