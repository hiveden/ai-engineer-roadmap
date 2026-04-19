"""
Demo 01 — 用房子的特征预测房价
你要看到的三件事：数据长什么样、模型怎么学、预测准不准
"""

import pandas as pd

# ── 第一步：拿到历史数据 ──────────────────────────────
# 加州房价数据集（sklearn 内置，不用联网）
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

raw = fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target  # 房价（单位：10万美元）

print("=== 数据长这样（前 5 行）===")
print(df.head())
print(f"\n总共 {len(df)} 条数据，{len(df.columns) - 1} 个特征\n")

# ── 第二步：分训练集和测试集 ────────────────────────────
# 80% 拿来学规律，20% 留着验证
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集：{len(X_train)} 条 | 测试集：{len(X_test)} 条\n")

# ── 第三步：学规律（训练）──────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("=== 模型学到的规律（每个特征的权重）===")
for name, weight in zip(X.columns, model.coef_):
    print(f"  {name:>12s} → {weight:+.4f}")
print(f"  {'偏置(底价)':>10s} → {model.intercept_:+.4f}")

# ── 第四步：对新数据做预测 ─────────────────────────────
y_pred = model.predict(X_test)

print("\n=== 预测 vs 真实（看前 10 条）===")
comparison = pd.DataFrame(
    {
        "真实房价": y_test.values[:10],
        "预测房价": y_pred[:10],
        "误差": y_pred[:10] - y_test.values[:10],
    }
).round(3)
print(comparison.to_string(index=False))

# 整体表现
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"\n平均每次预测偏差：{mae:.3f}（单位：10万美元，约 {mae * 10:.1f} 万美元）")
