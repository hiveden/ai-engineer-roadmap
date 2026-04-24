"""
Step 3 — 训练一个分类模型
问自己：和 demo-01 的训练代码有什么区别？

代码知识点：
──────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
  — 逻辑回归：名字带"回归"，但干的是分类
  — 输出是一个概率（0~1），超过 0.5 判良性，否则判恶性

和 demo-01 的区别：
  — demo-01：LinearRegression  → 输出一个数字（房价）
  — demo-02：LogisticRegression → 输出一个类别（0 或 1）
  — 用法完全一样：创建 → fit → predict

model.predict(X_test[:5])
  — 预测前 5 条，输出是 [1, 0, 1, 1, 0] 这种类别数组

model.predict_proba(X_test[:5])
  — 预测前 5 条的概率，每条返回 [恶性概率, 良性概率]
  — 例：[0.03, 0.97] 表示 97% 概率是良性

max_iter=10000
  — 最大迭代次数，模型内部优化的上限
  — 数据复杂时默认值(100)可能不够，会报警告，调大就行
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print("=== 模型对前 5 条测试数据的判断 ===")
sample_X = X_test[:5]
sample_y = y_test.values[:5]
predictions = model.predict(sample_X)
print("predictions =========== " + str(predictions))
probas = model.predict_proba(sample_X)
print("probas =============== " + str(probas))

label = {0: "恶性", 1: "良性"}
for i in range(5):
    confidence = max(probas[i]) * 100
    print(f"  真实: {label[sample_y[i]]}  预测: {label[predictions[i]]}  置信度: {confidence:.1f}%")
