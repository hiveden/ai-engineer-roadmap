"""
Step 4 — 评估：分类模型怎么衡量好不好
问自己：和 demo-01 的评估有什么区别？

对比：
  — 回归用"误差"（平均偏了多少万）
  — 分类用"准确率"（判对了几个 / 总共几个）

代码知识点：
──────────────────────────────────────────────
from sklearn.metrics import accuracy_score
  — 准确率 = 预测正确的数量 / 总数量
  — 例：114 条里判对 110 条 → 96.5%

from sklearn.metrics import classification_report
  — 更详细的报告，包含：
  — precision（精确率）：模型说"恶性"的里面，真的是恶性的有多少
  — recall（召回率）：真正是恶性的，模型抓出来了多少
  — 类比：precision = 宁可放过不要错杀，recall = 宁可错杀不要放过
  — 医疗场景下 recall 更重要——漏掉一个恶性比误判一个良性后果严重得多
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"准确率：{acc:.1%}（{int(acc * len(y_test))}/{len(y_test)} 判对了）\n")

print("=== 详细报告 ===")
print(classification_report(y_test, y_pred, target_names=["恶性(0)", "良性(1)"]))
