"""
Step 4 — 预测：拿模型没见过的数据试试
问自己：预测得准吗？误差大吗？能接受吗？

代码知识点：
──────────────────────────────────────────────
y_pred = model.predict(X_test)
  — 用训练好的模型对测试集做预测
  — 输入：X_test（4128 条数据，每条 8 个特征）
  — 输出：y_pred（4128 个预测房价）
  — 模型内部做的事：每条数据的 8 个特征 × 对应权重 + 偏置 = 预测值

y_test.values[i]
  — .values 把 pandas Series 转成 numpy 数组
  — [i] 取第 i 个元素

from sklearn.metrics import mean_absolute_error
  — 导入 MAE（平均绝对误差）计算函数

mae = mean_absolute_error(y_test, y_pred)
  — 计算所有预测的平均误差
  — 公式：把每条的 |预测值 - 真实值| 加起来，除以总条数
  — 类比：你让新员工估 4128 套房的价，平均每套偏了多少万

y_test.mean()
  — 测试集房价的平均值，用来算误差占比
  — 误差占比 = mae / 平均房价 × 100%，衡量"偏得离不离谱"

"██" * int(abs(err) * 5)
  — 可视化误差大小，误差越大色块越长
  — abs() 取绝对值，int() 取整，字符串 * N 重复 N 次
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

raw = fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 用测试集预测
y_pred = model.predict(X_test)

print("=== 预测 vs 真实（前 10 条）===")
for i in range(10):
    real = y_test.values[i]
    pred = y_pred[i]
    err = pred - real
    bar = "██" * int(abs(err) * 5)  # 误差越大条越长
    print(f"  真实 {real:.2f}  预测 {pred:.2f}  误差 {err:+.2f}  {bar}")

mae = mean_absolute_error(y_test, y_pred)
print(f"\n平均误差：{mae:.3f}（× 10万美元 = {mae * 10:.1f} 万美元）")
print(f"平均房价：{y_test.mean():.3f}（× 10万美元 = {y_test.mean() * 10:.1f} 万美元）")
print(f"误差占比：{mae / y_test.mean() * 100:.1f}%")
