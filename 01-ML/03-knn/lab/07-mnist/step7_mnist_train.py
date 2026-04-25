# MNIST 手写数字识别 - 训练 + 持久化
# 流程：读 CSV → 像素归一化 → 切分 → 训练 KNN → 评估 → joblib.dump 保存
#
# 注意几个工程要点：
# 1. 像素归一化的"捷径"：x = x / 255（不用 MinMaxScaler，因为知道像素天然在 [0, 255]）
# 2. 模型保存用 joblib（pickle 增强版，对 numpy 数组更高效）
# 3. KNN 在 33600 样本上 fit 几乎瞬间，predict 8400 样本可能要几秒到几十秒（lazy 代价）

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


# 数据路径（按本机情况调整）
path = '../../../../assets/source-materials/手写数字识别.csv'

df = pd.read_csv(path)
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 像素归一化捷径：x / 255
# 等价于 MinMaxScaler 但省掉 fit 步骤
# 前提：知道像素值上限是 255（这是 PNG/JPG 的标准，可信）
x = x / 255.0    # 把 [0, 255] 压到 [0, 1]

# 切分
# stratify=y 保证 10 类比例一致（虽然类别接近均衡，依然加上）
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# 模型训练
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)   # 几乎瞬间（lazy learner）

# 评估
y_predict = model.predict(x_test)   # 这步慢（brute force O(n_train · n_test · d)）
print('准确率1：', accuracy_score(y_test, y_predict))
print('准确率2：', model.score(x_test, y_test))   # 等价

# 保存模型
# joblib.dump 比 pickle 对 numpy 数组优化得更好（压缩 + 内存映射友好）
# .pth 是 PyTorch 习惯，sklearn 社区更常用 .pkl 或 .joblib
joblib.dump(model, '手写数字识别.pth')

# ⚠️ 模型大小看一眼：会有 ~200MB
# 因为 KNN 没有真正"训练参数"，"模型"就是整个训练集
# 33600 样本 × 784 特征 × 8 bytes (float64) ≈ 211 MB
# 这是 lazy learner 部署成本的活实证 → 见 00-basicapi answer Q10
