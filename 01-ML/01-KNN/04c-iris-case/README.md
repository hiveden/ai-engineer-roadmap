# 第 4 章 c · 鸢尾花分类案例

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-鸢尾花案例.md`](./01-鸢尾花案例.md) — ★ 掌握：KNN + 标准化完整 6 步实战

## 互动 demo

| demo | 内容 | 启动 |
|---|---|---|
| [`demos/01-iris-pipeline.py`](./demos/01-iris-pipeline.py) | 6 步 pipeline 拖参（test_size/k/seed/标准化）+ 4D vs 2D 准确率对比 + 决策边界 + 混淆矩阵 + 分类报告 | `marimo run demos/01-iris-pipeline.py --port 2725` |

## 学完掌握

- 标准 ML pipeline 6 步：load → split[stratify] → scale → fit → eval → predict
- `train_test_split(stratify=y)` 保证类别比例平衡，类不平衡时**必做**
- 4 维全特征 vs 2 维投影准确率差距体现"维度信息量"
- 混淆矩阵的 TP/FN/FP/TN 4 格 + classification_report 的 precision/recall/f1
- KNN 在 iris 上的局限：花瓣长×花瓣宽几乎线性可分，KNN 优势不显著（树模型也行）

## 底稿

> 04 · 【实操】利用 KNN 算法进行鸢尾花分类

鸢尾花 Iris Dataset 数据集是机器学习领域经典数据集，包含 150 条鸢尾花信息，每 50 条取自三个鸢尾花中之一：Versicolour、Setosa 和 Virginica。每个花用 4 个属性描述。

代码实现：

```python
'''
鸢尾花数据集的 KNN 模型训练 以及 评估
步骤:
1 加载数据
2 数据预处理（清洗）
3 特征工程（提取，归一化标准化）
4 模型训练
5 模型评估
6 模型预测
'''

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mydataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(
    mydataset.data, mydataset.target,
    test_size=0.3, random_state=22, stratify=mydataset.target
)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
myscore1 = accuracy_score(y_test, y_predict)
myscore2 = model.score(x_test, y_test)
print('准确率 1：', myscore1)
print('准确率 2：', myscore2)

x_ceshi = [[3, 5, 4, 2]]
x_ceshi = transfer.transform(x_ceshi)
y_predict = model.predict(x_ceshi)
y_predict_probability = model.predict_proba(x_ceshi)
print('预测结果：', y_predict)
print('预测结果的概率分布：', y_predict_probability)
```

→ 完整含原版注释见 [`01-鸢尾花案例.md`](./01-鸢尾花案例.md#底稿)。
