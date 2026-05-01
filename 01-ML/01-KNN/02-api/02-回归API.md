# 回归 API：KNeighborsRegressor

## 课程原文

```python
sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
```

---

## 直觉：换个类就是回归

回归预测的是 **数值**（如"你给电影打几分"），不是类别。在 sklearn 里只需要把 `KNeighborsClassifier` 换成 `KNeighborsRegressor`，**API 形态完全一致**。

## 最简回归 demo：预测打几分

延续 [01-intro 数据集](../01-intro/02-接近程度.md)：9 部已知电影 + 流浪地球 3。

预测"你给流浪地球 3 打几分" —— 标签从分类标签换成数值：

```python
from sklearn.neighbors import KNeighborsRegressor

# X 同 [01-分类API](./01-分类API.md)（9 部电影特征不变）
X = [[8.3, 9.0], [7.8, 8.5], [8.0, 9.5],
     [6.4, 7.0], [5.5, 4.0],
     [8.5, 8.0], [8.4, 9.2],
     [7.2, 6.5], [7.4, 8.0]]

# y 改成：你历史给每部电影打过的具体星数（数值）
y = [4.5, 4.0, 4.3,   # 流浪地球 2 / 阿凡达 / 泰坦尼克
     3.0, 1.5,         # 哥斯拉 / 喜羊羊
     4.5, 4.5,         # 沙丘 2 / 复仇者联盟 4
     3.0, 3.5]         # 寒战 / 长津湖

model = KNeighborsRegressor(n_neighbors=5)  # ← 唯一区别
model.fit(X, y)
prediction = model.predict([[8.5, 9.5]])
print(prediction)
# 输出: [4.36]（预测打约 4.36 分）
```

## 与分类 API 的差异

| 对比项 | 分类 | 回归 |
|---|---|---|
| API 类 | `KNeighborsClassifier` | `KNeighborsRegressor` |
| 标签 y | 离散类别（如 `'喜欢'` / `'不喜欢'`）| 连续数值（如 `4.5` / `3.0`）|
| 内部第 4 步 | 多数表决（投票）| 平均（K 个邻居数值的均值）|
| 输出 | 类别标签 | 数值 |
| 三件套 | 构造 / fit / predict | **完全一致** |

→ 这就是 [第 1 章工作流程](../01-intro/05-工作流程.md) 强调的"分类/回归只差最后一步"在代码层面的体现。

## 术语版

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 回归任务的目标值 | **y**（小写，**目标** targets）| — |
| K 个邻居数值的均值 | **平均** mean / average | [01-intro 05](../01-intro/05-工作流程.md) |

**关键启示**：`KNeighborsRegressor` 是 `KNeighborsClassifier` 的回归版本，API 形态完全一致 —— 唯一区别是 sklearn 内部用平均代替投票。

→ 02-api 默认用 **欧式距离**。如果想用其他距离公式，通过 `metric` 参数指定 —— 见 [第 3 章 距离度量](../03-distance/)。
