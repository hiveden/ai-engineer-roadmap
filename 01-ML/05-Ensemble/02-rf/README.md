# 第 2 章 · 随机森林

> 维度：**算法 + 代码**
>
> **按知识点拆分的讲解版**（待补）：
>
> 1. `01-算法思想.md` — 【理解】构建方法
> 2. `02-API.md` — 【知道】sklearn API
> 3. `03-泰坦尼克实践.md` — 【实践】生存预测

## 底稿

> 02 · 随机森林

**学习目标**：

1. 理解随机森林的构建方法
2. 知道随机森林的 API
3. 能够使用随机森林完成分类任务

> 【理解】算法思想

**Random Forest**：基于 Bagging 思想 + 决策树作为基学习器的集成算法。

**两层随机性**：

1. **样本随机**：有放回抽样产生每棵树的训练集（bootstrap）
2. **特征随机**：每个节点分裂时，从全部特征中随机选 $k$ 个再挑最优（一般 $k = \sqrt{n}$）

**预测**：分类用平权投票，回归用平均。

**为什么不剪枝也不过拟合**：两层随机保证树之间充分独立，方差被投票平均掉。

**两个关键超参**：

- 森林中树的数量 `n_estimators`（一般取较大值）
- 每节点候选特征数 `max_features`

**思考题速答**：

- 为什么要随机抽样？→ 不抽样则所有树训练集相同，结果完全一致，集成没意义
- 为什么要**有放回**？→ 无放回则各树训练集互斥、各自"片面"，差异过大反而难以聚合

> 【知道】随机森林 API

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=10,        # 树的数量
    criterion="gini",       # gini / entropy
    max_depth=None,         # 树最大深度
    max_features="auto",    # auto/sqrt = sqrt(n_features), log2, None
    bootstrap=True,         # 是否有放回抽样
    min_samples_split=2,    # 节点分裂最小样本数
    min_samples_leaf=1,     # 叶节点最小样本数
)
```

**调参直觉**：

- `n_estimators` ↑ → 稳但慢，到一定值后边际收益递减
- `max_depth` 不限制时树会长到底；样本量大时建议限制防 OOM
- `max_features` 越小树差异越大、方差越低、但偏差可能上升

> 【实践】泰坦尼克号生存预测

```python
# 流程骨架
# 1. 读 titanic.csv → 选特征 [Pclass, Age, Sex] → onehot
# 2. train_test_split
# 3. RandomForestClassifier().fit(X_train, y_train)
# 4. score / GridSearchCV 调 n_estimators + max_depth
```

**超参搜索**：用 `GridSearchCV` 在 `n_estimators ∈ {10, 50, 100, 200}`、`max_depth ∈ {3, 5, 8, None}` 上做 5 折 CV。
