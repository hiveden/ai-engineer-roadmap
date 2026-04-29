# 复习题：数据集分割 + 分层抽样

> 对应代码：[`step4_split_stratify.py`](./step4_split_stratify.py)
> 范围：`train_test_split` API / random_state / stratify

---

## A. train_test_split 基础

### Q1 `train_test_split(X, y, test_size=0.2, ...)` 返回什么？为啥是 4 个值？
- 返回顺序：`X_train, X_test, y_train, y_test`（顺序不能记错！）
- 为什么不返回 2 个 tuple `(X_train, y_train), (X_test, y_test)`？
- 一行解构常出错——如果写成 `X_train, X_test = train_test_split(X, y)` 会发生什么？

### Q2 `test_size=0.2` 和 `test_size=30` 有什么区别？
- 浮点数：比例
- 整数：绝对样本数
- `train_size` 参数也存在，可以两个都设吗？

### Q3 `random_state=10` 起什么作用？换成 `random_state=42` 会怎样？
- 固定随机种子的目的
- 不写 random_state 默认行为？（提示：每次都不同）
- 学习/调参阶段为啥必须固定？
- 生产正式训练时呢？（不一定要固定，看场景）

---

## B. stratify 分层抽样（分类任务核心）

### Q4 `stratify=mydataset.target` 在做什么？不加会怎样？
- 分层抽样的定义：保证训练集和测试集**类别比例**和原始数据集一致
- 不加 stratify 的极端情况：测试集可能全是某一类，模型评估失真
- iris 类别完全均衡（50/50/50），加 stratify 几乎无差别——但养成习惯

### Q5 假设你有个二分类数据集，正样本 1%，负样本 99%（典型风控/反欺诈场景）。不加 stratify 切 80/20，会发生什么？
- 测试集大小：n × 0.2
- 测试集中正样本期望数：n × 0.2 × 0.01
- 实际抽样会有方差，可能抽到 0 个正样本——后果？
- 模型评估指标（accuracy / precision / recall）会怎样？

### Q6 stratify 只能传 y 吗？能不能按其他列分层？
- 可以传任意 1D 数组
- 多分类多标签时怎么办？（提示：需要先合并多列变成"组合标签"，或者用 `iterative-stratification` 库）
- 时间序列数据能用 stratify 吗？（不能！时间数据要 `TimeSeriesSplit`）

---

## C. 切分前后的检验

### Q7 切完之后第一件事检查什么？
- shape 检查：`x_train.shape[0] + x_test.shape[0] == X.shape[0]`
- 类别分布检查：`Counter(y_train)` vs `Counter(y_test)` 比例是否一致
- 特征分布检查（可选）：训练/测试集的 mean/std 接近吗？（提示：`covariate shift`）

### Q8 如果切完发现训练集和测试集**特征分布**差很多怎么办？（不是类别分布）
- 这种现象叫什么？（covariate shift）
- 可能原因：数据收集时间不同 / 设备差异 / random_state 运气不好
- 解决思路：换 random_state / 增加数据量 / 用 cross-validation 多次切

---

## D. 工程坑

### Q9 `shuffle=False` 是什么？什么场景需要关掉打乱？
- 默认 `shuffle=True`：切分前先打乱顺序
- 关闭场景：时间序列（必须按时间顺序切，前 80% 训练后 20% 测试）
- 关闭 + stratify 不能同时用，为什么？

### Q10 train / val / test 三段式怎么用 train_test_split 实现？
- 一次切不出来，得连切两次：
  ```python
  X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, ...)
  X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, ...)
  # 0.8 × 0.25 = 0.2，所以最终 60/20/20
  ```
- 第二次切的 test_size 0.25 是怎么算的？
- 为啥工程上要 train/val/test 三段而不是 train/test 两段？

---

## 答题状态

- [ ] Q1 返回 4 个值
- [ ] Q2 test_size 浮点 vs 整数
- [ ] Q3 random_state
- [ ] Q4 stratify 是啥
- [ ] Q5 不平衡分类的极端情况
- [ ] Q6 stratify 限制
- [ ] Q7 切分后检查
- [ ] Q8 covariate shift
- [ ] Q9 shuffle=False
- [ ] Q10 三段式切分
