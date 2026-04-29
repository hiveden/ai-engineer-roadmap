# 复习题：KNN 性能基准（brute-force timing）

> 对应代码：[`step1_brute_force_timing.py`](./step1_brute_force_timing.py)
> 用法：盖住 hint，先口头/手写答主问；卡住再看 hint；最后翻 [`../../99-DISTILL.md`](../../99-DISTILL.md) 对答案。

---

## A. 数据与流程

### Q1 `load_digits` 数据集是什么？X.shape 的两个维度分别意味着什么？
- 总样本数？特征维度？
- 64 维特征怎么来的？（提示：8×8 手写数字图）
- y 的取值范围？
- 这数据集对 KNN 是友好还是不友好？为什么？

### Q2 `train_test_split` 的 `random_state=42` 起什么作用？
- 不写 random_state 会怎样？
- 为什么学习/调参阶段需要固定它？（可复现）
- 生产模型训练时还需要固定吗？
- `test_size=0.2` 在小数据集（1797 样本）上够不够稳？怎么补救？（提示：交叉验证）

---

## B. lazy learner 的时间分布

### Q3 为什么 `fit` 耗时几乎是 0 ms？
- KNN 的 fit 实际做了什么？
- 这跟 lazy learner 的定义怎么对应？
- 如果 `algorithm='kd_tree'`，fit 还会是 0 ms 吗？

### Q4 为什么 `predict` 是耗时大头？
- brute-force 模式下 predict 一条样本的复杂度？（提示：O(n_train · d)）
- 360 条测试样本一起预测的总复杂度？
- 这个时间能不能拆成"算距离"和"找 top-k"两段？哪段更贵？
- 想象训练集放大 100 倍（17 万样本），predict 时间会怎么变？

---

## C. algorithm 选择

### Q5 这段代码没指定 `algorithm`，默认 `'auto'` 在 digits 数据上会选哪个？为什么？
- digits 是 64 维——KD tree / Ball tree 在这个维度还有效吗？
- `'auto'` 的启发式规则大概怎么判断？（特征数阈值 / 样本数阈值）
- 怎么验证默认选了什么？（提示：`model._fit_method`）

### Q6 如果硬切 `algorithm='kd_tree'` 在 64 维上跑会怎样？
- 比 brute 快还是慢？为什么？
- fit 阶段会发生什么？（建树本身就贵）
- 哪个维度阈值之后 KD tree 开始"反向加速失败"？

---

## D. 工程加速手段

### Q7 这段代码能怎么加速？按"投入产出比"排序你会先做哪个？
- 加 `n_jobs=-1`：哪个阶段会受益？（fit 还是 predict？）
- 切 `algorithm='ball_tree'`：在 64 维有用吗？
- 先 PCA 降到 20 维再 KNN：影响精度吗？
- 换 ANN（近似最近邻）库（faiss / annoy）：什么场景值得？

### Q8 单条预测延迟和批量预测吞吐有什么区别？
- 这段代码量的是"360 条总耗时"，单条平均延迟是多少？
- 如果生产场景是 QPS=1000 的实时推理，这个延迟够用吗？
- KNN 在哪种场景反而合适？（小数据 / 离线批处理 / 低 QPS）

---

## E. 度量陷阱

### Q9 用 `time.time()` 量 ms 级耗时靠谱吗？
- 系统调用的精度问题（提示：`time.perf_counter()`）
- 单次量测的方差有多大？怎么改成 N 次取均值/中位数？
- 测试集刚加载完缓存是热的，第二次 predict 会不会更快？

### Q10 这段没量准确率（accuracy）——但调优不能只看速度。补一行 accuracy 该怎么写？
- 用 `model.score(X_te, y_te)` 还是 `accuracy_score(y_te, y_pred)`？
- 默认 `n_neighbors=5` 在 digits 上大概什么准确率？（凭直觉估）
- 速度 vs 准确率怎么联合评估？（提示：cv_results_ 多指标）

---

## 答题状态追踪

- [ ] Q1 load_digits / X.shape
- [ ] Q2 random_state
- [ ] Q3 fit 几乎 0
- [ ] Q4 predict 是大头
- [ ] Q5 默认 algorithm
- [ ] Q6 KD tree 高维退化
- [ ] Q7 加速手段
- [ ] Q8 延迟 vs 吞吐
- [ ] Q9 计时精度
- [ ] Q10 准确率配套
