# 复习题：鸢尾花 KNN 端到端流程

> 对应代码：[`step5_iris_pipeline.py`](./step5_iris_pipeline.py)
> 范围：标准 6 步流程 / accuracy_score vs model.score / predict_proba

---

## A. 6 步流程的"为什么"

### Q1 监督学习标准 6 步流程是什么？每步对应代码哪部分？
```
1. ?  →  2. ?  →  3. ?  →  4. ?  →  5. ?  →  6. ?
```
- 试着不看代码先默写
- 哪步可以省？哪步不能省？

### Q2 为什么"切分"必须在"特征工程"之前？
- 反着做（先 scaler 再切）会导致什么？（data leakage）
- 顺序逻辑：测试集要模拟"未知数据"，scaler 不能提前看到

### Q3 这份代码 step 3 的两行：
```python
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
```
为什么测试集只 `transform` 不 `fit_transform`？
- 复盘 `02-scaling` Q10 里的 data leakage 三选一
- 如果 `x_test = transfer.fit_transform(x_test)` 会怎样？

---

## B. 模型评估两种打分

### Q4 这两种写法等价吗？为什么有两种 API？
```python
myscore1 = accuracy_score(y_test, y_predict)   # 函数式
myscore2 = model.score(x_test, y_test)         # 方法式
```
- 输入参数差异？
- 内部行为差异？（`model.score` 内部其实又调了一次 predict）
- 什么时候用哪个？

### Q5 `accuracy_score(y_test, y_predict)` 参数顺序是 `(y_true, y_pred)` 还是 `(y_pred, y_true)`？
- 顺序记反会怎样？（accuracy 对称所以结果一样，但其他指标会错！）
- 类似的指标 precision/recall/f1 顺序错就完蛋
- sklearn 的统一惯例：永远 `y_true` 在前

### Q6 KNN 分类任务默认 `model.score` 算的是什么指标？
- 回归任务的 `model.score` 算啥？（提示：R²，不是 MSE）
- 想换其他指标怎么办？（`from sklearn.metrics import f1_score, classification_report`）

---

## C. 推理阶段（step 6 的细节）

### Q7 新数据 `x_new = [[3, 5, 4, 2]]` 为啥还要 `transfer.transform(x_new)`？
- 为啥不能直接 `model.predict([[3, 5, 4, 2]])`？
- "训练时怎么处理特征，推理时也得怎么处理"——这条契约是怎么保证的？
- 如果忘了 transform 推理结果会怎样？（数值大很多，距离全错）

### Q8 `predict_proba` 在 KNN 这里返回什么？iris 三分类下输出 shape？
- shape = (n_samples, n_classes) = (1, 3)
- 三个数加起来等于多少？
- 这"概率"是怎么算出来的？（k 个邻居的类别占比）
- 当 k=3 时，可能的概率值有哪几种？（提示：0/3, 1/3, 2/3, 3/3）

### Q9 假设 `predict_proba` 输出 `[[0.0, 0.667, 0.333]]`，model.predict 返回什么？为什么？
- predict 默认取概率最大的类
- 如果两类并列最大（如 `[0.5, 0.5, 0]`）怎么办？
- 工程上更倾向用 predict_proba + 自定义阈值的场景？（提示：不平衡分类、风控）

---

## D. 工程坑

### Q10 这份代码的 `transfer` 变量在 step 6 还能用——为啥？
- StandardScaler fit 后会**保存** mean_ 和 var_ 到对象内
- 这是 sklearn "stateful transformer" 设计的核心
- 推理服务怎么把 `transfer` 也持久化？（joblib.dump 一并保存）

### Q11 如果换数据集（不是 iris 了），这份代码改哪几行就能跑？
- step 1 换 `load_xxx()`
- step 4 的 k 可能要重调
- 其他步骤几乎不变——这就是 sklearn API 一致性的威力
- 反思：为什么 sklearn 能做到"一套 API 适配几十种算法"？

---

## 答题状态

- [ ] Q1 6 步流程默写
- [ ] Q2 切分在前 scaler 在后
- [ ] Q3 测试集只 transform
- [ ] Q4 两种打分等价性
- [ ] Q5 accuracy_score 参数顺序
- [ ] Q6 默认指标
- [ ] Q7 推理也要 transform
- [ ] Q8 predict_proba 含义
- [ ] Q9 predict 决策规则
- [ ] Q10 transformer 持久化
- [ ] Q11 换数据集复用代码
