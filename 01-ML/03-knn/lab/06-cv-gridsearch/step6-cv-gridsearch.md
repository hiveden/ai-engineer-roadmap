# 复习题：交叉验证 + 网格搜索 + Pipeline（含 data leakage 对比）

> 对应代码：
> - [`step6_cv_gridsearch_naive.py`](./step6_cv_gridsearch_naive.py) — 反面教材（朴素版，含 leakage）
> - [`step6_cv_gridsearch_pipeline.py`](./step6_cv_gridsearch_pipeline.py) — 正确版（Pipeline）
>
> 范围：交叉验证 / GridSearchCV / Pipeline / data leakage（**今天最重要的金矿**）

---

## A. 交叉验证 cross validation

### Q1 为啥要交叉验证？单次 train/test split 有什么问题？
- 单次切分的最大问题：**单次评估方差大**，运气好坏决定 score
- 同一份数据切 10 次（不同 random_state），accuracy 可能在 0.85~0.95 之间跳——你信哪个？
- CV 通过多次切分取平均，**降低评估方差**

### Q2 K-fold CV 是怎么工作的？画一下 4 折的过程
```
原始训练集（120 样本）划分成 4 份（A, B, C, D）

第 1 折：train = B+C+D, val = A → score_1
第 2 折：train = A+C+D, val = B → score_2
第 3 折：train = A+B+D, val = C → score_3
第 4 折：train = A+B+C, val = D → score_4

最终：mean(score_1, ..., score_4) 作为这个超参组合的评分
```
- 每个样本被用作 val 几次？（恰好 1 次）
- 每个样本被用作 train 几次？（k-1 次）
- 4 折和 5 折哪个更稳？为啥业界默认 5 或 10？

### Q3 K-fold 的 K 怎么选？
- K=2：等于"半训练半测试"，方差大
- K=N（留一法 LOO）：方差最小但开销爆炸
- K=5 / K=10：业界常用甜蜜点
- 数据少（<100）→ K 大一点甚至 LOO；数据多（>10000）→ K=5 够了

---

## B. GridSearchCV API

### Q4 `GridSearchCV(model, param_grid, cv=4)` 在做什么？
- 拆解动作：**遍历 param_grid 所有组合**，每个组合**做 K-fold CV**，挑分数最高的
- iris 这里：6 个 k 值 × 4 折 = **24 次训练**
- `param_grid` 字典格式 `{'param_name': [v1, v2, ...]}`

### Q5 `estimator.fit(x_train, y_train)` 之后能查哪些结果？
- `best_score_`：最优组合的 CV 平均分
- `best_params_`：最优超参字典
- `best_estimator_`：用最优超参在**全部 x_train**重新训练后的模型
- `cv_results_`：详细每个组合 × 每折的所有分数（DataFrame 友好）

### Q6 `GridSearchCV` 自动会做"refit on full training set"吗？
- 默认 `refit=True`：CV 选出最优参数后，**用全部 x_train 重训**一次最终模型
- 为什么要 refit：CV 时每折只用了 (k-1)/k 数据训练，refit 让最终模型见到全部训练数据
- 所以可以直接 `estimator.predict(x_test)` 而不用手动 refit

### Q7 多个超参怎么搜？组合数会爆炸吗？
- `{'n_neighbors': [3,5,7], 'weights': ['uniform','distance'], 'p': [1,2]}`
- 总组合数：3 × 2 × 2 = 12 → CV K=4 → 48 次训练
- 组合爆炸怎么办？（提示：`RandomizedSearchCV`、`HalvingGridSearchCV`、贝叶斯优化）

---

## C. Pipeline + GridSearch（核心金矿）

### Q8 朴素版（`step6_cv_gridsearch_naive.py`）的 data leakage 在哪？画图说明
- 流程：`train_test_split` → `scaler.fit_transform(x_train)` → `GridSearchCV.fit(x_train, ...)`
- 第 2 步出问题：scaler 的 mean/std 是在 **整个 x_train** 上算的（含 120 个样本）
- 第 3 步 GridSearchCV 把 x_train 切成 4 折，每折 30 验证 + 90 训练
- **每折的 30 个"验证样本"在第 2 步已经被 scaler 看过了**
- 这就是泄漏：scaler 的统计量"包含了"验证集的信息

### Q9 正确版（`step6_cv_gridsearch_pipeline.py`）怎么避免 leakage？
- 把 scaler 和 model 串进 Pipeline
- GridSearchCV 内部每折独立调 `pipe.fit(train_fold)`：
  - scaler 只在 `train_fold` 的 90 样本上 fit
  - val_fold 的 30 样本只 transform，从未参与 fit
- 关键：**Pipeline 的 fit/transform 在每折内部独立做**

### Q10 朴素版 best_score_ ≈ 0.971，正确版 ≈ 0.933。差的 4% 是怎么来的？
- 朴素版：scaler 在含 val 样本的"完整训练集"上 fit，val 样本被"标准化得很完美"
- 这种"完美"在真实推理时不存在（新数据从未参与 scaler fit）
- 所以朴素版的高分是**乐观偏差** optimistic bias
- 上线后真实表现接近 0.933 甚至更低

### Q11 Pipeline 的语法细节
```python
pipe = Pipeline([
    ('std', StandardScaler()),
    ('knn2', KNeighborsClassifier()),
])
param_dict = {'knn2__n_neighbors': [1, 3, 5, 7]}
```
- `'knn2'` 是什么？（自定义 step 名）
- `'knn2__n_neighbors'` 双下划线是什么意思？
- 单下划线 `'knn2_n_neighbors'` 会怎样？（识别不了）
- 嵌套更深：如果 step 里嵌套 Pipeline，参数名是 `'outer__inner__param'` 三段

### Q12 Pipeline 还能用在哪些场景？
- 多个 transformer 串联：`OneHotEncoder` + `StandardScaler` + `PCA` + `KNN`
- 持久化：`joblib.dump(pipe, 'pipeline.pkl')` 一个文件搞定全流程
- 推理：`pipe.predict(new_raw_data)` 自动走完所有变换，无需手动管 scaler 状态
- 上线工程：训练 / 评估 / 推理用同一个 pipeline 对象，**契约由代码保证**

---

## D. 工程坑 / 进阶

### Q13 朴素版还有一个隐藏的"二次 fit_transform" 在哪？（提示：找最后那段被注释的代码块）
- 原始代码（已删除复杂部分，但概念保留）有 "5.2 重新训练" 段：
  - GridSearchCV 选出 best_params 后，又**手动**新建 `KNeighborsClassifier(n_neighbors=7)` 然后 `fit(x_train_, y_train)`
  - 但 `x_train_` 又被重新 `fit_transform` 一次
- 这是**双重 leakage** 嫌疑——本节代码已经简化删除，但理解上要提防

### Q14 `cv_results_` 怎么用？
- 是 dict，可以直接 `pd.DataFrame(estimator.cv_results_)`
- 关键列：`mean_test_score`, `std_test_score`, `params`, `rank_test_score`
- 实战：画 score vs k 的曲线，肉眼看拐点

### Q15 GridSearchCV 慢，怎么提速？
- `n_jobs=-1`：并行（每个组合 × 每折独立训练，并发友好）
- `RandomizedSearchCV`：随机采样，不遍历所有组合
- `HalvingGridSearchCV`：先小数据集筛掉差组合，再大数据集精调
- 减小 cv 折数（从 10 → 5）

### Q16 测试集的角色？
- 测试集**全程不参与 GridSearchCV**——CV 只在 x_train 上做
- 选出 best_params + refit 后，测试集只用一次：评估"最终模型"的真实表现
- 如果反复用测试集调参 → 又是 data leakage（间接）

---

## 答题状态

- [ ] Q1 单次切分的方差问题
- [ ] Q2 K-fold 流程
- [ ] Q3 K 怎么选
- [ ] Q4 GridSearchCV 干啥
- [ ] Q5 best_* 属性
- [ ] Q6 自动 refit
- [ ] Q7 多超参组合爆炸
- [ ] Q8 朴素版 leakage 在哪
- [ ] Q9 Pipeline 怎么救
- [ ] Q10 4% 差距的来源
- [ ] Q11 Pipeline 语法
- [ ] Q12 Pipeline 应用场景
- [ ] Q13 双重 leakage
- [ ] Q14 cv_results_
- [ ] Q15 GridSearchCV 提速
- [ ] Q16 测试集隔离
