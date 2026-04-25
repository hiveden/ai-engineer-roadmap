# 复习题：特征缩放（归一化 + 标准化）

> 对应代码：[`step2_feature_scaling.py`](./step2_feature_scaling.py)
> 范围：**只围绕这份代码能直接观察到的现象**。具体落地（pipeline 集成 / 异常值处理 / 其他 scaler）留给后续主题。
> 用法：盖住 hint，先答主问；卡住再看 hint；最后翻 [`step2-feature-scaling-answers.md`](./step2-feature-scaling-answers.md) 对答案。

---

## A. 为什么要做特征缩放

### Q1 看 dm01 的输入数据：4 列分别像是 [年龄, 工龄, 学历, 评分] 这种"量纲完全不同"的特征。如果直接拿去喂 KNN，会发生什么？
- 第 1 列范围 60~90，第 2 列范围 2~4，算欧氏距离时哪列说了算？
- 这种现象叫什么？（提示：dominant feature / 量纲霸占）
- 为什么逻辑回归/树模型受影响小，KNN/SVM/神经网络受影响大？

### Q2 哪些算法依赖特征缩放，哪些不依赖？
- 依赖派：KNN、SVM（带核）、神经网络、K-means、PCA
- 不依赖派：决策树、随机森林、XGBoost
- 它们的差别是"距离/梯度敏感" vs "分裂点敏感"——展开说

---

## B. MinMaxScaler 的计算细节

### Q3 dm01 的输出第 1 列应该是什么？手算给我看。
- 第 1 列原值：`[90, 60, 75]`
- min = ?, max = ?
- 套公式 `(x - min) / (max - min)`，三个值分别变成多少？
- 验证：fit_transform 后第 1 列是 `[1.0, 0.0, 0.5]` 吗？

### Q4 MinMaxScaler 是按行还是按列归一化？为什么这是合理的？
- 提示：每列代表一个特征（feature），每行代表一个样本（sample）
- 想象一下"按行归一化"会发生什么——把不同特征的量纲混在一起算 min/max，没意义
- sklearn 所有 scaler 默认都是 column-wise

### Q5 MinMax 默认范围是 `[0, 1]`，怎么改成 `[-1, 1]`？
- 构造参数：`MinMaxScaler(feature_range=(-1, 1))`
- 什么场景需要 `[-1, 1]`？（提示：神经网络激活函数中心化输入）

---

## C. StandardScaler 的计算细节

### Q6 dm03 的 `transformer.mean_` 和 `transformer.var_` 输出会是什么？挑第 1 列手算。
- 第 1 列原值：`[90, 60, 75]`
- mean = ?
- var = ?（注意：sklearn 的 var_ 是**总体方差** population variance，分母 n，不是样本方差 n-1）
- std = √var = ?
- 套公式 `(x - mean) / std`，三个值分别是？

### Q7 `transformer.var_` 命名有坑——为什么不叫 `std_`？工程上要小心什么？
- 拿到 `var_` 想求标准差怎么算？（`np.sqrt(transformer.var_)`）
- 类似命名坑还有：`transformer.scale_` 是啥？（其实就是 `sqrt(var_)`，sklearn 真正用来 transform 的值）
- 自己写代码时记成"std" 然后调 `var_` → 结果差一个开方，bug 难查

### Q8 dm03 输出的标准化结果第 1 列加起来应该等于多少？为什么？
- 标准化后每列均值 = 0 → 三个值之和应该 ≈ 0（浮点误差除外）
- 标准差 = 1 → 三个值的平方和 ≈ 列数（这里是 3）
- 这是检验 fit_transform 有没有跑对的快速 sanity check

---

## D. fit / transform / fit_transform 的区别（最易踩坑）

### Q9 这三个方法分别在做什么？为什么 sklearn 要拆成两步？
- `fit(X)` 做啥？（算统计量：min/max 或 mean/var，存到对象属性里）
- `transform(X)` 做啥？（套公式，用之前算好的统计量）
- `fit_transform(X)` = ?
- 拆开的工程理由：训练集 fit 一次，测试集 / 在线请求复用同一组参数

### Q10 假设你已经在训练集上 `fit_transform` 了。来了一份测试集 X_test，下面 3 种写法哪个对？为什么错的那两种叫 data leakage？
```python
# 写法 A
X_test_scaled = transformer.fit_transform(X_test)

# 写法 B
new_t = MinMaxScaler()
X_test_scaled = new_t.fit_transform(X_test)

# 写法 C
X_test_scaled = transformer.transform(X_test)
```
- 哪个对？为什么？
- 写法 A 错在哪？（用测试集的 min/max 重新拟合，等于提前看了答案）
- 写法 B 错在哪？（训练集和测试集用不同的 min/max，量纲对不上，模型预测会胡来）
- 这种坑在生产怎么避免？（提示：sklearn Pipeline）

---

## E. MinMax vs Standardize 怎么选

### Q11 同一份数据用 MinMax 和 Standardize 跑出来的数值范围差很多，分别什么场景该选哪个？
- MinMax 适合：?（提示：神经网络输入、图像像素 [0,255]→[0,1]、需要明确范围的场景）
- Standardize 适合：?（提示：数据近似正态分布、有异常值、KNN/SVM 默认首选）
- 异常值场景下两者表现差异？（一个极大值会把 MinMax 的正常值挤到接近 0；Standardize 的均值/标准差也会被拉偏，但相对没那么戏剧）
- 如果不确定选哪个？（实操：两个都试，CV 看效果）

---

## 答题状态追踪

- [ ] Q1 量纲霸占
- [ ] Q2 哪些算法依赖 scaling
- [ ] Q3 MinMax 手算
- [ ] Q4 column-wise 归一化
- [ ] Q5 feature_range 参数
- [ ] Q6 Standardize 手算
- [ ] Q7 var_ vs std_ 命名坑
- [ ] Q8 sanity check
- [ ] Q9 fit / transform / fit_transform 区别
- [ ] Q10 测试集别 fit_transform（data leakage）
- [ ] Q11 MinMax vs Standardize 选型
