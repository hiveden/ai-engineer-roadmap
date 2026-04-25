# 复习题：MNIST 手写数字识别端到端

> 对应代码：
> - [`step7_mnist_explore.py`](./step7_mnist_explore.py) — 数据探索 + 图像复原
> - [`step7_mnist_train.py`](./step7_mnist_train.py) — 训练 + joblib 持久化
> - [`step7_mnist_predict.py`](./step7_mnist_predict.py) — 推理 + 图像加载坑
>
> 范围：CSV 数据读取 / pandas iloc / 像素归一化捷径 / 28×28 reshape / joblib / png/jpg 差异

---

## A. 数据读取与还原（explore）

### Q1 `df.iloc[:, 1:]` 和 `df.iloc[:, 0]` 分别在做什么？
- `iloc` 是什么的缩写？跟 `loc` 区别？
- `[:, 1:]` 这个切片含义：所有行 + 第 1 列到末尾
- `[:, 0]` 含义：所有行 + 第 0 列
- 为什么是先列再行？跟 numpy `[行, 列]` 一致

### Q2 `Counter(y)` 输出什么？为什么要看？
- 类别分布：`Counter({1: 4684, 7: 4401, ...})`
- 为什么探索阶段必看？（决定要不要 stratify、要不要 oversampling）
- 业务锚：类比 SQL `SELECT label, COUNT(*) FROM ... GROUP BY label`

### Q3 `x.iloc[idx].values.reshape(28, 28)` 这一长串在干嘛？
- `.iloc[idx]` 取第 idx 行 → 返回 pandas Series（长度 784）
- `.values` 转 numpy ndarray（pandas 不直接给 reshape）
- `.reshape(28, 28)` 把 1D 784 → 2D 28×28
- 为什么 28×28 = 784？（MNIST 标准分辨率）

### Q4 `plt.imshow(data, cmap='gray')` 的 `cmap` 是啥？
- 默认 cmap 是 `'viridis'`（黄绿紫渐变），不适合数字
- `'gray'` = 灰度
- 其他常用：`'hot'`（热度图）、`'binary'`（黑白二值）
- 灰度图 0=黑 / 255=白（matplotlib 默认）

---

## B. 像素归一化捷径（train）

### Q5 `x = x / 255.0` 为什么不用 MinMaxScaler？
- 像素天然在 [0, 255]，**已知边界**
- MinMaxScaler 要 fit 算 min/max，多此一举
- 等价于 `MinMaxScaler(feature_range=(0,1))` + 假设 min=0, max=255
- 这种"知道边界、省掉 fit"的捷径还有什么场景？（提示：归一化经纬度、归一化年龄到 [0,1]）

### Q6 `x = x / 255.0` 和 `x = x / 255` 有区别吗？
- 如果 x 是整数 dtype：`x / 255` 在 Python 3 没区别，但在 numpy 早期版本会触发整数除法
- 显式写 `255.0` 强制浮点运算更安全
- 工程习惯：除法的分母**永远写浮点**避免 dtype 陷阱

### Q7 这份代码的 `stratify=y` 还有必要加吗？类别已经基本均衡了
- 类别均衡场景下加不加 stratify 差别小
- 但养成习惯：分类任务**默认加**，零成本保险
- 复盘 `04-split-stratify` Q4 的论点

### Q8 `model.fit(x_train, y_train)` 在 33600 × 784 数据上耗时几秒？
- KNN 是 lazy learner，fit 接近 O(1)
- 实测在 M4 Pro 上 < 100ms
- 那慢在哪？（predict 阶段，brute force O(n_train · n_test · d) ≈ 33600 × 8400 × 784）

---

## C. joblib 持久化（train）

### Q9 `joblib.dump(model, '手写数字识别.pth')` 跟 `pickle.dump` 区别？
- joblib 对 numpy 数组优化得更好（压缩 + 内存映射）
- pickle 是 Python 通用序列化，啥都能存
- sklearn 模型一般推荐 joblib（大数据集 → 文件小很多）
- `.pth` 后缀是 PyTorch 习惯，sklearn 用 `.pkl` 或 `.joblib` 更地道

### Q10 这个 .pth 文件大概多大？为什么这么大？
- 估算：33600 训练样本 × 784 特征 × 8 bytes (float64) ≈ ?
- 实测 ~200MB
- 为什么 KNN 模型 = 训练集？（KNN 没"参数"，要预测必须存所有训练点）
- 对比逻辑回归同任务的模型大小？（10 类 × 784 + 10 偏置 ≈ 60KB，差 3000+ 倍）

### Q11 `joblib.load('手写数字识别.pth')` 加载时会发生什么？
- 反序列化：把磁盘 200MB 数据全部读回内存，重建 KNeighborsClassifier 对象
- 启动时间：磁盘 I/O 主导，SSD 上 1-2 秒
- 推理服务部署时这意味着什么？（每实例至少 200MB 内存基线）
- 多副本部署时怎么省内存？（提示：`joblib.dump(..., mmap_mode='r')` 共享内存）

---

## D. 图像加载坑（predict）

### Q12 `plt.imread` 对 PNG 和 JPG 行为不一致——具体差在哪？
- PNG：自动归一化到 [0, 1]，可能含 alpha 通道（4 通道）
- JPG：保持 [0, 255]，3 通道（RGB）
- 灰度图可能 1 通道
- 这种"看格式吃饭"的 API 在工程上叫什么？（提示：silent failure / surprising behavior）

### Q13 训练时用 `x = x / 255.0`，推理时如果传入 PNG 用 `plt.imread` 会怎样？
- PNG 已经被 imread 归一化到 [0, 1]
- 再除 255 → 范围变成 [0, 1/255] ≈ [0, 0.004]
- 模型在 [0, 1] 上训练，看到 [0, 0.004] 的输入 → 距离全变小，预测错乱
- 工程教训：**训练和推理必须用同一份预处理代码**

### Q14 `x.reshape(1, -1)` 中 `-1` 是什么意思？
- "自动推断这一维的大小"
- 这里 x 是 28×28 = 784 元素，reshape(1, -1) 推断 -1 = 784
- 等价于 `x.reshape(1, 784)` 但更稳健（万一图片不是 28×28 不会硬报错）
- 反过来：`x.reshape(-1, 784)` = 自动推断行数

### Q15 注释里"3 通道彩图转 1 通道" `x = x[:, :, 0]` 在干啥？
- shape (H, W, 3) → 取第 0 个通道（红色）→ shape (H, W)
- 为啥取第 0 个就能用？（彩色数字图本来就近似灰度，三通道值差不多）
- 标准做法是加权平均：`gray = 0.299*R + 0.587*G + 0.114*B`（人眼亮度公式）
- 用 PIL 一行：`Image.open(path).convert('L')`

### Q16 真实场景"用户上传一张拍的数字图"要做哪些预处理？
完整 pipeline：
1. 加载（用 PIL 而不是 plt.imread，行为稳定）
2. 转灰度（`.convert('L')`）
3. resize 到 28×28（`.resize((28, 28))`）
4. 归一化（除 255 或 MinMax）
5. reshape 到 (1, 784)
6. 喂模型 predict

每步漏一个，模型就翻车。

---

## E. 综合 / 反思

### Q17 KNN 在 MNIST 上能达到什么准确率？跟 SOTA 差多远？
- KNN（k=5）：~96-97%
- 简单 CNN：~99%
- ResNet：~99.5%+
- KNN 已经"够用"但远不够好——为什么图像任务后来都用 NN？

### Q18 这个 200MB KNN 模型如果要部署到手机端，能用吗？
- 推理代价 ∝ 训练集大小：33600 样本每次预测都要算距离
- 200MB 内存 + 几百毫秒延迟 → 手机端不可接受
- 替代方案：训练一个轻量 NN（< 1MB） / 用 ANN 库（faiss/annoy）压缩近邻索引

### Q19 这份代码哪些地方用了我们前面 lab 的知识点？
- 像素归一化 ↔ `02-scaling`
- train_test_split + stratify ↔ `04-split-stratify`
- model.fit / predict / score ↔ `00-basicapi` + `05-iris-pipeline`
- KNN lazy 特性 ↔ `01-bruteforce`（200MB 模型是部署成本的实证）

### Q20 哪些没用上但应该考虑加的？
- `06-cv-gridsearch` 的 GridSearchCV 调 k——k=5 是拍脑袋的，应该网格搜索
- Pipeline——这里手动 `x / 255` + `model.fit` 也能写成 Pipeline
- 这是后续优化方向

---

## 答题状态

- [ ] Q1 iloc 切片
- [ ] Q2 Counter 类别分布
- [ ] Q3 reshape(28, 28)
- [ ] Q4 cmap='gray'
- [ ] Q5 x/255 捷径
- [ ] Q6 浮点除法
- [ ] Q7 stratify 必要性
- [ ] Q8 fit 几乎免费
- [ ] Q9 joblib vs pickle
- [ ] Q10 200MB 真相
- [ ] Q11 加载延迟
- [ ] Q12 plt.imread 不一致
- [ ] Q13 双重归一化坑
- [ ] Q14 reshape(-1)
- [ ] Q15 通道处理
- [ ] Q16 完整推理 pipeline
- [ ] Q17 KNN 准确率天花板
- [ ] Q18 部署可行性
- [ ] Q19 知识点串联
- [ ] Q20 还能加什么
