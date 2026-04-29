# 答案：MNIST 手写数字识别端到端

> 配套题目：[`step7-mnist.md`](./step7-mnist.md)
> 对应代码：[`step7_mnist_explore.py`](./step7_mnist_explore.py) / [`step7_mnist_train.py`](./step7_mnist_train.py) / [`step7_mnist_predict.py`](./step7_mnist_predict.py)

---

## A. 数据读取与还原

### Q1 iloc 切片

**`iloc`** = **i**nteger **loc**ation，按**位置**索引（0-based）。

| 工具 | 索引方式 |
|---|---|
| `df.loc[row_label, col_label]` | 按**标签**（行索引值 / 列名） |
| `df.iloc[row_pos, col_pos]` | 按**位置**整数（0, 1, 2...） |

**这份代码用法**：

```python
df.iloc[:, 1:]   # 所有行，第 1 列到末尾（784 像素特征）
df.iloc[:, 0]    # 所有行，第 0 列（label）
df.iloc[10000]   # 第 10000 行，所有列
df.iloc[10000, 1:]  # 第 10000 行，第 1 列到末尾
```

**先行后列**：跟 numpy `arr[row, col]` 完全一致。

**对比 SQL**：
- `df.iloc[:, 1:]` ≈ `SELECT * EXCEPT(col_0) FROM df`
- `df.iloc[:, 0]` ≈ `SELECT col_0 FROM df`

**冒号 `:` 含义**：
- `:` = 全选
- `1:` = 从 1 到末尾
- `:5` = 从开头到 5（不含 5）
- `1:5` = 从 1 到 5（不含 5）
- `:, 1:` 第一个 `:` 是行（全选），第二个 `1:` 是列（从 1 开始）

---

### Q2 Counter 类别分布

```python
from collections import Counter
print(Counter(y))
# Counter({1: 4684, 7: 4401, 3: 4351, 9: 4188, 2: 4177,
#          6: 4137, 0: 4132, 4: 4072, 8: 4063, 5: 3795})
```

**为什么探索阶段必看**：

| 决策 | 看类别分布的价值 |
|---|---|
| 要不要 stratify | 不平衡时 stratify 是救命的（参考 `04` Q5） |
| 要不要 oversample / undersample | 极度不平衡（如 1:99）光 stratify 不够，要 SMOTE 等 |
| 哪种评估指标 | 平衡用 accuracy / 不平衡用 F1 / AUC |
| 是否合适用 KNN | 多类不平衡时 KNN 投票偏向多数类 |

**业务锚 SQL 等价**：

```sql
SELECT label, COUNT(*) AS cnt
FROM mnist
GROUP BY label
ORDER BY cnt DESC;
-- 1     4684
-- 7     4401
-- ...
```

**MNIST 这里**：基本平衡（最多 4684 / 最少 3795 = 1.23 比），加不加 stratify 差别小。

---

### Q3 reshape(28, 28) 三连

```python
data = x.iloc[idx].values.reshape(28, 28)
```

**逐步拆解**：

```python
step1 = x.iloc[idx]
# 类型: pandas.Series, shape (784,), 长度 784
# 内容: [0, 0, 0, 23, 154, ...] - 一行的 784 个像素值

step2 = step1.values
# 类型: numpy.ndarray, shape (784,), 把 pandas Series 转成 numpy 数组
# pandas Series 没有 .reshape 方法，必须先转 numpy

step3 = step2.reshape(28, 28)
# 类型: numpy.ndarray, shape (28, 28), 把 1D 784 重排成 2D 28×28
# 元素总数必须相等（784 = 28 × 28），否则报错
```

**为什么 28×28 = 784**：MNIST 的标准分辨率。NIST 数据集原始更大，1998 年 LeCun 团队 resize 到 28×28（足够保留数字形状，又能放进当年的小内存）。

**reshape 的内存视角**：不复制数据，只换"读取顺序"。底层数据还是连续的 784 浮点数，view 变成 28×28。

---

### Q4 cmap

`cmap` = **c**olor**map** 颜色映射，把数值映射到颜色。

```python
plt.imshow(data)                # 默认 'viridis'（黄绿紫渐变）
plt.imshow(data, cmap='gray')   # 灰度
plt.imshow(data, cmap='hot')    # 黑红黄白渐变（热图）
plt.imshow(data, cmap='binary') # 黑白二值（小=白 / 大=黑，跟 gray 反过来）
```

**默认 viridis 不适合数字图**：会显示成黄绿色块，不直观。

**灰度模式**：
- matplotlib 默认：0=黑，最大值=白
- 一些库反过来（OpenCV 默认 BGR 也奇怪）
- 加 `vmin=0, vmax=255` 强制范围避免自动拉伸

**MNIST 数字**：背景黑（像素 0），笔画白（像素 ~255），所以 `cmap='gray'` 看起来就是黑底白字。

---

## B. 像素归一化捷径

### Q5 x / 255 vs MinMaxScaler

**捷径成立的前提**：你**事先知道**特征的 min/max 边界。

| 数据 | 已知边界？ | 推荐做法 |
|---|---|---|
| 像素值 | min=0, max=255（不会变） | `x / 255` 捷径 |
| 经纬度 | lat ∈ [-90, 90], lon ∈ [-180, 180] | 手动除一个常数 |
| 年龄 | 大致 [0, 120] | 可手动 / 也可 fit |
| 用户余额、销售额 | 不知道 | 必须 fit |

**捷径的优点**：
- 没有 fit 步骤，**没有数据泄漏可能**（无状态）
- 训练 / 推理代码完全一致
- 部署时不用持久化 scaler 对象

**捷径的代价**：
- 万一来了个超出 [0, 255] 的脏数据，会输出超出 [0, 1] 的值（但 MinMaxScaler 也有这个问题）
- 假设错了就翻车（比如以为像素 [0, 255] 实际是 [0, 65535] 16-bit 图像）

---

### Q6 写 255.0 还是 255

**Python 3 中**：`x / 255` 和 `x / 255.0` 行为一样（`/` 永远是真除法返回 float）。

**numpy / pandas 中**：取决于 x 的 dtype。

```python
import numpy as np
x_int = np.array([100, 200], dtype=np.uint8)
print(x_int / 255)     # [0.392, 0.784] dtype float64 ✅
print(x_int / 255.0)   # [0.392, 0.784] dtype float64 ✅
print(x_int // 255)    # [0, 0] dtype int ⚠️ 整数除法
```

**Python 2 时代** `x / 255` 在整数 dtype 上会触发整数除法（变成 0）——这是历史遗留坑。

**工程习惯**：除法分母**永远写 `.0`**，让意图明确（"我要的是浮点结果"）。读代码的人也一眼明白。

---

### Q7 stratify 必要性

**MNIST 类别 1.23:1 比**——基本平衡。不加 stratify 影响不大。

**为啥还是建议加**：
1. **零成本**：参数加上去几乎不影响速度
2. **保险**：万一切分时运气坏（30000 样本里某类被抽稀），评估失真
3. **习惯**：分类任务默认开，避免哪天换数据集忘了加翻车

**何时可以不加**：
- 类别极度均衡（如 50/50/50 完全相等）
- 只是探索性跑通流程，不在乎评估精度
- 时间序列任务（不能 stratify）

参考 `04-split-stratify` Q4。

---

### Q8 fit 几乎免费

**实测**（M4 Pro）：

```python
import time
t = time.time(); model.fit(x_train, y_train); print(f'fit: {(time.time()-t)*1000:.1f}ms')
# fit: ~50-100ms（绝大部分是 numpy 数据校验和 deepcopy 的开销）
```

**predict 才慢**：

```python
t = time.time(); model.predict(x_test); print(f'predict: {(time.time()-t):.1f}s')
# predict: 5-20s （取决于 algorithm 是 brute / kd_tree / ball_tree）
```

**为啥 predict 慢**：
- 8400 测试样本，每个要算到 33600 训练样本的距离
- 每个距离 = 784 维欧氏距离 = 784 次平方 + 求和 + 开方
- 总操作量：8400 × 33600 × 784 ≈ **2.2 亿次**距离计算

**算法层加速**：
- `algorithm='kd_tree'` 在低维（< 20 维）有效，**MNIST 784 维退化回 brute**
- `algorithm='ball_tree'` 在 MNIST 上略好但有限
- 加 `n_jobs=-1` 并行 → 12 核 → 大约 12 倍加速
- 真要快：用 `faiss` 库（Facebook 开源 ANN）

复盘 `01-bruteforce`。

---

## C. joblib 持久化

### Q9 joblib vs pickle

| | pickle | joblib |
|---|---|---|
| 适用 | 任意 Python 对象 | 大型 numpy 数组优化 |
| 大小 | 标准 | 自动压缩 |
| 速度 | 通用 | 大数组更快 |
| 内存 | 全加载 | 支持 mmap_mode（共享内存） |
| sklearn 推荐 | 可以用 | **首选** |

**底层**：joblib 内部就是 pickle，但对 numpy 数组单独优化（直接二进制存，而不是 pickle 协议序列化每个元素）。

**MNIST 模型实测**：
- pickle 序列化：~210MB
- joblib（默认）：~210MB（差不多）
- joblib（compress=3）：~80MB（zlib 压缩）

**`.pth` 后缀**：
- PyTorch 习惯 `.pth` 或 `.pt`
- sklearn 社区习惯 `.pkl` / `.joblib`
- 课程代码混了——无害但不规范

---

### Q10 200MB 模型的真相

**精确估算**：

```
训练样本数: 33600
特征数: 784 (28 × 28)
dtype: float64 (8 bytes per number)

X_train: 33600 × 784 × 8 = 210,739,200 bytes ≈ 211 MB

KNN 模型 = X_train + y_train + 一些元数据（< 1MB）
≈ 211 MB ✅
```

**为啥 KNN 模型 = 训练集**：

KNN 没有"参数"（weights / bias / 树结构都没有）。要预测 → 必须算"查询点到训练点的距离" → 必须**保留所有训练点**。

**对比逻辑回归同任务**：

```
LR 多分类（10 类）参数：
  weights: 10 × 784 = 7840 个 float
  bias: 10 个 float
  共 7850 × 8 = 62,800 bytes ≈ 60 KB

模型大小比: 211 MB / 60 KB = 3500 倍
```

LR 训完丢掉训练集，只留参数。KNN 训完**等于训练集本身被打包了**。

**这是工程学的"lazy 代价转移"**——延迟决策的代价不是消失，是从训练阶段转移到了推理阶段，并以"模型膨胀"的形式承担。

---

### Q11 加载延迟和内存基线

```python
import time
t = time.time()
model = joblib.load('手写数字识别.pth')
print(f'load: {time.time()-t:.2f}s')
# load: ~1.5s （SSD）/  ~5s （HDD）
```

**主要开销**：
- 磁盘 I/O：210MB 从 SSD 读到 RAM
- 反序列化：joblib 重建 numpy 数组对象（很快）

**部署影响**：

| 部署场景 | 影响 |
|---|---|
| 单实例服务 | 启动 ~1.5s，内存占用 ~210MB |
| Docker 镜像 | 镜像里要包含 .pth → 镜像额外大 200MB |
| Lambda / Cloud Function | 冷启动慢，内存配额可能不够 |
| 多副本（k8s 10 pod） | 总内存 2.1GB（每 pod 自己一份） |

**多副本省内存**：

```python
# 写入时
joblib.dump(model, 'model.pkl')

# 读取时用 mmap_mode
model = joblib.load('model.pkl', mmap_mode='r')
# 多个进程共享同一份磁盘文件的内存映射
# 物理内存只占一份
```

但这只对**同一台机器**多进程有效。跨机器还是要每个节点一份。

**轻量替代**：
- 训练个小 NN（< 1MB） → MNIST 上准确率反而更高
- 用 `faiss` 把 KNN 索引压缩成几十 MB（精度略降）

---

## D. 图像加载坑

### Q12 plt.imread PNG vs JPG

| 格式 | shape | 数值范围 | 自动归一化？ |
|---|---|---|---|
| PNG (RGBA) | (H, W, 4) | [0.0, 1.0] | ✅ 是（除以 255） |
| PNG (RGB) | (H, W, 3) | [0.0, 1.0] | ✅ 是 |
| PNG (灰度) | (H, W) | [0.0, 1.0] | ✅ 是 |
| JPG | (H, W, 3) | [0, 255] uint8 | ❌ 否 |
| BMP | (H, W, 3) | [0, 255] uint8 | ❌ 否 |
| TIFF | 看情况 | 看情况 | 不一定 |

**这种行为叫**：silent inconsistency / format-dependent behavior。**API 不报错但行为变**——最难 debug 的那种 bug。

**反映的工程哲学问题**：matplotlib 的 imread 设计于 PNG 流行的时代，沿用了 PNG 的"alpha 通道隐含归一化"假设。但 JPG 没有 alpha 通道，所以保留原始 uint8。**两种格式分别遵循自己生态的惯例，混在一个函数里就一致性破产**。

**解药**：用更稳定的库：

| 库 | 一致性 | 推荐 |
|---|---|---|
| `matplotlib.pyplot.imread` | ❌ 跟格式走 | 不推荐（除非你只读 PNG） |
| `PIL.Image.open` | ✅ 永远 uint8 [0, 255] | 标准选择 |
| `cv2.imread` | ⚠️ BGR 顺序（不是 RGB） | OpenCV 项目用 |
| `imageio.imread` | ✅ 一致 uint8 | 替代品 |

---

### Q13 双重归一化坑

**错误流程**：

```python
# 训练时
x = x / 255.0   # uint8 [0, 255] → float [0, 1]
model.fit(x, y)

# 推理时（用 plt.imread 读 PNG）
x = plt.imread('demo.png')   # 已经是 [0, 1] 了！
x = x / 255.0                # ⚠️ 又除一次！[0, 1/255]
model.predict(x.reshape(1, -1))   # 输入数值小了 255 倍
```

**后果**：
- 训练时的"距离 1.0"在推理时变成"距离 1/255 ≈ 0.004"
- 所有距离都被压缩 255 倍 → 排序不变（理论上）→ 但精度损失（浮点误差被放大）
- 实际可能预测错乱（特别是 weights='distance' 时）

**工程教训**：**训练和推理必须用同一份预处理代码**。

**最佳实践**：
1. 用 Pipeline 把预处理打包（推理时自动一致）
2. 不用就把预处理写成函数，训练和推理都调它
3. 测试集和真实推理样本至少抽样验证一遍范围

---

### Q14 reshape(-1)

**`-1` 意思**：让 numpy 自动推断这一维的大小（基于"总元素数 / 其他维度"）。

```python
x.shape  # (28, 28)，总元素数 784

x.reshape(1, -1)    # (1, ?)，? = 784/1 = 784 → (1, 784)
x.reshape(-1, 1)    # (?, 1)，? = 784/1 = 784 → (784, 1)
x.reshape(-1, 28)   # (?, 28)，? = 784/28 = 28 → (28, 28)（等于不变）
x.reshape(2, -1)    # (2, ?)，? = 784/2 = 392 → (2, 392)
x.reshape(-1, -1)   # ❌ 报错，不能两个 -1
```

**为啥用 -1 比写死维度好**：

```python
# 假设输入图像 size 不固定（可能 28×28 也可能 32×32）
x.reshape(1, 784)  # ❌ 32×32 的图会报错
x.reshape(1, -1)   # ✅ 不管多大都自动适配
```

**工程小习惯**：能用 -1 就用 -1，让代码对输入 size 更鲁棒。

---

### Q15 通道处理

```python
x = x[:, :, 0]   # (H, W, 3) → (H, W)
```

**含义**：取所有行、所有列、**第 0 个通道**。结果是单通道 2D 数组。

**为啥取第 0 个就能用**：
- 黑白数字图本来就近似灰度
- RGB 三通道值差不多（不像彩色照片每通道差异大）
- 简单粗暴，但**信息有损**

**标准做法**：加权平均（亮度感知）

```python
# 灰度公式（ITU-R BT.601）
gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
```

**为什么这个权重**：人眼对绿色最敏感（0.587），蓝色最不敏感（0.114）。这是电视行业 1953 年定的标准。

**最优雅写法**：

```python
from PIL import Image
img = Image.open(path).convert('L')  # 'L' = Luminance 灰度
x = np.array(img)
# convert('L') 内部就用了 BT.601 公式
```

`PIL.Image` 的 `.convert('L')` 是工业级灰度转换，比手写 `x[:, :, 0]` 准确得多。

---

### Q16 真实推理 pipeline

```python
import numpy as np
from PIL import Image
import joblib

def predict_digit(image_path, model_path):
    # 1. 加载（PIL，行为稳定）
    img = Image.open(image_path)

    # 2. 转灰度（标准亮度公式）
    img = img.convert('L')

    # 3. resize 到 28×28
    img = img.resize((28, 28))

    # 4. 转 ndarray
    x = np.array(img, dtype=np.float32)
    # shape (28, 28), dtype float32, range [0, 255]

    # 5. 归一化（和训练一致）
    x = x / 255.0

    # 6. reshape (1, 784)
    x = x.reshape(1, -1)

    # 7. 推理
    model = joblib.load(model_path)
    return model.predict(x)[0]
```

**工程坑清单**：

| 步骤 | 漏掉会怎样 |
|---|---|
| 加载库选错（用 plt.imread 读 JPG/PNG 混合） | 数值范围不一致 |
| 没转灰度（保留 RGB） | shape 对不上 (1, 2352) ≠ (1, 784) |
| 没 resize | shape 对不上 |
| 没归一化（训练用了但推理没用） | 数值大 255 倍，预测错乱 |
| reshape 维度错 | 报错或预测错 |
| dtype 是 uint8（没转 float） | 距离算成整数除法 |

**生产建议**：把这 7 步包成函数 + 单测，**绝不在推理服务里裸写**。

---

## E. 综合反思

### Q17 KNN MNIST 准确率 vs SOTA

| 算法 | MNIST 准确率 | 模型大小 | 推理时间 |
|---|---|---|---|
| KNN (k=5) | ~96.5% | 200MB | 慢 |
| SVM (RBF) | ~98.5% | 几十 MB | 中 |
| 逻辑回归 | ~92% | 60KB | 极快 |
| 简单 MLP | ~98% | 几 MB | 快 |
| 简单 CNN（LeNet） | ~99% | 几 MB | 快 |
| ResNet | ~99.5% | 几十 MB | 中 |
| 当前 SOTA | ~99.84% | 较大 | - |

**KNN 96% 是"够用"水平**：
- 比逻辑回归强（说明非线性决策边界有用）
- 比 NN 弱 3 个百分点（看起来不多，但 96% → 99% 错误率从 4% 降到 1%，**降了 4 倍**）

**为什么图像任务都换 NN**：
- KNN 把每个像素当独立特征 → 没有"局部性"概念（28×28 里相邻像素跟远处像素被同等对待）
- CNN 通过卷积核**显式建模空间局部性**（"这片区域是不是横线"）
- 这个建模能力 = NN 在图像任务上的根本优势

---

### Q18 部署可行性

**手机端 KNN 几乎不可行**：

| 维度 | 现状 | 手机能接受？ |
|---|---|---|
| 模型大小 | 200MB | ❌ 用户不会下载 200MB 的 app |
| 内存占用 | 200MB+ | ❌ 多数手机 4-8GB，单 app 占 200MB 太多 |
| 推理延迟 | 几百 ms-几秒 | ❌ 实时识别用户感知极差 |
| CPU 占用 | 高（暴力算距离） | ❌ 发热、耗电 |

**替代方案**：

| 方案 | 模型大小 | 准确率 | 推荐度 |
|---|---|---|---|
| 训练个小 CNN | < 1MB | 99% | 🌟🌟🌟 端侧首选 |
| KNN + 降维（PCA → 50 维） | ~50MB | ~95% | 🌟 |
| KNN + ANN 库（faiss） | ~30MB | ~95% | 🌟🌟 |
| 服务端推理（API） | 不限 | 不限 | 🌟🌟🌟 不限模型 |

**这是 lazy learner 的根本工程局限**：模型大小 ∝ 训练集，没法压缩。

---

### Q19 知识点串联

| 用到的 | lab 主题 | 怎么用 |
|---|---|---|
| `x / 255` 归一化 | `02-scaling` | 像素的"已知边界"捷径 |
| train_test_split + stratify | `04-split-stratify` | 8400 测试样本 |
| `model.fit / predict / score` | `00-basicapi` + `05-iris-pipeline` | 标准三件套 |
| KNN 预测慢的实证 | `01-bruteforce` | 33600 训练 × 8400 测试，predict 几秒 |
| 200MB 模型 = 训练集 | `00-basicapi` Q10 | KNN 部署成本的活实证 |
| `predict_proba` | `05-iris-pipeline` Q8 | 这里没用，可加 |

---

### Q20 还能加什么

**最值得补的**：

1. **GridSearchCV 调 k**（`06-cv-gridsearch`）
   - k=5 是拍脑袋值，应该 GridSearchCV 跑 [3,5,7,9,11] 看 CV 曲线
   - 大概率能再涨 0.5-1%

2. **包成 Pipeline**（`06-cv-gridsearch`）
   - `Pipeline([('scaler', FunctionTransformer(lambda x: x/255)), ('knn', KNeighborsClassifier())])`
   - 训练 / 推理用同一个对象，避免 Q13 的双重归一化坑

3. **分类报告**
   - `from sklearn.metrics import classification_report`
   - 看每一类的 precision / recall / f1，找哪类最难分（大概率是 4 vs 9 或 3 vs 8）

4. **混淆矩阵**
   - `from sklearn.metrics import confusion_matrix`
   - 可视化哪两类容易混

5. **n_jobs=-1 加速**
   - 这份代码没加，predict 阶段并行能省一半时间

```python
model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
```

---

## 一句话总结

> **MNIST 是 KNN 的"压力测试"**：33600 训练样本 → 200MB 模型 → predict 几秒。
> 准确率 ~96.5% 不算差，但跟 CNN 99% 的差距说明**对图像这种有空间结构的数据，KNN 的"扁平化距离"是有上限的**。
>
> 工程坑三连：**训练 / 推理预处理必须一致**（Q13）；**plt.imread 行为不一致**（Q12）；**模型 = 训练集**（Q10/Q18）。
> Pipeline + GridSearchCV + n_jobs=-1 是最简单的"提分三板斧"——但根本提升要换算法。
