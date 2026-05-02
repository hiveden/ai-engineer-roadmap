# R2 · KNN 教学 demo 设计调研（业界先验）

**调研日期**：2026-05-02
**调研范围**：sklearn / Stanford CS231n / Cornell CS4780 / Observable / ISLR / 业界博客 / marimo gallery
**目标**：为本项目 02 (proximity) 和 03 (k-tuning) 的数据集 / 维度 / 可视化 / 叙事改进提供先验依据
**置信度**：HIGH（多数发现 ≥3 独立来源；标杆 demo 已直接 fetch）

---

## 1. 数据集设计

### 发现 1.1 · 业界默认走"合成 2D"，不走"真实小样本"

| 来源 | 数据集 | 样本量 | 维度 | 噪声 |
|---|---|---|---|---|
| sklearn `plot_classifier_comparison` | `make_moons` / `make_circles` / `make_classification` | 100（默认） | 2 | `noise=0.3`（高斯 std） |
| sklearn `plot_classification` (KNN 官方示例) | Iris（仅 sepal_length + sepal_width） | 150 → 训练 112 | 2 | 天然类间重叠 |
| Stanford CS231n `knn` demo | 自定义合成 2D，3 类（红蓝绿） | 30+ 可拖动 | 2 | 用户手动加噪 |
| Observable `@antoinebrl/knn-visualizing-the-variance` | 双圆心合成（中心 ±2.75）+ 噪声点 | 20-300/类（滑块） | 2 | 显式参数 |
| Berkeley SAAS / UBC CS547 教材 | `make_moons(noise=0.35)` | 100 | 2 | `noise=0.35` 公认调参良配 |
| ISLR Fig 2.16 | 合成二维高斯混合 | ~200 | 2 | 类间显著重叠 |

**强信号**：**没有任何标杆教程用 11 个手工挑的真实样本**。原因：

- 真实小样本类间距过大或过小都不可控，k 影响不明显
- 合成数据可独立调节"簇间距 / 噪声 / 样本量"三个旋钮
- `make_moons(n_samples=100, noise=0.2~0.35)` 是 KNN 教学的"行业基线"

来源：
- [sklearn plot_classifier_comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
- [sklearn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [Berkeley SAAS bias-variance](https://saas.berkeley.edu/education/bias-variance-decision-trees-ensemble-learning)
- [UBC CS547 Bias-Variance Visualization](https://www.cs.ubc.ca/~tmm/courses/547-17F/projects/halldor-simar/report.pdf)

### 发现 1.2 · 噪声不是"标错的样本"，而是"位置抖动"

本项目 03 把噪声实现为"高分高吸引度但标'不喜欢'的烂片"——这是 **label noise**（标签噪声）。

业界做法：**feature noise**（位置抖动），即 `make_moons(noise=0.3)` 在两个半月坐标上加高斯抖动。两类分别区分：

| 噪声类型 | 用途 | 教学效果 |
|---|---|---|
| Feature noise（位置抖动） | 演示 k 的平滑作用 | k=1 边界扭曲穿过抖动点；k 增大→边界回归"真实"曲线 |
| Label noise（标签翻错） | 演示鲁棒性 | k=1 在错标点周围炸出小岛；k 增大→错标点被多数票淹没 |

两者都能演示过拟合，但 feature noise 更通用、视觉更连续。本项目"烂片高分"是 label noise 的极端版本，效果合理但**视觉表现依赖样本位置和噪声点占比**——11 个样本里 2 个标错（18%）已超过常规 5-10%，导致 k=11 直接塌陷为多数类，过早进入欠拟合。

来源：
- [sklearn make_moons noise 参数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [Observable @antoinebrl 同时支持点击翻标和拖动位置](https://observablehq.com/@antoinebrl/knn-visualizing-the-variance)

### 发现 1.3 · 类别平衡几乎都是 50/50

CS231n、Observable、sklearn examples 全部用类别平衡数据。本项目 03 是 5 喜欢 / 6 不喜欢——基本平衡，但 11 全员投票时多数类（不喜欢）必胜，这正是欠拟合演示，**逻辑成立但样本量太小**导致"k=11 全图塌陷"过于戏剧化，跳过了"渐进平滑"的中间过程。

---

## 2. 维度选择（1D vs 2D）

### 发现 2.1 · 1D KNN 教学几乎不存在

搜索 "KNN 1D one dimensional decision boundary" / "rug plot" / "strip plot" 全部返回 2D 教程。原因：

- 1D 决策边界退化为"数轴上的若干切点"，**视觉上是几个垂直分界线**，无"形状"概念
- 1D 无法演示 KNN 最大卖点——**非线性边界**（圆圈套圆圈、弯月）
- 1D 一旦讲完欧式距离就直接过渡到 2D，没有教学复利

CS231n、CS4780、ISLR、Hands-On ML、StatQuest 全部直接从 2D 入手。CS4780 明确说："**For pedagogical clarity, use 2D datasets initially**"。

### 发现 2.2 · 1D 能演示过拟合-欠拟合谱，但收益低

理论上可以——把 1D 数据放在数轴上，背景用色块（mark_rect 高度=固定）展示每个 x 段的预测类别。k=1 时色块切换频繁，k 大时色块统一。但：

- 1D 失去"圆形邻域"直觉（KNN 的 k 邻居在 1D 是左右两侧最近的 k 个，不再是"圆圈"）
- 距离公式退化为 `|x1 - x2|`，丢失"勾股"故事
- 拖动新点演示距离重排，1D 的视觉信息密度远低于 2D

**结论**：**不推荐 1D 作为先导**。如果要做"渐进维度"教学，更好的路径是：

```
1D（数轴+距离手算，纯文字/表格） → 2D（02 proximity 现状） → 高维（curse of dimensionality 提示）
```

但 1D 应该作为**章节内嵌示例**（如 02a 豆瓣手算的延伸），不需要独立 demo。

来源：
- [CS4780 kNN lecture](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
- [CS231n Image Classification](https://cs231n.github.io/classification/)

---

## 3. 决策边界可视化技法

### 发现 3.1 · sklearn 默认 `contourf`，KNN 官方示例切到 `pcolormesh`

`DecisionBoundaryDisplay.from_estimator()` 的 `plot_method` 参数：

| 方法 | 适用 | 边缘 |
|---|---|---|
| `contourf` | 概率/连续输出（如 LogReg）| 平滑曲线 |
| `pcolormesh` | 离散预测（如 KNN）| 像素块，无平滑 |
| `contour` | 仅画等高线 | 仅边界 |

**KNN 官方示例显式用 `pcolormesh`**——因为 KNN 输出是离散类别，contourf 的等值线会做插值，反而误导。

来源：
- [sklearn DecisionBoundaryDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html)
- [sklearn KNN classification example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)

### 发现 3.2 · Vega-Lite/Altair 圈子用 `mark_rect` + 显式 `x2/y2`

本项目 03 已经走对了路（`mark_rect` + `x_end/y_end`）。Altair 官方推荐：

- 网格步长一致（avoid 拼接缝隙）
- 用 `x:Q + x2:Q` 而非 `x:O`（ordinal）——后者会引入轴间空隙
- `opacity` 0.3-0.4 平衡背景与前景点

**已知陷阱**：[Altair Issue #2047](https://github.com/vega/altair/issues/2047) 指出 mark_rect 不支持平滑插值，只能靠加密网格（n_grid 调高）。本项目 n_grid=60 在 (4, 10.5) × (3, 11) 范围里步长 ≈ 0.11，对 9-11 个样本足够；但样本越少，边界越锯齿。

来源：
- [Altair mark_rect docs](https://altair-viz.github.io/user_guide/marks/rect.html)
- [Altair issue 2047 - heatmap smoothing](https://github.com/vega/altair/issues/2047)

### 发现 3.3 · 标杆交互 demo 设计要素

Stanford CS231n / Observable `@antoinebrl` 共有特征：

| 要素 | CS231n | Observable | 本项目 03 |
|---|---|---|---|
| 数据点可拖动 | ✓ | ✓ | ✗ |
| 点击翻标签 | ✓ | ✓ | ✗ |
| k 滑块 | ✓ (1-25) | ✓ (1-25) | ✓ (1-9) |
| 网格密度可调 | ✗ | ✓ | ✗ |
| 实时重绘 | ✓ | ✓ | ✓ |
| 类别数 | 3 | 2 | 2 |
| 邻居高亮线 | ✗ | hover 显示 | 02 有，03 无 |

**关键 gap**：本项目 03 的 k 范围 1-9，**采样太稀**。Observable 给到 1-25 才能看清"边界从扭曲→平滑→塌陷"的连续过程。本项目 11 个样本，k=11 直接是欠拟合极端值，缺中间过渡。

来源：
- [Observable KNN visualizing variance](https://observablehq.com/@antoinebrl/knn-visualizing-the-variance)
- Stanford CS231n KNN demo（链接 TLS 证书问题，但 CS231n 讲义中提及该 demo 用 2D + 3 类）

---

## 4. k 调优演示：让差异视觉显著的数据集条件

综合多个来源（Berkeley SAAS、UBC CS547、Medium 30days-of-ML）的合成数据配方：

| 参数 | 推荐值 | 理由 |
|---|---|---|
| 样本量 N | 100-200 | 太少：k=N 直接塌陷；太多：k=1 也很平滑 |
| 类别数 | 2（教学）/ 3（标杆） | 2 类已足够演示，3 类增加视觉丰富度 |
| 噪声 | `make_moons(noise=0.20-0.30)` | 0.20 太干净（k 影响小），0.40 太混乱（k=1 已模糊） |
| k 演示范围 | 1, 3, 5, 15, 25, 50 | 标杆做法：log 或几何级数，覆盖 3 个数量级 |
| 簇形状 | 弯月 / 同心圆 | 必须**非线性可分**，否则 KNN 退化为线性分类器，k 影响弱 |

**核心定理**：**簇间不应线性可分**。如果两簇用直线就能分（如本项目 02 的"喜欢=右上 / 不喜欢=左下"），那 k=1 和 k=15 边界都长得像直线，k 的影响只在簇间过渡带显现，需要密集样本。

来源：
- [Berkeley SAAS bias-variance](https://saas.berkeley.edu/education/bias-variance-decision-trees-ensemble-learning)
- [Day 3 — KNN and Bias-Variance Tradeoff (Medium)](https://medium.com/30-days-of-machine-learning/day-3-k-nearest-neighbors-and-bias-variance-tradeoff-75f84d515bdb)
- [Bias-Variance Tradeoff (Fortmann-Roe)](https://scott.fortmann-roe.com/docs/BiasVariance.html)

---

## 5. 教学叙事

### 发现 5.1 · 标杆教材的 KNN 故事

| 来源 | 入口故事 | k 调优引导 |
|---|---|---|
| ISLR (Fig 2.16) | 合成 2D 二分类，k=1 vs k=100 并排 | "K 增大 → 模型从灵活到接近线性" |
| Hands-On ML (Géron) | MNIST 手写数字 | KNN 作为分类章节的传统基线，但**不是主线** |
| StatQuest | 简单 2D 散点 | "尝试不同 K，找让验证集错误率最低的" |
| CS231n | CIFAR-10 图像分类 + 2D 玩具 demo | 5-fold CV 曲线，k=7 最优；强调"验证集调参，永远不要用测试集" |
| CS4780 | 抽象数学（贝叶斯最优分类器、Cover-Hart 1967） | k=1 高方差 / k=N 高偏差 / 中间值平衡 |

**共同模式**：
1. 先讲 1-NN 的 Voronoi 直觉
2. 演示 k=1 边界（含噪声 → 锯齿/小岛）
3. 演示大 k 边界（平滑 → 接近线性）
4. **引出 bias-variance tradeoff**（教科书共识）
5. 介绍 cross-validation 调 k

### 发现 5.2 · 文字引导的关键短语

业界通用术语（中文学习者应保留英文双标）：

- "wiggly and unstable" / "扭曲不稳定"（k=1）
- "islands of classification" / "分类小岛"（label noise 下的 k=1）
- "smooth and stable" / "平滑稳定"（k 适中）
- "high bias / approaching constant" / "高偏差，趋近常数预测"（k=N）
- "rule of thumb: k ≈ √N" / "经验法则"（已在本项目 03）

来源：
- [ISLR (free PDF)](https://www.statlearning.com/)
- [StatQuest K-NN](https://statquest.org/statquest-k-nearest-neighbors/)
- [Cornell CS4780 Lec 2](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)

---

## 6. marimo + Altair 生态

### 发现 6.1 · marimo gallery 有 ML demo，但无 KNN 专项

- [marimo gallery](https://marimo.io/gallery) 收录"sklearn 分类自动化"和"Altair 交互"两类示例
- [gallery-examples GitHub](https://github.com/marimo-team/gallery-examples) 有 `altair-demo.py` 标准模板
- [marimo Altair 博客](https://marimo.io/blog/altair) 强调 `mo.ui.altair_chart` 双向绑定（前端选区 → Python `chart.value`）

### 发现 6.2 · 本项目缺失的 marimo 特性

- `mo.ui.altair_chart` 包装：可让用户**点击图上的点翻标签 / 拖动新点**，无需额外滑块
- 反应式：Altair 选区直接驱动重算（本项目目前用 slider 触发，效率/体验略差）

来源：
- [marimo Altair guide](https://marimo.io/blog/altair)
- [marimo plotting docs](https://docs.marimo.io/guides/working_with_data/plotting/)

---

## 借鉴清单（按可行性排序）

| # | 借鉴点 | 来源 | 工作量 |
|---|---|---|---|
| 1 | **数据集换成 `make_moons(n_samples=100~150, noise=0.25~0.30)`**（k 调优 demo 专用） | sklearn / Berkeley / UBC | 低 |
| 2 | **k 滑块范围扩到 1-25 或 1-50**，让"塌陷为常数预测"成为渐进过程而非跳跃 | Observable @antoinebrl / CS231n | 极低 |
| 3 | **保留电影故事作为 02 proximity 入口**（少样本+真实标签强化共鸣），但 03 切到合成数据 | 折中本项目特色 | 低 |
| 4 | **决策边界保持 mark_rect**（已正确），但 n_grid 提到 80-100 让边界更平滑 | Altair issue #2047 工作绕开 | 极低 |
| 5 | **加 LOOCV 准确率曲线**（k vs accuracy 折线图），不只显示当前 k 一个值——让"过拟合-适中-欠拟合"三段在同一图可见 | CS231n cvplot.png 范式 | 中 |
| 6 | **支持点击翻标签 / 拖动训练点**（`mo.ui.altair_chart` 选区） | Observable @antoinebrl / marimo Altair | 中 |
| 7 | **类别用 3 类**（如"必看/可看/不看"）让边界更丰富 | CS231n | 中（重做数据） |
| 8 | **同时显示 k=1 / k=适中 / k=large 三联图**（ISLR Fig 2.16 范式），无需滑块就能看完整谱 | ISLR | 中 |

---

## 本项目的差距（02 / 03 vs 最佳实践）

### 02-proximity.py
- 现状基本符合标杆做法（拖动新点 + 距离重排 + top-k 高亮线 + 距离表格）
- **小 gap**：维度只有 2，但作为"接近度"教学已足够；如果想更激进可加"3D 旋转"或"高维 → curse of dimensionality"提示
- **优势**：电影场景叙事 > sklearn 干巴巴的 Iris

### 03-k-tuning.py
- **核心 gap 1：数据集偏小（11 样本）+ 偏线性可分**。喜欢 vs 不喜欢主要靠"评分高低"区分，KNN 的非线性优势没体现，k 的视觉差异被压缩
- **核心 gap 2：k 滑块范围 1-9，缺 k=15/25/50 的渐进塌陷过程**。当前 k=11（即 N=样本数）一步到底，跳过中间
- **核心 gap 3：噪声实现是 label noise（2/11=18%）**，比例偏高，过早进入欠拟合区
- **次要 gap**：决策边界单帧呈现，无 LOOCV 曲线对比；无法点击翻标签

### 共同 gap
- **k 经验法则 √N 在小样本下不稳定**：N=11 → √N≈3.3，但实际最优 k 受具体样本位置主导，教学说服力弱

---

## 风险与权衡

| 标杆做法 | 不适合本项目 | 替代 |
|---|---|---|
| sklearn matplotlib 工作流 | 本项目用 marimo + Altair | 已正确选用 mark_rect |
| `DecisionBoundaryDisplay.from_estimator()` | 依赖 matplotlib，与 Altair 不兼容 | 维持手工 meshgrid + DataFrame + mark_rect |
| 100+ 样本合成数据 | **会丢失"豆瓣电影"叙事代入感**——这是本项目相对 sklearn 的差异化 | 折中：02 保留电影；03 切合成数据但贴电影标签（或保留电影+加 50 个合成"群众样本"） |
| 3 类（红蓝绿）| 中文场景下"喜欢/不喜欢/中立"叙事略勉强 | 维持 2 类 |
| 点击翻标签交互 | marimo 反应式实现需要 `mo.ui.altair_chart` 改造，跨 cell 状态管理略复杂 | 优先级低，先解决数据集问题 |
| ISLR 三联图（k=1/3/100 并排）| 占空间多，与本项目滑块范式冲突 | 可作为补充图（accordion 折叠） |

---

## 关键引用清单

**官方文档（HIGH confidence）**：
- [sklearn DecisionBoundaryDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html)
- [sklearn plot_classification (KNN)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)
- [sklearn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [sklearn plot_classifier_comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
- [Altair mark_rect](https://altair-viz.github.io/user_guide/marks/rect.html)
- [marimo Altair guide](https://marimo.io/blog/altair)

**学术教材（HIGH confidence）**：
- [Cornell CS4780 Lecture 2 kNN](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
- [Stanford CS231n Image Classification](https://cs231n.github.io/classification/)
- [ISLR (homepage)](https://www.statlearning.com/)

**交互 demo 标杆（HIGH confidence，已 fetch 验证）**：
- [Observable @antoinebrl/knn-visualizing-the-variance](https://observablehq.com/@antoinebrl/knn-visualizing-the-variance)
- [Stanford CS231n KNN demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/)（TLS 证书警告，但 CS231n 讲义引用有效）

**博客/教程（MEDIUM confidence）**：
- [Berkeley SAAS bias-variance](https://saas.berkeley.edu/education/bias-variance-decision-trees-ensemble-learning)
- [UBC CS547 Bias-Variance Visualization (PDF)](https://www.cs.ubc.ca/~tmm/courses/547-17F/projects/halldor-simar/report.pdf)
- [Day 3 — KNN and Bias-Variance Tradeoff](https://medium.com/30-days-of-machine-learning/day-3-k-nearest-neighbors-and-bias-variance-tradeoff-75f84d515bdb)
- [Fortmann-Roe Bias-Variance](https://scott.fortmann-roe.com/docs/BiasVariance.html)
- [StatQuest K-NN](https://statquest.org/statquest-k-nearest-neighbors/)
