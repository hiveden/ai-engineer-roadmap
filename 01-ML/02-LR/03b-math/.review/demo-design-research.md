# 数学交互 Demo 设计调研

> 日期：2026-05-02 · 调研对象：03b-math 两个 demo（partial-derivative.py / matrix-mul-viz.py）
> 调研基线：3blue1brown / Distill / Bret Victor / setosa.io / immersivemath / Khan Academy / R2D3
> 输出受众：Tech Lead Alex（自查 + 修订 `_3-demo-guide.md` 决策依据）

---

## 1. 业界最佳实践（5 个核心原则）

### 原则 1 · "一个核心隐喻 vs 多个解构视图"——业界倾向**单核心隐喻**

3b1b 教偏导/梯度只用**一个**隐喻：函数 = 空间变换，导数 = 局部拉伸/挤压系数（chapter 12 "The other way to visualize derivatives"）。教矩阵乘法只用**一个**隐喻：矩阵 = 线性变换 = 网格的拉伸旋转，乘法 = 变换的复合（chapter 3-4）。3b1b **不**同时把"颜色块 + 几何变换 + 切片"三个隐喻怼一屏，因为隐喻切换是高认知成本动作。

**参考**：[3b1b · Linear transformations](https://www.3blue1brown.com/lessons/linear-transformations/) · [3b1b · Matrix multiplication as composition](https://www.3blue1brown.com/lessons/matrix-multiplication/) · [3b1b · The other way to visualize derivatives](https://www.3blue1brown.com/lessons/derivatives-and-transforms/)

### 原则 2 · "渐进披露（Progressive Disclosure）"——Distill / immersivemath 都遵守

Distill 标杆文章 [Why Momentum Really Works](https://distill.pub/2017/momentum/) 把交互按"读者卷起来"的节奏切：先看一个 1D 抛物线（一个滑块），滚一段；再换 phase diagram；再换 α-β 收敛热图。**每屏只一个图 + 一两个滑块**，从不一次性把 5 张图都堆到一屏。immersivemath（[Ström/Åström/Akenine-Möller](http://immersivemath.com/ila/tableofcontents.html)）每个 illustration **1-2 个滑块**，多了就拆下一段。

### 原则 3 · "Linked views 必须服从一个主叙事"——R2D3 / 设o sa.io 的金标准

[R2D3 决策树](https://r2d3.us/visual-intro-to-machine-learning-part-1/) 是 scrollytelling 的天花板：图永远是同一张主图，**视图变化由叙事推进而非控件**。[setosa.io PCA](https://setosa.io/ev/principal-component-analysis/) 双视图（原始空间 ↔ PC 空间）联动，但**两张图同位同构、互为镜像**——是同一个数据的两种坐标系，不是 5 个独立视角。**多视图联动 ≠ 把所有视角全摆出来**，而是**一个核心视角 + 一个互为镜像的辅助视角**。

### 原则 4 · "5 秒测试"——单焦点>炫技

学界（认知负荷论 [Sweller / Springer 2025](https://link.springer.com/article/10.1007/s10648-021-09624-7)）已确认：交互媒体里"redundant elements" 和"过度真实"会引入 task-irrelevant load。Bret Victor 在 [Explorable Explanations](https://worrydream.com/ExplorableExplanations/) 里讲的"reactive document"原则——**每次交互应回答一个具体问题**，控件不是给的越多越好，而是"低门槛点拽 → 立刻看到一个具体的因果关系"。Alex Rogozhnikov 直接批评过：[过度的交互演示反而让学生绕开数学本身](https://arogozhnikov.github.io/2016/04/28/demonstrations-for-ml-courses.html)。

### 原则 5 · "几何隐喻>颜色块"——尤其矩阵乘法

3b1b、immersivemath、setosa 教矩阵乘**全部**用几何变换（基向量落在哪、网格怎么扭）。颜色块仅作**辅助计算示意**，不作为主隐喻。原因：颜色块解释"怎么算"（已经会算的人看着舒服），几何变换解释"它在干嘛"（学生最缺的直觉）。LR 场景里**矩阵=数据排布**反而是颜色块更合适，但要明确"这不是教线性代数主义，是教数据视角下的批量化"。

---

## 2. partial-derivative.py 评估

### ✅ 当前做对了什么

1. **视图 1 用独立 g(x)=(x-2)²+1 做一元导数热身**——好。先 1D 后 2D 是 3b1b 也用的渐进路径（chapter 1→2→...→12）。一元函数有红点+橙虚切线+绿钻极小，**5 秒内能看懂"切线斜率=导数"**。
2. **状态卡的徽章颜色（红/黄/绿 = 远/近/到极值）**——好。这是 [Distill momentum](https://distill.pub/2017/momentum/) 也用的"立即反馈"范式：参数变化 → 一个明确的语义标签变化。
3. **鞍点（saddle）作为反例选项**——好。3b1b 也强调"必要非充分条件"的反例，避免让学生以为梯度=0 就一定是极值。
4. **梯度箭头叠在等高线上**——好。这是 [Andrew Ng CS229](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf) 的标准范式，业界共识。

### ⚠️ 反模式（含证据）

#### 反模式 1：5 视图同屏 = 隐喻切换轰炸（**高严重度**）

视图 1（一元函数 g(x)=(x-2)²+1）+ 视图 2（二元 f(x,y)）+ 视图 2-x、2-y 切片 + 视图 3（3D 曲面）= **5 张图、2 套不同函数**。学生要在 5 秒内：
- 切换"g 是 1D 教学辅助" vs "f 是真正的多元函数"两套数学对象
- 切换"等高线俯视" vs "切片侧视" vs "3D 透视"三种坐标系
- 还要联动 x_cur / y_cur 滑块

**证据**：[3b1b 的偏导讲解](https://www.youtube.com/watch?v=dfvnCHqzK54)（"Partial derivatives and graphs"）整个视频**只用 1 个 3D 曲面 + 1 把刀切下去的红色平面**就讲完了 ∂f/∂x；Khan Academy 同样只用"slicing the graph"一个隐喻。我们用了 4 倍的视觉，但传递的信息密度不一定是 4 倍——**很可能只有 1.2 倍，剩下都是认知开销**。

#### 反模式 2：视图 1 用独立函数 g(x) ≠ 视图 2 的 f(x,y) → 隐喻断层（**中严重度**）

x_cur 滑块从 [-3,3] 线性映射到 g(x) 的 [-1,5]，**学生看不到这个映射**。结果：拖滑块时，视图 1 的红点和视图 2 的红十字**位置数字对不上**（视图 1 显示 x=4.5，视图 2 显示 x=2.5）——破坏了"单一来源原则"（你们自己 `_3-demo-guide.md` §交互状态决策树写过）。

应该：要么视图 1 直接用 f(x, y_cur) 沿 x 切的那条曲线（和视图 2-x 切片合并），要么把 g(x) 拿掉单独做成"前置 1D 热身 demo"。

#### 反模式 3：视图 2-x、2-y 双切片 = 信息冗余（**中严重度**）

`f(x,y) = x²+y²` 是各向同性的对称函数，两条切片**长得一模一样**，只是颜色（蓝/紫）不同。鞍点 `x²-y²` 才看得出区别（一条向上抛物线、一条向下抛物线）。**对称碗** preset 下，2-x 和 2-y 视图传递的有效信息 ≈ 1 张图。

应该：合并成"切片"单视图 + 一个 dropdown 切（沿 x / 沿 y），或者保留双切片但默认选鞍点 preset 让差异显现。

#### 反模式 4：视图 3 的 3D Surface 只是"装饰"（**低严重度**）

3D Surface 没有切片平面、没有梯度箭头、没有切线——只是"同一个 f 又画了一遍"。3b1b 的 3D 视图**总是有切片刀进去**（[chapter 12 transformations view](https://www.3blue1brown.com/lessons/derivatives-and-transforms/) 的核心动画）。我们这张 3D 图的边际信息增量低。

如果要保留 3D，应该：让 x_cur / y_cur 滑动时，**3D 上画一把切片平面**，这样和 2-x、2-y 切片图就联动起来了——切片图变成 3D 切片的 2D 投影。

### 必修 / 建议修改项

| 优先级 | 项 | 原因 |
|---|---|---|
| **必修** | 视图 1 改为 f(x, y_cur) 沿 x 切片，与视图 2 共享同一个 f | 修复"单一来源"破缺，消除 g(x) ↔ f(x,y) 的隐喻断层 |
| **必修** | 默认 preset 改成"鞍点"或"偏移碗" | 对称碗的双切片冗余度太高，鞍点才能展示偏导差异 |
| 建议 | 收 5 视图为 3 视图：1D 切片（合并 2-x/2-y） + 等高线（带梯度箭头） + 3D（带切片平面） | 渐进披露 / 单核心隐喻原则 |
| 建议 | 给 3D Surface 加切片平面（plane at y=y_cur 沿 x 切的红色透明矩形） | 让 3D 不只是装饰 |
| 可选 | 加一个"显示/隐藏切片"toggle，默认隐藏 1D 切片，只在用户点击时展开 | Distill 式渐进披露 |

---

## 3. matrix-mul-viz.py 评估

### ✅ 当前做对了什么

1. **视图 3（X·w → ŷ）作为 LR 矩阵化的最终落点**——绝佳。这正是 3b1b 风格的"compress to one expression"叙事高潮。把 4 个样本 `for` 循环压成一行 `X @ w` 是这个 demo 的最大价值。
2. **教学要点回顾收尾时点 numpy `X @ w` vs Python for**——好。把数学和工程接口接上了，符合 Alex 的"全栈视角"。
3. **状态卡里 dot product 公式逐项展开（绿色高亮结果）**——好。reactive document 的标准动作。

### ⚠️ 反模式（含证据）

#### 反模式 1：完全没用"几何变换"隐喻——和业界教法分叉（**高严重度**）

3b1b、immersivemath、setosa.io 教矩阵乘法**全部主推"基向量去哪了 / 网格怎么变形"**。你们视图 1 是向量加法（标量几何），跳到视图 2 直接颜色块，**中间缺了"矩阵作用于向量 = 几何变换"这一步**。

**严重后果**：学生学完这个 demo，会知道"矩阵乘=按行 dot product"（计算视角），但**不知道矩阵在干嘛**（语义视角）。后面学 PCA、SVD、激活函数线性部分时直觉断档。

**证据对比**：
- [3b1b chapter 3](https://www.3blue1brown.com/lessons/linear-transformations/) 用"网格被矩阵拉/旋转"开场，明确说"matrix = where do î and ĵ land"
- [immersivemath ch6](http://immersivemath.com/ila/ch06_matrices/ch06.html) 用"矩阵作用于矩形"
- [setosa PCA](https://setosa.io/ev/principal-component-analysis/) 用两个对照坐标系

**反驳**（如果 Alex 想反驳）：**这个 demo 是 LR 配套的，LR 里矩阵=数据矩阵不是变换矩阵**——这个反驳有效。但应该**明确写在 docstring 和章节首段**："本 demo 用数据视角看矩阵，不是线性变换视角，几何变换留到未来 PCA/SVD 章节再补"。否则学生不知道自己看到的是"局部视角"。

#### 反模式 2：视图 1（向量加法）和视图 2/3（矩阵乘）之间没有桥梁（**中严重度**）

视图 1 教 `a + b` 平行四边形 + dot product。视图 2 突然就 `A @ w` 了。**缺一个过渡视图**：把 `A @ w` 拆解为"`A` 的每一行就是一个向量，每行和 `w` 做 dot product 就是视图 1 教的那个点积"——这个连接是这个 demo 最该爆的"啊哈时刻"，但目前没显式画出来。

应该：在视图 2 里，当 `row_focus = i` 时，**右边小图画出 A_row_i 和 w 两个箭头叠加做 dot product 的几何**（cosθ + 长度），让视图 1 的几何点积语言和视图 2 的颜色块计算语言**连起来**。

#### 反模式 3：3 视图 × 各自独立控件 = 控件迷宫（**中严重度**）

视图 1：a_x, a_y, b_x, b_y（4 滑块）
视图 2：A_preset (dropdown), w1, w2, row_focus（4 控件）
视图 3：sample_focus（1 滑块） + 共享 w1/w2

**总共 9 个控件**。Distill momentum 整篇文章总共也就 2-3 个滑块。Bret Victor 反复强调"控件不是好东西，是不得已"——每多一个控件就多一份学习成本。

应该：
- 视图 1 的 4 个滑块**默认全部隐藏**，只展示一个"重置"按钮，让学生先看一个固定的好例子；想自己玩再展开。
- 视图 2 的 row_focus 改成"自动每 2 秒扫描一行"循环动画（参考 3b1b 风格），不需要手动拖。
- 视图 3 的 sample_focus 同上。

#### 反模式 4：颜色块图的"标签头 + 等号 + 乘号"是 altair vconcat 拼出来的——视觉对齐脆弱（**低严重度**）

代码里用 `alt.vconcat(header, body)` 拼"A (3×2) × w (2×1) = Aw (3×1)"的标签和等号。altair 不是为这种"几何对齐"设计的，**字号/字体/缩放变化时容易错位**。

应该：直接用 `mo.md` 写 LaTeX 形式 $A \mathbf{w} = \mathbf{y}$ 放在颜色块图上方，颜色块只画三块矩阵本身，**别在 altair 里拼版式**。

### 必修 / 建议修改项

| 优先级 | 项 | 原因 |
|---|---|---|
| **必修** | docstring + 章节首段明确"用数据视角，不是线性变换视角" | 避免学生形成只懂计算不懂语义的错觉 |
| **必修** | 视图 1 → 视图 2 加桥梁：高亮 `A` 的第 i 行作为一个"行向量"，画出它和 `w` 的几何 dot product | 这是这个 demo 最该爆的啊哈时刻，目前缺失 |
| 建议 | 控件大幅瘦身：默认隐藏视图 1 的 4 个滑块；row_focus / sample_focus 默认改自动扫描 | 9 控件 → 3 控件 |
| 建议 | 颜色块的标签头改用 `mo.md` 上方写 LaTeX，不在 altair 里拼版 | 视觉对齐稳定 |
| 可选 | 增加一个第 0 视图（最前面）：基向量 î 被 A 拉去哪、ĵ 被 A 拉去哪——埋伏笔，等 PCA 章节展开 | 给"几何变换隐喻"留一个引子 |

---

## 4. 5 秒测试（自我评估）

| Demo | 5 秒能 get 到什么 | 5 秒应该 get 到什么 | 差距 |
|---|---|---|---|
| partial-derivative | "有一堆图，红点能拖，下面有数字" | "偏导=切片导数，梯度=偏导组成的向量，极值=梯度为 0" | **大**——5 视图分散，视图 1 用了不同函数干扰 |
| matrix-mul-viz | "三块颜色矩形，红/绿框扫行" | "矩阵乘=每行做点积=批量化加权求和=一行代码吞 4 个样本" | **中**——视图 3 的叙事到位了，但视图 1→2 的桥断了 |

---

## 5. 标杆案例（带链接）

| # | 案例 | 看什么 | URL |
|---|---|---|---|
| 1 | 3b1b · Essence of Calculus chapter 12（导数的另一种视角） | 单一隐喻贯穿全章；3D 曲面切刀的标杆动画 | https://www.3blue1brown.com/lessons/derivatives-and-transforms/ |
| 2 | 3b1b · Essence of Linear Algebra chapter 3（线性变换） | 矩阵=网格变形；基向量去哪了 | https://www.3blue1brown.com/lessons/linear-transformations/ |
| 3 | 3b1b · Essence of Linear Algebra chapter 4（矩阵乘=变换复合） | 乘法=函数复合的几何动画 | https://www.3blue1brown.com/lessons/matrix-multiplication/ |
| 4 | Distill · Why Momentum Really Works | 渐进披露 / 多控件不挤同屏 / 联动视图金标准 | https://distill.pub/2017/momentum/ |
| 5 | Bret Victor · Explorable Explanations（原文） | reactive document / active reading 哲学源头 | https://worrydream.com/ExplorableExplanations/ |
| 6 | setosa.io · PCA explained visually | 双视图镜像联动；同一份数据两个坐标系 | https://setosa.io/ev/principal-component-analysis/ |
| 7 | Immersive Linear Algebra · ch6 矩阵 | 几何变换为主、颜色为辅；每屏 1-2 滑块 | http://immersivemath.com/ila/ch06_matrices/ch06.html |
| 8 | R2D3 · A Visual Introduction to Machine Learning | scrollytelling 的天花板；同一张主图叙事推进 | https://r2d3.us/visual-intro-to-machine-learning-part-1/ |
| 9 | Khan Academy · Partial derivatives and graphs | 单一 3D + 切片刀讲偏导（最简最有效） | https://www.youtube.com/watch?v=dfvnCHqzK54 |
| 10 | Alex Rogozhnikov · Demonstrations for ML courses（批评性视角） | "过度交互让学生绕开数学"——反思 demo 的边界 | https://arogozhnikov.github.io/2016/04/28/demonstrations-for-ml-courses.html |

---

## 6. `_3-demo-guide.md` 修订建议

**结论：是的，建议修订**。具体补三条原则、调一个表、改一处反模式列表。

### 6.1 §设计三原则 → 升级为"五原则"

补两条：

**原则 4 · 单核心隐喻**：一个数学概念**只用一种主隐喻**（几何/代数/数据/物理任一），别同屏混用。例：偏导用"切片刀"或"切线斜率"二选一，不要 5 视图怼上。

**原则 5 · 渐进披露**：默认露出**最少**的视图和控件，"想看更多"的让学生主动展开（accordion / toggle）。Distill 标准：每屏 1 个图 + 1-2 个滑块。

### 6.2 §标配可视化模式 → 加一行"几何变换"

| 模式 | 用途 | 实现要点 |
|---|---|---|
| **几何变换** | 教矩阵乘语义（不是计算） | 画 unit grid / unit square 经过 A 后的形变；标 î、ĵ 落点 |

LR 场景虽然主要是数据视角，但应**预告**这一隐喻存在，避免学生形成局部直觉。

### 6.3 §反模式 → 新增 3 条

- ❌ **多视图把所有视角全摆出来**：5 视图同屏 ≠ 5 倍信息，可能只是 1.2 倍信息 + 4 倍认知开销。砍到 2-3 视图聚焦。
- ❌ **视图间用不同函数对象做"教学辅助"**：违反单一来源原则，制造数字不对应的认知断层。例：视图 1 用 g(x)，视图 2 用 f(x,y) → 改用同一个 f 沿轴切片。
- ❌ **几何隐喻和颜色块隐喻随意切换**：教矩阵乘要么全程几何（3b1b 风格）要么全程颜色块（数据风格），混用会让学生抓不住主线。

### 6.4 §交互状态决策树 → 补一句

> 控件越少越好。每多一个控件就多一份学习成本。Distill 标杆是每屏 1-2 个滑块。多控件场景考虑：默认隐藏 + 折叠展开 / 改自动扫描动画 / 用 preset 替代滑块。

---

## 7. 优先级 TL;DR（给 Alex 的 3 条）

1. **partial-derivative · 必修**：视图 1 的 g(x) 替换为 f(x, y_cur) 沿 x 切片，消除"两套函数对象"的隐喻断层；默认 preset 改鞍点。
2. **matrix-mul-viz · 必修**：在 docstring + 章节首段明确"数据视角不是变换视角"；在视图 2 给 A 的高亮行加一个几何 dot product 的小图，让视图 1 → 视图 2 接上。
3. **`_3-demo-guide.md` 必修**：升级为五原则（补"单核心隐喻 + 渐进披露"），反模式新增"5 视图同屏"和"多视图用不同函数"两条。
