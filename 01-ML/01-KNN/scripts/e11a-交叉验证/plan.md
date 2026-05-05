# E11a · 交叉验证完整版 · 计划

## 拆分决策

E11 原计划单期 45 min（CV + GridSearch + MNIST），信息密度极高。决策：拆 e11a + e11b。

- **e11a · 交叉验证（CV）** ~18 min：单切分的运气问题 → CV 思路 → 1/√n 误差缩放 → 三集职责 → cv=4 走查 → cross_val_score API
- **e11b · 网格搜索 + 数字识别** ~22 min：GridSearchCV API → Pipeline 防泄漏 → 维度爆炸 → 手写数字综合实战 → 整章 KNN 收束

理由：
1. CV 是独立完整概念（核心是 1/√n 直觉 + 三集职责），单独成期可以讲透
2. GridSearch + 数字识别是另一套工具链 + 综合实战，单独成期能给数字识别足够时间
3. 三期总和 ~55 min（10 min e10 + 18 min e11a + 22 min e11b），节奏更合理
4. 整章 KNN 收束（核心 1 / 变种 2 / 三大问题 / 三件套 / 缩放 / 超参搜索）放 e11b 最自然

## 覆盖

- `05-hyperparameter/01-交叉验证.md`（全）
  - 单切分运气问题（6 个 random_state 实测：0.90 ~ 1.00）
  - CV 核心思想（n 折轮流当验证集取均值）
  - 体重秤类比 + 1/√n 误差缩放公式
  - 三集职责：训练折 / 验证折 / 测试集
  - cv=4 数学走查
  - sklearn `cross_val_score` API
- `05-hyperparameter/README.md` 关于 LOOCV 的延展（e02 留下的伏笔）

## Demo 片段

- `05-hyperparameter/demos/01-cv-folds.py`（CV 折数 vs 单次拆分方差对比，端口 2755）

**实际 UI**（按这个写录屏稿，**已读 demo 代码**）：
- 系列共 3 个独立 demo：① `01-cv-folds.py` CV 折数对方差（本期）② `02-gridsearch.py` GridSearchCV 热力图（e11b）③ `03-digits.py` 手写数字 KNN（e11b）
- **控件**（hstack）：
  - `cv_k` 滑块（1-30, step=2, default=5）—— 固定 k 看抖动
  - `cv_folds` 下拉（"2", "3", "5", "10", "LOO"，default="5"）
  - `cv_n_repeats` 滑块（1-30, default=20）—— 不同 random_state 重复次数
- **方差对比图**：boxplot + strip plot 并排两行
  - 红色行：单次 train/test 准确率（20 个种子的散点）
  - 绿色行：CV-n 平均（同样 20 个种子，每个跑一次 CV）
  - 横轴：准确率 0.7-1.02
- **数字总结**：单次均值/标准差 vs CV 均值/标准差
- 配文："CV 把单次拆分变成 n 次取平均，标准差按 1/√n 缩"

**关键档位**：

| 档位 | 现象 | 教学要点 |
|---|---|---|
| folds=5, repeats=20, k=5 | 红行宽（std~0.04），绿行窄（std~0.01） | 主对比，主线 |
| folds=10, k=5 | 绿行更窄（std~0.005-0.008） | 折数翻倍 → 误差缩 √2 |
| folds=2, k=5 | 绿行变宽（接近红行） | 折数太少 CV 优势减弱 |
| folds=LOO, k=5 | 绿行极窄但计算慢 | LOOCV 是 e02 留的伏笔 |
| 切到 k=11 | 同样的对比，新数字 | 验证 CV 不依赖具体 k |

**states 操作清单**（草拟）：

```yaml
states:
  1. intro    — 静帧介绍 demo UI
  2. baseline — folds=5, repeats=20, k=5
                 expect: 红行 std ~0.04，绿行 std ~0.01；红行明显比绿行宽
  3. fold-up  — folds=10, k=5, repeats=20
                 expect: 绿行 std ~0.005-0.008（更窄）
  4. fold-low — folds=2, k=5, repeats=20
                 expect: 绿行 std 接近 0.02-0.03（CV 优势缩小）
  5. loo      — folds=LOO, k=5, repeats=20
                 expect: 绿行最窄（但计算慢，明显延迟）
```

## 砍掉

- `cv_results_` 详细字段 → 留给 e11b 讲 GridSearchCV 时再讲
- nested CV / stratified K-fold 等 CV 变种 → 教材没明显展开，e11a 给一句话"sklearn 默认对分类自动启用 stratified"
- 1/√n 公式数学推导 → 给体重秤类比 + 表格验证就够（教材已经做了），不展开方差/标准差代数

## 口吻提示

- E11a 核心命题：**数据自己挑 K，要先有可信评分**——CV 解决评分可信问题
- 接 e10 cta："手动选 k 的尴尬"——开篇直接抛出 k=5 是不是最优的问题
- **关键直觉**：把"称体重 1/√n"类比讲透，这是 CV 误差缩放的核心
- **三集职责**讲清楚：训练折 / 验证折 / 测试集，**测试集藏到最后**——这是工程纪律
- 第 2 期留下的"留一交叉验证"伏笔在这里兑现：LOOCV 是 cv=N 的极限特例
- cta 末句："交叉验证给我们一个可信评分，下一期我们用它做超参数搜索——网格搜索 + 手写数字识别综合实战，KNN 章节也就此收束。如果讲得有不对的地方，欢迎在评论区指正。"

## 段落骨架（7 段，~18 min）

| id | type | topic | 源 | 字数 |
|---|---|---|---|---|
| 1 | hook | 接 e10：k=5 是不是最优？单切分运气问题 | A | ~180 |
| 2 | content | 6 个 random_state 实测：0.90~1.00 跨度 | A | ~250 |
| 3 | content | CV 核心思想：n 折轮流当验证集取均值 + cv=4 走查 | A | ~330 |
| 4 | content | 体重秤类比 + 1/√n 误差缩放 + 兑现 LOOCV 伏笔 | A | ~340 |
| 5 | content | 三集职责：训练折 / 验证折 / 测试集 | A | ~280 |
| 6 | content | **Demo · CV 折数对方差**（01-cv-folds 实测对比） | **B** | ~340 |
| 7 | cta | sklearn `cross_val_score` API + 钩 e11b | A | ~280 |

合计 ~2000 字 + demo 录屏 ~3 min → 估算 **~17-18 min**

## 待办

- [ ] script.json 正稿
- [ ] 实跑 demo 校准 baseline / fold-up 标准差数字
- [ ] cv=4 走查动画素材（4 行表格逐折切换）
- [ ] 1/√n 表格动画（n=4/16/100 误差缩 1/2、1/4、1/10）
- [ ] 三集职责图（训练折 + 验证折 + 测试集分层框图）
