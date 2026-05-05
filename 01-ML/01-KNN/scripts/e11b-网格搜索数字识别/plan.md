# E11b · 网格搜索 + 手写数字识别 + KNN 章总结 · 计划

## 角色定位

- **承接 e11a**：CV 给可信评分 → 这一期把 CV 嵌套进网格搜索，把"挑最优 k"自动化
- **KNN 章最后一期**：cta 段做整章 KNN 收束（核心 1 + 变种 2 + 三大问题 + 三件套 + 缩放 + 超参搜索）
- **下一站预告**：第 2 章逻辑回归 LR

## 覆盖

- `05-hyperparameter/02-网格搜索.md`（全）
  - 网格搜索本质：穷举所有超参组合，每组用 CV 评分
  - 维度爆炸：n × ∏cᵢ 训练次数公式
  - sklearn `GridSearchCV` API（best_score_ / best_params_ / best_estimator_ 三件套）
  - Pipeline + 双下划线命名（`knn2__n_neighbors`）
  - **Pipeline 防数据泄漏**：标准化必须包进 Pipeline，不能在 CV 之前 fit_transform
  - 替代方案点名（不展开）：RandomizedSearchCV / 贝叶斯优化
- `05-hyperparameter/03-手写数字识别.md`（全）
  - MNIST 数据集介绍（28×28 像素，但 demo 用 8×8 简化版 `load_digits`）
  - KNN 在低维清晰任务上的强表现（~98% 准确率）

## Demo 片段

- `05-hyperparameter/demos/02-gridsearch.py`（GridSearchCV 热力图，端口 2756）+ `03-digits.py`（手写数字，端口 2757）

**实际 UI**（按这个写录屏稿，**已读 demo 代码**）：

### 02-gridsearch.py · GridSearchCV 热力图
- 静态图（无控件，预先跑好的结果）
- 4 行 × 9 列热力图
  - 行：(uniform·p=1, uniform·p=2, distance·p=1, distance·p=2)
  - 列：n_neighbors ∈ [1, 3, 5, 7, 9, 11, 15, 21, 29]
  - 36 组超参 × cv=5 = 180 次训练
- 颜色：viridis 5 折 CV 准确率，色深=高，色浅=低
- 文本叠加：每格 3 位小数
- 下方文字：最优组合 `best_params_` + 最差组合 + 跨度 pp
- accordion 2 个：训练次数公式 / refit=True 机制

### 03-digits.py · 手写数字 KNN
- 顶部说明卡：测试集准确率（~98-99%）+ 训练集 1347 / 测试集 450
- 控件：`pick` 滑块（0-449，选测试样本编号）
- 主图：当前测试样本 8×8 灰度图（mark_rect + 灰度 viridis）
  - 标题写：测试样本 #N · 真实=X · 预测=Y · ✓ / ✗
- 概率分布柱状图：predict_proba 输出 10 个类别（绿色高亮真实类）
- Top-3 邻居 8×8 缩略图：3 张并排，下方列出每张的标签 + 距离

**关键档位**：

| Demo | 档位 | 现象 | 教学要点 |
|---|---|---|---|
| 02-gridsearch | 静态 | 36 组热力图 best=distance·p=2·k=9 · CV 0.967 | 网格搜索 + best_params_ |
| 03-digits | pick=0 | 真实=数字 X，预测=X ✓，proba 单峰 | KNN 跑通 |
| 03-digits | pick=310 | 真=8 预=1，proba 双峰 {1:0.689, 8:0.311}，top-3=[1,1,8] | 看 KNN 错在哪 |

**states 操作清单**（草拟）：

```yaml
states:
  1. gs-intro     — 打开 02-gridsearch demo，静帧介绍热力图
  2. gs-best      — hover 最优格子（distance·p=2·k=9），tooltip 显示 0.967
  3. dg-intro     — 切到 03-digits demo，pick=0
                    expect: 真实=显示的数字，预测=同 ✓，proba 集中
  4. dg-correct   — pick 拖到第二个，预测仍对
                    expect: 类似上面但不同数字
  5. dg-wrong     — pick=310（真=8 预=1）
                    expect: 预测 ✗，proba 双峰 {1:0.689, 8:0.311}，top-3=[1,1,8]
```

## 砍掉

- 28×28 真实 MNIST（Kaggle train.csv）→ demo 用 sklearn `load_digits` 8×8 简化版，本期同样口径，不展开 28×28
- `cv_results_` 字段细节展开 → 一句话带过：包含每组超参 5 折的 split0~split4 准确率 + 均值 + 标准差
- RandomizedSearchCV / 贝叶斯优化 → 教材点名了，本期同样点名（cta 推迟到后续优化章节）
- Pipeline 多步骤更复杂例子（多个 transformer）→ 本期只讲两步骤的 std + knn

## 口吻提示

- E11b 核心命题：**网格搜索把 CV 套外层循环自动化** + **整章 KNN 收束**
- 接 e11a："上一期 CV 解决了评分可信，这一期把外层循环自动化"
- **维度爆炸**这个数字直觉要讲透：1 维 6 个值 × cv=5 = 30 次；3 维各 10 个值 × cv=10 = 10000 次。线性加超参 → 指数级训练
- **Pipeline 防数据泄漏**讲清楚：CV 前 fit_transform 是隐蔽 bug，验证折提前摸过统计量
- 手写数字段：KNN 在 64 维（8×8）清晰任务上 ~98%，反直觉地强——后面深度学习时代来临前，KNN 一直是 MNIST 基线
- **cta 段做整章 KNN 收束**——这是整个 KNN 章节最后一段口播，要厚重一点
  - 核心 1 个：找 K 个最近邻
  - 变种 2 个：分类投票 / 回归平均
  - 三大问题：距离怎么算 / K 取多少 / 决策边界长啥样
  - sklearn 三件套：KNeighborsClassifier / KNeighborsRegressor / GridSearchCV
  - 缩放：归一化 / 标准化（量纲对齐）
  - 超参搜索：CV + 网格搜索
- 下一站第 2 章 LR 逻辑回归 logistic regression 预告：从 KNN 这种"不学只记"的方法，进入"真正学一个数学模型"的世界
- cta 末句："这就是 KNN 完整章节的全部。下一站第 2 章逻辑回归，我们进入真正学一个数学模型的世界。如果讲得有不对的地方，欢迎在评论区指正。下期见。"

## 段落骨架（9 段，~22 min）

| id | type | topic | 源 | 字数 |
|---|---|---|---|---|
| 1 | hook | 接 e11a：CV 评分 + 外层循环自动化 | A | ~150 |
| 2 | content | 网格搜索本质：穷举 + 网格表 | A | ~280 |
| 3 | content | 维度爆炸 + 训练次数公式 | A | ~250 |
| 4 | content | sklearn `GridSearchCV` 三件套 + Pipeline 双下划线 | A | ~330 |
| 5 | content | Pipeline 防数据泄漏（关键工程纪律） | A | ~280 |
| 6 | content | **Demo · GridSearchCV 热力图**（02-gridsearch 36 组实测） | **B** | ~330 |
| 7 | content | **Demo · 手写数字识别**（03-digits + KNN 在 64 维表现） | **B** | ~340 |
| 8 | content | 整章 KNN 收束：核心 1 + 变种 2 + 三大问题 + 三件套 + 缩放 + 超参搜索 | A | ~370 |
| 9 | cta | 下一站第 2 章 LR 预告 + 章节告别 | A | ~300 |

合计 ~2630 字 + demo 录屏 ~5 min → 估算 **~21-22 min**

## 待办

- [ ] script.json 正稿
- [x] 实跑 02-gridsearch.py 校准 best_params_：实测 distance·p=2·k=9 · CV 0.9667（Step 3.5 已核）
- [x] 实跑 03-digits.py 校准手写数字测试准确率：实测 0.9844 / 错 7 张（Step 3.5 已核）
- [ ] 网格搜索动画素材：1 维 → 2 维 → 3 维网格表
- [ ] 维度爆炸表格动画
- [ ] Pipeline 防泄漏图：CV 前 fit_transform 错 vs Pipeline 包起来对
- [ ] 整章 KNN 收束思维导图（A 段最重）
