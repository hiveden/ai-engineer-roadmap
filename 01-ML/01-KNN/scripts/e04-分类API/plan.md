# E04 · 分类 API · 计划

## 覆盖

- `02-api/01-分类API.md`（全）
- 兑现 E03 cta 留的"下一站 sklearn 一行代码跑五步"和"lazy learner"伏笔

## Demo 片段

**没有 marimo demo**，本期是 Jupyter 代码录屏（B 段，但工具切到 Jupyter）。

录屏脚本（实测过）：

```python
# Cell 1 · 9 部电影数据（同 E01）
X = [[8.4, 9.0], [8.7, 8.5], [9.5, 9.5],
     [6.4, 7.0], [2.9, 3.5],
     [8.0, 8.0], [8.5, 9.2],
     [7.5, 6.5], [7.4, 8.0]]
y = ['喜欢', '喜欢', '喜欢',
     '不喜欢', '不喜欢',
     '喜欢', '喜欢',
     '不喜欢', '不喜欢']

# Cell 2 · 三件套
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)
print(model.predict([[8.5, 9.5]]))   # ['喜欢']

# Cell 3 · lazy learner 实证
import time
t0 = time.perf_counter(); model.fit(X, y); t_fit  = time.perf_counter()-t0
t0 = time.perf_counter(); model.predict([[8.5, 9.5]]); t_pred = time.perf_counter()-t0
print(f'fit  耗时 {t_fit*1000:.2f} ms')   # ~0.4 ms
print(f'predict 耗时 {t_pred*1000:.2f} ms')  # ~19 ms（含初始化）

# Cell 4 · model.__dict__ 看 fit 存了什么
model._fit_X.shape    # (9, 2)
model.classes_        # array(['不喜欢', '喜欢'], dtype='<U3')
```

录屏 states（粗对齐 + 静帧拉伸）：

```yaml
states:
  1. cells_intro  — 静帧展示 Notebook 三个 cell（数据 / 模型 / 实证），不执行
  2. run_3lines   — 执行 Cell 2，输出 ['喜欢']
                    expect: 输出区显示 ['喜欢']
  3. timing       — 执行 Cell 3，对比 fit 和 predict 耗时
                    expect: fit ~0.4ms / predict ~19ms · 数量级反差明显
  4. peek_inside  — 执行 Cell 4，显示 _fit_X.shape (9,2) 和 classes_
                    expect: shape 输出 (9, 2) · classes_ 显示俩类别

cue 词:
  "我们打开 Jupyter"             → state 1
  "三件套：构造、fit、predict"   → state 2
  "我们给 fit 和 predict 计时"   → state 3
  "看看 fit 存了什么"            → state 4
```

## 砍掉

- 教材开头那段 dog/cat/duck/xx 的"四样本三类别"早期 demo（一维玩具，已被 9 部电影替代，重复且不直观）
- "sklearn 包名 ≠ 导入名" 工程坑点（与 ML 主题无关，cta 不展开）
- "三件套通用模板"完整列出决策树/SVM 等其他算法（在 cta 列名字即可，不展开）
- 教材"故事元素 → 术语名"完整表（口播挑核心：X / y / fit / predict 4 个，其余字幕带过）

## 口吻提示

- 第一人称"我们"
- 关键对照：E01 的 9 行手算公式 → 这一期 3 行 sklearn（视觉锚连续性）
- lazy learner 是本期高潮，必须给它一段独立的 segment（seg 5），不能轻描淡写
- 工程术语白话化：
  - "API" → 首次给中英标 + "工具接口"
  - "lazy learner" → 先用"懒学生 / fit 几乎不做事" 直觉铺垫，再正式给术语
  - "封装" → "把刚才那 5 步流程打包到一个零件里"
- cta 末句固定："如果讲得有不对的地方，欢迎在评论区指正。"

## 跨期衔接钩

- **E03 → E04 接住**：E03 cta 承诺"下一站 sklearn 一行代码跑五步"+"lazy learner 推迟到 sklearn 章" → E04 hook 必须正面呼应这两个点
- **E04 → E05 留伏笔**：cta 末提"分类切回归只换一个零件" → E05 hook 用"换零件"接住

## 段落骨架（8 段）

| id | type | topic | 源 | 字数 |
|---|---|---|---|---|
| 1 | hook | 接住 E03：sklearn 一行跑五步 + lazy learner 兑现预告 | A | ~170 |
| 2 | content | sklearn 是什么 + 三件套总览（构造 / fit / predict）| A | ~330 |
| 3 | content | Jupyter 录屏：3 行 sklearn 跑通 9 部电影分类 | B | ~400 |
| 4 | content | 对照 E01：9 行手算 vs 3 行 sklearn 视觉锚 | A | ~290 |
| 5 | content | **lazy learner 实证**：fit/predict 计时 + 看 fit 存了什么 | B | ~270 |
| 6 | content | **lazy learner 概念定名** + 对照 eager learner | A | ~210 |
| 7 | content | 三件套是 sklearn 全家桶通用模板 | A | ~250 |
| 8 | cta | 概念地图 + E05 伏笔（换零件）| A | ~360 |

合计 ~2280 字 → 估算 ~12 min（中文 220 字/min ≈ 10 min 口播 + demo 录屏 ~2 min）

**注**：seg 5 + seg 6 = lazy learner 完整高潮（实证 → 定名），从原 470 字单段拆为两段，避免超 450 上限并给概念定名段独立呼吸空间。

## 待办

- [ ] script.json 正稿 3-Agent 审稿
- [ ] Jupyter 录屏脚本实跑（验证 timing 数量级反差稳定）
- [ ] 与 E03 cta 文本逐字对账，确认衔接钩落实
