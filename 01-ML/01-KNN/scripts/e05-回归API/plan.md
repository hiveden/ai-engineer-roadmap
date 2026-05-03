# E05 · 回归 API · 计划

## 覆盖

- `02-api/02-回归API.md`（全）
- 接住 E04 cta 留的伏笔："3 行代码里只动 1 个字"
- 整章（02-api 两期）的最终收束

## Demo 片段

**没有 marimo demo**，Jupyter 代码录屏（B 段）。

录屏脚本（实测过）：

```python
# Cell 1 · 同 E04 的 9 部电影 X，y 换成具体打分
X = [[8.4, 9.0], [8.7, 8.5], [9.5, 9.5],
     [6.4, 7.0], [2.9, 3.5],
     [8.0, 8.0], [8.5, 9.2],
     [7.5, 6.5], [7.4, 8.0]]

# 你历史给每部打过的具体星数（1-5 分）
y = [4.5, 4.0, 4.3,   # 流浪地球 2 / 阿凡达 / 泰坦尼克
     3.0, 1.5,         # 哥斯拉 / 上海堡垒
     4.5, 4.5,         # 沙丘 2 / 复仇者联盟 4
     3.0, 3.5]         # 寒战 / 长津湖

# Cell 2 · 三件套，唯一变动是 import 和 类名
from sklearn.neighbors import KNeighborsRegressor   # ← 改这个 import
model = KNeighborsRegressor(n_neighbors=5)          # ← 改这个类名
model.fit(X, y)
print(model.predict([[8.5, 9.5]]))   # [4.36]

# Cell 3 · 验证 4.36 怎么来的（手验 5 个最近邻平均）
import numpy as np
from numpy.linalg import norm
arr = np.array(X); target = np.array([8.5, 9.5])
d = norm(arr - target, axis=1)
order = np.argsort(d)[:5]
neighbors = [(d[i].round(2), y[i]) for i in order]
# [(0.30, 4.5), (0.51, 4.5), (1.00, 4.3), (1.02, 4.0), (1.58, 4.5)]
mean_score = np.mean([y[i] for i in order])   # 4.36
```

录屏 states：

```yaml
states:
  1. cells_intro  — 静帧展示 Cell1 和 Cell2，未执行
  2. spot_diff    — 静帧 + 红框标 import 和类名（仅这两处与 E04 不同）
                    expect: 红框圈出 KNeighborsRegressor 两次
  3. run_predict  — 执行 Cell 2，输出区显示 [4.36]
                    expect: 输出 [4.36]
  4. verify_mean  — 执行 Cell 3，列出 5 个邻居和它们的分数 + 平均
                    expect: 5 行邻居数据 · 平均 = 4.36

cue 词:
  "Cell 1 数据部分"             → state cells_intro
  "Cell 2 你能找到差别吗"       → state spot_diff
  "运行——输出 4.36"            → state run_predict
  "我们手验一下"                → state verify_mean
```

## 砍掉

- 教材开头那段 `[[0,0,1],[1,1,0],...]` 三特征玩具数据 + 注释里的"成绩 / 薪资"伪数据（教材本身就有 9 部电影豆瓣版，玩具数据只是先讲再升级，视频直接用电影数据）
- "应用场景：KNN 回归适用于特征空间局部平滑" 这种结论性句子（说了观众也理解不了"局部平滑"的精确含义，本期不展开）
- `metric` 参数指向 03-distance（cta 提一下即可，不展开）

## 口吻提示

- 第一人称"我们"
- 这是整章 02-api 的收尾期，时长比 E04 短（10min vs 12min），核心策略是"高对照 + 短而锐"
- 整期主线："找不同游戏" —— 把 E04 和 E05 的代码并排放，让观众发现 99% 都一样
- 工程术语白话化：
  - "对偶" → "对照版本"
  - "API 形态完全一致" → "用法长得一模一样"
- cta 末句固定："如果讲得有不对的地方，欢迎在评论区指正。"

## 跨期衔接钩

- **E04 → E05 接住**：E04 cta 承诺"3 行代码里只动 1 个字" → E05 hook 必须正面呼应（实际上 import 和类名各改一次，共 2 处。但口播仍可保留"换一个零件"的核心画面，明确指出"严格说改两处：import 一处 + 类名一处，但都是同一个变化点"）
- **E05 → E06 留伏笔**：cta 末预告下一站 03-distance "如果欧氏距离不够用怎么办" → e06 hook 接

## 段落骨架（6 段）

| id | type | topic | 源 | 字数 |
|---|---|---|---|---|
| 1 | hook | 接住 E04 "只动一个字"+ 本期主题 | A | ~140 |
| 2 | content | 数据准备：X 不变 / y 从标签换成具体打分 | A | ~250 |
| 3 | content | Jupyter 录屏：找不同游戏 + 跑出 4.36 | B | ~310 |
| 4 | content | 4.36 怎么来的：5 个最近邻平均（手验）| B | ~280 |
| 5 | content | 分类 vs 回归对照表 + 三件套是真正的核心 | A | ~250 |
| 6 | cta | 整章收敛 + e06 距离族预告 | A | ~330 |

合计 ~1560 字 → 估算 ~10 min（口播 ~7 min + demo 录屏 ~3 min）

## 待办

- [ ] script.json 正稿 3-Agent 审稿
- [ ] Jupyter 录屏脚本实跑（已 Python 验证 4.36 + 5 邻居 = (4.5, 4.5, 4.3, 4.0, 4.5)）
- [ ] 与 E04 cta 文本逐字对账，确认衔接钩落实
