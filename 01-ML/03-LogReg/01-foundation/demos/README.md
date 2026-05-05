# 01-foundation · Demos

第 1 章数学复习节配套交互 demo。每个独占端口避免冲突。

## 索引

| Demo | 章节 | 端口 | 教学钩 |
|---|---|---|---|
| [`01-sigmoid.py`](./01-sigmoid.py) | 02-sigmoid 函数 | 2731 | 拖 z → S 形曲线 + 概率条 + 导数副曲线；z=10 看「饱和→梯度消失」具象 |
| [`02-mle-coin.py`](./02-mle-coin.py) | 04-极大似然估计 | 2732 | 改观测序列 + 拖 θ → 似然 / log 似然双曲线找峰值；θ=0.1 vs θ*=4/6 反差 |
| [`03-log-underflow.py`](./03-log-underflow.py) | 05-对数函数 | 2733 | 拖 N / P → 直接连乘 vs log 版；N=1000, P=0.5 直接塌成 0，log 版 -693.1 健康 |
| [`04-probability-cards.py`](./04-probability-cards.py) | 03-概率 | 2734 | 蒙特卡洛抽两张全 A → 放回 0.59% / 不放回 0.45% 收敛曲线，独立性反例可视化 |

## 启动

```bash
# 单独跑某个 demo
.venv/bin/marimo edit 01-ML/03-LogReg/01-foundation/demos/01-sigmoid.py --port 2731 --headless --no-token

# 同时跑全部 4 个（不同端口共存）
.venv/bin/marimo edit .../demos/01-sigmoid.py          --port 2731 --headless --no-token &
.venv/bin/marimo edit .../demos/02-mle-coin.py         --port 2732 --headless --no-token &
.venv/bin/marimo edit .../demos/03-log-underflow.py    --port 2733 --headless --no-token &
.venv/bin/marimo edit .../demos/04-probability-cards.py --port 2734 --headless --no-token &
```

## 自测（不开浏览器）

```bash
for f in 01-ML/03-LogReg/01-foundation/demos/0[1-4]-*.py; do
  name=$(basename "$f" .py)
  .venv/bin/marimo export script "$f" -o "/tmp/_${name}_lint.py" \
    && .venv/bin/python "/tmp/_${name}_lint.py" \
    && echo "[$name] ✓"
done
```

## 端口管理

```bash
# 杀指定端口
lsof -iTCP:2731 -sTCP:LISTEN | awk 'NR>1{print $2}' | xargs kill 2>/dev/null

# 杀本章 4 个
for p in 2731 2732 2733 2734; do
  lsof -iTCP:$p -sTCP:LISTEN | awk 'NR>1{print $2}' | xargs kill 2>/dev/null
done
```

## 设计约定

| 项 | 选择 |
|---|---|
| 数据帧 | pandas（venv 未装 polars） |
| 图表 | altair（mark_line / mark_bar / mark_circle）|
| 公式 | `mo.md(r"$$...$$")` 走 marimo 内置 KaTeX |
| 输入 | `mo.ui.slider` / `dropdown` / `radio` / `text` —— 纯 reactive，不用 mo.state / run_button |
| 字符串引号 | 中文场景用 `「」`，避免 Python 3.14 直引号歧义 |
| 布局文件 | 不预生成 `layouts/`，由 Step 5 用户在 marimo edit 里自己拖（Cmd+Shift+L）|
