# 04b-scaling · 互动演示

> Marimo 反应式 Notebook · 本地跑、git 友好、纯 .py

## 安装（一次性）

```bash
pip install marimo scikit-learn matplotlib numpy
```

可选：matplotlib 中文字体（避免散点图标签乱码）

```bash
# Mac 用系统字体
echo 'matplotlib.rcParams["font.family"] = "Heiti TC"' >> ~/.matplotlibrc
```

## 跑

```bash
cd demos
marimo edit 02-knn-scaling.py
```

→ 浏览器打开 `http://localhost:2718`，实时互动。

**关闭**：终端 Ctrl+C。

## 录屏

```bash
# QuickTime 文件 → 新建屏幕录制 → 选浏览器窗口
# 或 OBS：添加来源 → 窗口捕获 → 浏览器
```

## 当 app 跑（只读演示模式）

```bash
marimo run 02-knn-scaling.py
```

→ 隐藏代码，只显示 UI，适合给别人演示。

## 清单

| 文件 | 内容 |
|---|---|
| `02-knn-scaling.py` | KNN + 缩放对比（k / scaler / 异常值 / 新人特征） |

## 已知优化点

- 控件区是 `hstack([vstack(...), vstack(...)])` 嵌套结构，违反 `_2-demo-guide.md` §1。当前 layouts 是按这个 cell 数（7）调好的，重构会让 grid 配置失效。等下一轮 demo 调整时一并改：4 控件展平到一行 hstack widths=[1,1,1,1]，重生成 layouts。

## 玩法（详见 demo 文末"玩法建议"）

1. 基线 → 默认设置
2. 量纲问题 → 调新人体重，看预测被独裁
3. 缩放对比 → 切 MinMax / Standard，看邻居距离公平化
4. 异常值杀手锏 → 开异常 + MinMax，看正常数据被压扁
5. k 值 → 1 vs 7 的差异
