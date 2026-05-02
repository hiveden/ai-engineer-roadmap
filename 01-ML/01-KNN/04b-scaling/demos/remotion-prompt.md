# Remotion 实验 · KNN 缩放动画提示词

> 用法：把下面"提示词"整段（从 H2 "提示词正文" 到末尾）复制，
> 粘贴到 **astral-video 项目** 里的 Claude Code 对话框，让它在 `src/experiments/knn-scaling/` 生成动画。

---

## 提示词正文

```
在 src/experiments/knn-scaling/ 新建 Remotion 动画实验。
不要 import @v2/engine/*，不写 TTS / Audio / Captions。
参考已有 src/experiments/knn-demo/ 的代码风格。

# 目标
30 秒竖屏短片（1080×2060, 30fps），演示"特征缩放对 KNN 的影响"，分 4 段：

## Shot 1 · 体重独裁（0-7.5s, frames 0-225）
- 散点图（x=身高, y=体重）显示 8 个样本 + 蓝色星号代表新人
- 体重从 80 平滑动画到 120（interpolate clamp）
- 顶部 callout：🟢 健康 → 🔴 不健康（在 weight=95 翻转）
- scaler = "None"
- 字幕：「KNN 距离没缩放 → 体重独裁」

## Shot 2 · 切到 MinMaxScaler（7.5-15s, frames 225-450）
- 散点图坐标系平滑过渡到 [0,1] 缩放后位置（用 interpolate 给整个 transform）
- 同样体重 80→120 动画，但 callout 不再翻转
- 右上角 ScalerParamsCard 显示 min/max 表
- 字幕：「MinMax 后 → 三特征公平投票」

## Shot 3 · 异常值崩塌（15-22.5s, frames 450-675）
- 紫色异常值点 (170, 250, 1.0) 从顶部带 spring 弹跳落入数据
- 散点图 x 轴自动扩展到 250
- 8 个正常点视觉上被压扁到左下角 [0, 0.243] 区域
- callout 闪红：MinMax 失效
- 字幕：「一个异常值 → 正常数据被压扁」

## Shot 4 · StandardScaler 救场（22.5-30s, frames 675-900）
- 切到 StandardScaler，散点图重新展开
- 8 个正常点恢复合理分布（z 值在 [-0.6, 0.6]）
- 异常值在右上角 z≈2.55 位置
- callout 稳定：✅ 鲁棒
- 字幕：「StandardScaler → μ/σ 鲁棒」

# 文件结构
src/experiments/knn-scaling/
├── KnnScaling.tsx              主 composition
├── data.ts                     8 行数据 + KNN/缩放纯 JS 函数
├── timeline.ts                 4 段 frame 范围常量
└── components/
    ├── ScatterChart.tsx        散点图（响应 scaler 切换 + transform interpolate）
    ├── PredictionBadge.tsx     顶部 callout（绿/红色块 + 邻居票数）
    └── ScalerParamsCard.tsx    右上角小卡片（显示 min/max 或 μ/σ）

# 数据（写在 data.ts）

## 8 行健康预测样本
const SAMPLES = [
  {h: 175, w: 70,  v: 1.5, label: 1},  // s1 健康
  {h: 180, w: 78,  v: 1.2, label: 1},  // s2
  {h: 170, w: 65,  v: 1.8, label: 1},  // s3
  {h: 168, w: 72,  v: 1.5, label: 1},  // s4
  {h: 165, w: 95,  v: 0.4, label: 2},  // s5 不健康
  {h: 172, w: 100, v: 0.3, label: 2},  // s6
  {h: 178, w: 110, v: 0.5, label: 2},  // s7
  {h: 167, w: 92,  v: 0.6, label: 2},  // s8
];

## 异常值（Shot 3 注入）
const OUTLIER = {h: 170, w: 250, v: 1.0, label: 2};

## 新人查询点
const QUERY_BASE = {h: 173, v: 1.0};
const QUERY_WEIGHT_RANGE = [80, 120];

# 算法（实现在 data.ts）

```typescript
export function minMaxScale(rows: Sample[]): {
  mins: [number, number, number];
  maxs: [number, number, number];
  scale: (p: [number, number, number]) => [number, number, number];
}

export function standardScale(rows: Sample[]): {
  means: [number, number, number];
  stds: [number, number, number];
  scale: (p: [number, number, number]) => [number, number, number];
}

export function predictKNN(
  query: [number, number, number],
  k: number = 3,
  scaler: 'None' | 'MinMax' | 'Standard',
  data: Sample[]
): {
  prediction: 1 | 2;
  neighbors: Array<{idx: number; dist: number; label: 1 | 2}>;
  healthy: number;
  unhealthy: number;
}
```

# 缩放参数（用于 ScalerParamsCard 显示）

## MinMaxScaler · 干净数据
| 特征 | min | max | 跨度 |
|---|---|---|---|
| h | 165 | 180 | 15 |
| w | 65  | 110 | 45 |
| v | 0.3 | 1.8 | 1.5 |

## MinMaxScaler · 含异常值（Shot 3）
| 特征 | min | max | 跨度 |
|---|---|---|---|
| h | 165 | 180 | 15 |
| w | 65  | 250 | 185 |  ← 被异常值拉爆
| v | 0.3 | 1.8 | 1.5 |

## StandardScaler · 干净数据
| 特征 | μ | σ |
|---|---|---|
| h | 171.875 | 5.04 |
| w | 84.0    | 15.62 |
| v | 0.975   | 0.572 |

# 预计算翻转点（关键帧 verify）

## Shot 1 · scaler=None
| weight | 健康票 | 不健康票 | 预测 |
|---|---|---|---|
| 80  | 3 | 0 | 1 健康 ✅ |
| 90  | 2 | 1 | 1 健康 |
| **95**  | **1** | **2** | **2 不健康 ❌（翻转点）** |
| 100 | 0 | 3 | 2 不健康 |
| 120 | 0 | 3 | 2 不健康 |

## Shot 2 · scaler=MinMax（干净）
- 翻转更"延迟"且不剧烈（缩放后视力/身高也参与判断）

## Shot 3 · 异常值 + MinMax
- 8 个正常样本 w' 全部落在 [0, 0.243]
- 新人 w=80 → w'=(80-65)/185≈0.081
- 异常值 w'=1.0
- KNN 行为不可预期

## Shot 4 · 异常值 + Standard
- μ≈102, σ≈58（受异常值影响有限）
- 8 个正常样本 z 值在 [-0.6, 0.6]
- 异常值 z=(250-102)/58≈2.55
- KNN 前 3 邻居仍是正常样本 → 鲁棒

# 颜色规范
const COLORS = {
  healthy: '#2ca02c',     // 绿
  unhealthy: '#d62728',   // 红
  query: '#1f77b4',       // 蓝（新人星号）
  outlier: '#9467bd',     // 紫（异常值，Shot 3）
  bg: '#fafafa',
  axis: '#333',
  grid: '#e0e0e0',
};

# 字幕文案
const CAPTIONS = {
  shot1: 'KNN 距离没缩放 → 体重独裁',
  shot2: 'MinMax 后 → 三特征公平投票',
  shot3: '一个异常值 → 正常数据被压扁',
  shot4: 'StandardScaler → μ/σ 鲁棒',
};

# 视觉规范
- 竖屏 1080×2060, 30fps
- 字体节奏参考 src/experiments/knn-demo/KnnDemo.tsx
- interpolate 必须 'clamp'，关键节点（异常值落入、callout 翻转）用 spring()
- callout 颜色用 interpolateColors 平滑过渡，不要硬切
- 散点图坐标系切换给 1.5s 过渡时间
- **禁止 AI slop**：紫粉渐变 / 玻璃拟态 / 多色高亮 / 3D 旋转
- 坚持扁平 + 单色块 + 高对比度

# 注册
最后改 src/experiments/index.tsx，加：
<Composition
  id="exp-knn-scaling"
  component={KnnScaling}
  durationInFrames={900}
  fps={30}
  width={1080}
  height={2060}
/>

# 验证
完成后：
1. 跑 npm run check（tsc + ESLint 必须通过）
2. 告诉我两个命令：
   - 在 Studio 里预览：npx remotion studio
   - 直接渲染：npx remotion render src/index.ts exp-knn-scaling output/knn-scaling.mp4 --codec h264

# 类型严格
所有 props / state 用 TypeScript interface，不用 any。
```

---

## 跑完后的迭代清单

第一版可能糙的地方，预先列好待改项：

| 问题 | 修复方向 |
|---|---|
| Shot 切换硬切 | 在 KnnScaling.tsx 主 composition 加 1s 过渡（fadeOut + fadeIn） |
| callout 翻转突兀 | `interpolateColors([healthy_color, unhealthy_color])` 在翻转前后 30 帧渐变 |
| 散点图坐标轴跳变 | 给整个 `<g>` 的 viewBox / scale 用 interpolate |
| 异常值落入路径 | spring 加 damping 5 / stiffness 100 弹性 |
| 距离表行重排不流畅 | 每行用 absolute 定位 + interpolate 移动 y |
| 字幕出现/消失 | 渐入 + 翻译 + 渐出，每段 0.3s |

---

## 实验完后给我看什么

跑出 `output/knn-scaling.mp4` 后，把视频拖给我看（或截 4 个关键帧），我帮你判断：

1. 视觉品味（是不是踩了 AI slop 红线）
2. 节奏（4 段时间分配是否合理）
3. 信息密度（每秒传达了几个 idea）
4. 与 Marimo demo 的差异化（哪些是视频独有的优势）

---

## 备忘 · 这个实验是为了什么

1. **试水 Remotion 在 ML 教学上的天花板** — 一集需要多少时间，效果多炸裂？
2. **决定要不要规模化** — 每个算法柱子都做，还是只做封面？
3. **不替代 Marimo** — Marimo 继续做"互动 sandbox"，Remotion 只做"被动观看视频"
