# E03 Demo 录屏 · 04-regression

3 个 shot 全部用 `01-intro/demos/04-regression.py`（marimo run 端口 2752）。

## 文件

| 文件 | 规格 | 与 TTS 对齐 |
|---|---|---|
| `shot05.mp4` | 1278×1046 · H.264 · 53.00s · 1.8 MB | ↔ `astral-pipeline/ml03/tts/shot05.wav` (53.014s) · 差 14 ms |
| `shot06.mp4` | 1278×1046 · H.264 · 45.28s · 2.5 MB | ↔ `astral-pipeline/ml03/tts/shot06.wav` (45.314s) · 差 34 ms |
| `shot07.mp4` | 1278×1046 · H.264 · 29.44s · 1.3 MB | ↔ `astral-pipeline/ml03/tts/shot07.wav` (29.453s) · 差 13 ms |

## 时间窗口

### shot05 · 回归基线 + 介绍 viridis（53.0s）

| 时间 | State | 画面 |
|---|---|---|
| 0 - 53.0 | 默认（k=7 / weighted=off / rating=7.5 / lead=8.0） | 全程静帧；口播介绍 viridis 色阶 / MAE / RMSE/R² 留 sklearn |

cue 16.26s "K 取 7" 与 demo 默认值一致，无需操作 slider。

### shot06 · K=1 vs K=80 极端（45.3s）

| 时间 | State | 画面 |
|---|---|---|
| 0 - 3.16 | k=7 静帧 | 起始默认 |
| 3.16 - 16.22 | 拖 k=1（6 步 · 0.18s） | 决策面变斑驳 / MAE 飙升 |
| 16.22 - 45.3 | 拖 k=80（79 步 · ~2.4s） | 决策面单一颜色 ≈ 全局均值 41% / 极端欠拟合 |

### shot07 · 残差图（29.5s）

| 时间 | State | 画面 |
|---|---|---|
| 0 - 6.0 | k=80 静帧（鼠标在角落） | 整图聚焦 |
| 6.0 - 14.0 | 鼠标平滑移到右侧残差散点图中心（cx=1332, cy=614） | 视觉聚焦残差图 |
| 14.0 - 29.5 | 鼠标在残差图内横扫 | 经过若干点（可能触发 tooltip） |

shot07 起始 state（k=80）继承 shot06 末态：新 context 内 SETUP 阶段静默拖到 80。

## Slider DOM（probe 实测）

```
marimo-slider × 3 + marimo-switch × 1
  idx 0 = k        (1-80, step=1, val=7)   ← 任务清单写的 k=0 正确
  idx 1 = rating   (3-11, step=0.1, val=7.5)
  idx 2 = lead     (3-12, step=0.1, val=8)
  switch[0] = weighted（默认 off · 本期 3 shot 全程不开）
```

⚠️ **修正**：任务清单原写 "weighted=1 / rating=2 / lead=3"，实际 `weighted_switch` 是 `mo.ui.switch` 不是 `mo.ui.slider`，所以滑块只有 3 个，索引为 k=0 / rating=1 / lead=2。

## Crop 参数（probe 实测）

```
react-grid-layout bbox: x=321, y=33, w=1278, h=1047, maxWidth=1280
→ crop=1278:1046:321:34（高 1047 → 1046 偶数对齐 H.264）
```

## 下游用法

```bash
# 拼音视频
ffmpeg -i shot05.mp4 -i ../../../../astral-pipeline/ml03/tts/shot05.wav \
       -c:v copy -c:a aac -shortest shot05_with_audio.mp4
ffmpeg -i shot06.mp4 -i ../../../../astral-pipeline/ml03/tts/shot06.wav \
       -c:v copy -c:a aac -shortest shot06_with_audio.mp4
ffmpeg -i shot07.mp4 -i ../../../../astral-pipeline/ml03/tts/shot07.wav \
       -c:v copy -c:a aac -shortest shot07_with_audio.mp4
```

## 推下游（可选）

```bash
cp shot0{5,6,7}.mp4 ~/projects/astral-pipeline/ml03/recording/
touch ~/projects/astral-pipeline/ml03/recording/.ready
```

## 重新生成

```bash
# 1. 起 marimo（如未跑）
cd 01-ML/01-KNN/01-intro/demos
marimo run 04-regression.py --headless --port 2752 --no-token &

# 2. 跑录屏
cd ../../scripts/_recording
python3 e03_shot05.py   # 53s
python3 e03_shot06.py   # 45s
python3 e03_shot07.py   # 30s

# 3. 拷贝（脚本默认输出 output/e03_shotXX_cropped.mp4）
cp output/e03_shot05_cropped.mp4 ../e03-工作流回归/recording/shot05.mp4
# ...
```

## 验证清单

- [x] 视频时长 = TTS 总长（差 < 50ms）
- [x] crop 1278×1046（probe boundingBox）
- [x] viridis 色阶颜色正常（决策面 + 残差图 redyellowblue 互不冲突）
- [x] 滑块拖动可见（step_delay=30ms · k 1→80 拖动约 2.4s）
- [x] 各 state 稳定 ≥ 1.5s
- [x] 末尾静帧（最后一个 set_slider 完成后到 T_END 都是稳定画面）
