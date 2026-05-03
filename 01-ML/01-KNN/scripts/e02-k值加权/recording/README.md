# E02 Demo 录屏

源 demo：`01-ML/01-KNN/01-intro/demos/03-k-tuning.py` · 端口 2751
crop：`1400:740:260:34`（实测 grid boundingBox · viewport 1920×1080）

## 文件

| 文件 | 规格 | 与 TTS 对齐 |
|---|---|---|
| `shot05.mp4` | 1400×740 · H.264 · 45.52s · 1.5 MB | ↔ `astral-pipeline/ml02/tts/shot05.wav` (45.539s) |
| `shot06.mp4` | 1400×740 · H.264 · 86.52s · 3.2 MB | ↔ `shot06.wav` (86.525s) |
| `shot07.mp4` | 1400×740 · H.264 · 40.44s · 2.5 MB | ↔ `shot07.wav` (40.448s) |
| `shot09.mp4` | 1400×740 · H.264 · 69.28s · 3.6 MB | ↔ `shot09.wav` (69.291s) |

每个 mp4 时长与 TTS 差异 < 20ms。

## 时间窗口

### shot05 · K=1 过拟合（T_END=45.5s）

| 时间 | State | 画面 |
|---|---|---|
| 0 - 0.82s | 默认（k=11, off） | 6 个噪声点在喜欢区，但 k=11 已淹没噪声 |
| 0.82s - 45.5s | k=1 | 决策边界扭出 6 个红色孤岛，LOOCV ~82% |

### shot06 · K=11 / K=51 平台期（T_END=86.5s）

| 时间 | State | 画面 |
|---|---|---|
| SETUP | k=1（承接 shot05 末态） | — |
| 0 - 1.78s | k=1 | 红色孤岛仍在 |
| 1.78s - 61.5s | k=11 | 孤岛消失，边界平滑，LOOCV ~89% |
| 61.5s - 86.5s | k=51 | 边界更光滑，LOOCV 仍在 89-90% 平台 |

### shot07 · K=121 多数派绑架（T_END=40.4s）

| 时间 | State | 画面 |
|---|---|---|
| SETUP | k=51 | — |
| 0 - 1.72s | k=51 | 边界平滑 |
| 1.72s - 40.4s | k=121 | 决策面几乎全红，LOOCV ~52% |

### shot09 · 加权破远近差异（T_END=69.3s）

| 时间 | State | 画面 |
|---|---|---|
| SETUP | k=121, weighted=off | — |
| 0 - 26.06s | k=121 | 全红（讲加权概念） |
| 26.06s - 28.2s | k=21 | 边界恢复中段平滑 |
| 28.2s - 69.3s | k=21, weighted=on | 边界更紧贴训练点 |

注：cue T_WEIGHTED=27.02s，但 k=121→21 拖动需 ~2.2s，开关实际打开时刻 ~28.2s（晚 1.2s）。视觉上"K=21" 与"加权"几乎连续，不影响理解。

## 下游用法

```bash
ffmpeg -i shot05.mp4 -i ../../../../astral-pipeline/ml02/tts/shot05.wav \
       -c:v copy -c:a aac -shortest shot05_with_audio.mp4
```

## 重新生成

```bash
# 1. 启动 demo（端口 2751，03-k-tuning 专用）
cd 01-ML/01-KNN/01-intro/demos
marimo run 03-k-tuning.py --headless --port 2751 --no-token &

# 2. 跑 4 个录屏脚本
cd ../../scripts/_recording
python3 e02_shot05.py
python3 e02_shot06.py
python3 e02_shot07.py
python3 e02_shot09.py

# 3. 拷产物（脚本默认输出在 _recording/output/）
cp output/e02_shot05_cropped.mp4 ../e02-k值加权/recording/shot05.mp4
# ... shot06/07/09 同
```

## DOM 索引备忘

03-k-tuning 控件结构（**与 02-proximity 不同**）：

- `marimo-slider` 共 3 个：
  - `[0]` k 邻居数（step=2, range=[1,125]）
  - `[1]` 新电影评分（step=0.1, range=[3.0,11.5]）
  - `[2]` 主演吸引度（step=0.1, range=[3.0,12.5]）
- `marimo-switch` 共 1 个：
  - `[0]` 距离加权 — 用 `marimo-switch >> button[aria-checked]` 切换

注意：因为有独立的 `<marimo-switch>` 组件，slider 索引比"按 hstack 顺序数"少 1 位（switch 不算在 marimo-slider 集合里）。
