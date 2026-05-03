# Demo 录屏 Pipeline

> Playwright 自动驱动 marimo demo + 录屏 → ffmpeg 与 TTS 对齐

## 工作流（方案 D · 粗对齐 + 静帧拉伸）

```
1. 启动 marimo run（端口 2719，run 模式干净 UI）
2. Playwright 1920×1080 viewport 打开 demo
3. 按 states 清单顺序操作（拖滑块 + 等渲染稳定）
4. 录屏输出 webm
5. ffmpeg 转 mp4 + 与 TTS shotXX.wav 对齐
6. 推送到 astral-pipeline/mlXX/recording/
```

## 文件

| 文件 | 用途 |
|---|---|
| `e01_shot05.py` | E01 shot05 录屏（02-proximity，4 states） |
| `e02_shotXX.py` | 后续 |
| `output/` | 录屏产物（.gitignore） |

## 跑

```bash
# 1. 启动 marimo run（独立端口）
cd 01-ML/01-KNN/01-intro/demos
marimo run 02-proximity.py --headless --port 2719 --no-token &

# 2. 跑录屏
cd 01-ML/01-KNN/scripts/_recording
python3 e01_shot05.py

# 3. 输出在 output/e01_shot05.webm
```

## 设计原则

- **viewport 1920×1080**（1080p 标准）
- **保持 marimo grid 布局原样**（maxWidth=1200 居中，左右留白，不动用户手调比例）
- **拖滑块真实速度**（不是瞬移，让观众看到过程）
- **每 state 稳定后停 1.5s**（给 ffmpeg 拉伸缓冲）
