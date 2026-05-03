# E01 Demo 录屏

> KNN 第 1 集（概念距离）的 demo 段录屏产物。与 TTS 一对一对齐，下游可直接消费。

## 文件

| 文件 | 规格 | 与 TTS 对齐 |
|---|---|---|
| `shot05.mp4` | 1198×600 · H.264 · 78.000s · ~2.4 MB | ↔ `astral-pipeline/ml01/tts/shot05.wav` 同长度 |

## 视频规格

- 分辨率 **1198×600**（marimo grid 实际内容区，已裁掉左右上下留白）
- 帧率 30fps · 编码 H.264 CRF 18 · 静音（音频在 TTS 侧）
- 总长 **78.000 秒**，与 `shot05.wav` 精确同步

## 时间窗口对齐

按 TTS shot05 的 cue 词时间戳分段：

| 时间窗口 | State | 画面 |
|---|---|---|
| 0.00 - 39.23s | **intro** | 默认 (8.5, 9.5, 3) 静帧，配音介绍 UI |
| 39.23 - 55.87s | **baseline** | 拖到 (8.5, 9.5, 5)，5/0 全喜欢 |
| 55.87 - 68.61s | **dislike** | 拖到 (2.9, 3.5, 5)，4/1 不喜欢 |
| 68.61 - 78.00s | **single** | k=1，只剩 1 条黄虚线指向上海堡垒 |

cue 词触发时间戳来自 `astral-pipeline/ml01/tts/subtitles.json` shot05 数组。

## 下游 astral-video 怎么用

下游接收方直接拿 `shot05.mp4` + `shot05.wav` 合成有声视频段：

```bash
ffmpeg -i shot05.mp4 -i ../../../astral-pipeline/ml01/tts/shot05.wav \
       -c:v copy -c:a aac -shortest \
       shot05_with_audio.mp4
```

或在 astral-video 项目内：

```js
// 把视频 + 音频塞进 Remotion / 时间轴
{
  type: "video",
  src: "ml01/recording/shot05.mp4",  // 来自这个仓库
  audio: "ml01/tts/shot05.wav",       // 来自 astral-pipeline
  start: <整集累积时间戳>,
  duration: 78.0,
}
```

## 推到 astral-pipeline（如需）

按 HANDSHAKE 约定可以推到下游 pipeline 目录与 tts/ 平行：

```bash
mkdir -p ~/projects/astral-pipeline/ml01/recording
cp shot05.mp4 ~/projects/astral-pipeline/ml01/recording/
touch ~/projects/astral-pipeline/ml01/recording/.ready
```

## 重新生成

如果脚本/数据有变需重录：

```bash
# 1. 启动 marimo demo
cd 01-ML/01-KNN/01-intro/demos
marimo run 02-proximity.py --headless --port 2750 --no-token &

# 2. 跑录屏脚本
cd ../../scripts/_recording
python3 e01_shot05.py
# → 输出 output/e01_shot05_cropped.mp4

# 3. 拷回这个目录
cp output/e01_shot05_cropped.mp4 ../e01-概念距离/recording/shot05.mp4
```

录屏 pipeline 详见 [`scripts/_recording/README.md`](../../_recording/README.md)。

## 预览

`scripts/_recording/output/preview.html` 提供视频 + TTS 同步预览（双轨 + 字幕浮层）。
