# Pipeline 推送指南（skill）

> Step 8 子任务 1。把本仓 demo 录屏 mp4 推到下游 `astral-pipeline/<id>/recording/`。
>
> HANDSHAKE 第 4 方契约提案（recording/ 子目录），尚未入 HANDSHAKE 正文。先实跑稳定再走对齐流程。

## 1. 前置 checklist

- [ ] Step 7 预览审核已过（preview.html 6 项验证全 OK）
- [ ] `scripts/eXX-期名/recording/shotYY.mp4` 存在且时长与 TTS shotYY.wav 一致
- [ ] `~/projects/astral-pipeline/<id>/` 目录存在（id 映射见 §3）
- [ ] `ffprobe` 可用（脚本依赖它读时长 / 分辨率 / fps）
- [ ] 本仓 git 状态干净（避免推送时本地有未提交的 mp4 改动混淆）

## 2. 契约（pipeline 侧产物形态）

```
~/projects/astral-pipeline/<id>/
├── script/        ← script-agent-harness 写
├── tts/           ← tts-agent-harness 写
└── recording/     ← 本仓写【新】
    ├── shotYY.mp4
    ├── manifest.json
    └── .ready     ← 全部产物写完后最后 touch
```

**三条硬约束**（与 HANDSHAKE §2.3 一致）：

1. **单向写**：本仓只写 `recording/`，不读 `script/` / `tts/`，不写其它子目录
2. **原子写**：先写 `*.tmp`，`mv` 覆盖目标；防止下游读到半截文件
3. **`.ready` 最后**：`.ready` touch 必须在所有 mp4 + manifest 写完之后

`.ready` 在同步开始时**先删除**，防止下游在拷贝期间误读旧 `.ready` 配新 mp4。

### manifest.json 字段

```json
{
  "episode_id": "ml01",
  "source_episode": "e01-概念距离",
  "synced_at": "2026-05-04T12:34:56Z",
  "shots": {
    "shot05": {
      "file": "shot05.mp4",
      "duration_s": 78.000,
      "video": {"width": 1198, "height": 600, "fps": 30}
    }
  }
}
```

| 字段 | 类型 | 说明 | 来源 |
|---|---|---|---|
| `episode_id` | string | 下游 id（如 `ml01`） | 命令行参数 |
| `source_episode` | string | 本仓目录名（如 `e01-概念距离`） | 命令行参数 |
| `synced_at` | ISO-8601 UTC | 同步时间戳 | `date -u` |
| `shots.<shotId>.file` | string | mp4 文件名 | 实际 cp 的文件 |
| `shots.<shotId>.duration_s` | number (3 dec) | 视频总长（秒） | `ffprobe format=duration` |
| `shots.<shotId>.video.{width,height,fps}` | int | 视频规格 | `ffprobe stream=...` |

下游用途：astral-video 读 manifest 决定哪些 segment 渲染 `<Video>` 而不是动画；duration_s 用于校验与 `tts/durations.json` 对齐。

## 3. id 映射

KNN 12 期 ↔ astral-pipeline 路径：

| 本仓 epXX | pipeline id |
|---|---|
| e01-概念距离 | ml01 |
| e02-k值加权 | ml02 |
| e03-工作流回归 | ml03 |
| e04-分类API | ml04 |
| e05-回归API | ml05 |
| e06-距离族 | ml06 |
| e07-缩放动机 | ml07 |
| e08-归一化 | ml08 |
| e09-标准化高斯 | ml09 |
| e10-鸢尾花实战 | ml10 |
| e11a-交叉验证 | ml11 |
| e11b-网格搜索数字识别 | ml12 |

映射写死在 `scripts/_recording/sync-to-pipeline.sh` 的 `MAPPING` 数组里。新增章（如 `02-LR`）需要扩 mapping，并约定独立 id 段（如 `ml13-ml20` 给 LR）。

> **id 命名注意**：`ml` 前缀不在 astral-video `scaffold-v2.js` 的 SUPPORTED_SERIES 白名单（`flash/brief/show/run/core/report/meta/agui`）。下游 scaffold 时会报错——需 astral-video 扩白名单加 `ml`。本仓 sync 不受影响（不读 series 配置）。

## 4. 标准动作 3 步

```bash
# 单期
bash 01-ML/01-KNN/scripts/_recording/sync-to-pipeline.sh e01-概念距离 ml01

# 全推（按 §3 mapping 顺序遍历，缺 recording/ 的期 SKIP）
bash 01-ML/01-KNN/scripts/_recording/sync-to-pipeline.sh --all
```

脚本行为：

1. 删旧 `.ready`
2. 逐个 `cp shotYY.mp4 → .tmp → mv` 到 `pipeline/<id>/recording/`
3. `ffprobe` 读每个 mp4 → 写 `manifest.json.tmp` → `mv`
4. `touch .ready`

**幂等**：重跑同 epXX 只覆盖同名文件，不污染。

## 5. 验证清单

跑完后到 pipeline 侧验：

```bash
ls ~/projects/astral-pipeline/ml01/recording/
# 期望：shotYY.mp4 一个或多个 + manifest.json + .ready

cat ~/projects/astral-pipeline/ml01/recording/manifest.json | jq .

# 验 mp4 时长 vs tts/durations.json 对齐
python3 -c "
import json
m = json.load(open('$HOME/projects/astral-pipeline/ml01/recording/manifest.json'))
d = {x['id']: x['duration_s'] for x in json.load(open('$HOME/projects/astral-pipeline/ml01/tts/durations.json'))}
for sid, info in m['shots'].items():
    diff = abs(info['duration_s'] - d[sid])
    flag = 'OK' if diff < 0.05 else f'NG diff={diff:.3f}s'
    print(f'  {sid}  mp4={info[\"duration_s\"]:.3f}  wav={d[sid]:.3f}  {flag}')
"
```

时长偏差 < 50 ms 即合格（与 `_4-recording-guide.md` §4 标准一致）。

## 6. 失败模式

| 失败 | 教训 |
|---|---|
| `ffprobe: command not found` | `brew install ffmpeg`（含 ffprobe） |
| 下游读到半截 mp4 | 必须 `.tmp + mv` 原子写；脚本已实现，禁直接 cp 到目标 |
| 下游收到旧 `.ready` 但 mp4 还在拷 | 脚本开头先 `rm -f .ready`，禁省略 |
| pipeline 目录不存在 | 脚本会 `mkdir -p`；但 ml01-ml12 通常已被 tts 创建过 |
| manifest.json 时长与 wav 不对 | 录屏脚本里 T_END 写错——回 `_4-recording-guide.md` 复盘，不是 sync 的问题 |
| `--all` 跳过某期 | 该期 `recording/` 无 mp4，正常 SKIP |
| 推完 astral 端 scaffold 报 series 不识别 | 待 astral-video 扩 SUPPORTED_SERIES 白名单加 `ml`（HANDSHAKE C.4 流程） |

## 7. 推送之后

下游消费链路（不归本仓职责，记录用）：

```
本仓 sync .ready
      ▼
astral-video scaffold-v2.js --id mlXX --from-pipeline
  → symlink pipeline/mlXX/{script,tts}/ 到 episodes/mlXX/
  → 【待补】symlink pipeline/mlXX/recording/ 到 public/mlXX/recording/
  → 【待补】读 manifest.json，给 demo 段 segment 注入 visual.kind = 'recording'
      ▼
npx remotion preview
```

待补部分需要 astral-video 侧改 `scaffold-v2.js` + 加 `RecordingRenderer`。HANDSHAKE 第 4 方契约（recording/）正式入正文后，该改动一并跟进。
