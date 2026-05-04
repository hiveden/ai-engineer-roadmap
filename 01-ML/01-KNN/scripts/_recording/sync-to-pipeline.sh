#!/usr/bin/env bash
# sync-to-pipeline.sh — 把本仓 demo 录屏 mp4 推到 astral-pipeline。
#
# 用法：
#   bash sync-to-pipeline.sh <epXX-期名> <pipelineId>
#   bash sync-to-pipeline.sh e01-概念距离 ml01
#   bash sync-to-pipeline.sh --all                  # 按 mapping 全推
#
# 契约（HANDSHAKE 第 4 方提案，待入正文）：
#   ~/projects/astral-pipeline/<id>/recording/
#   ├── shotYY.mp4
#   ├── manifest.json
#   └── .ready
#
# 原子写：先写 .tmp 再 rename，防止下游读到半截文件。
# 重跑幂等：同一 epXX 反复跑只会覆盖同名文件，不会污染。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/01-ML/01-KNN/scripts"
PIPELINE_ROOT="${ASTRAL_PIPELINE_ROOT:-$HOME/projects/astral-pipeline}"

# epXX → mlXX 映射（与 scripts/README.md 12 期索引顺序一致）
declare -a MAPPING=(
  "e01-概念距离:ml01"
  "e02-k值加权:ml02"
  "e03-工作流回归:ml03"
  "e04-分类API:ml04"
  "e05-回归API:ml05"
  "e06-距离族:ml06"
  "e07-缩放动机:ml07"
  "e08-归一化:ml08"
  "e09-标准化高斯:ml09"
  "e10-鸢尾花实战:ml10"
  "e11a-交叉验证:ml11"
  "e11b-网格搜索数字识别:ml12"
)

usage() {
  echo "用法："
  echo "  bash $(basename "$0") <epXX-期名> <pipelineId>"
  echo "  bash $(basename "$0") --all"
  exit 1
}

# ---------------------------------------------------------------
# 同步单期
# ---------------------------------------------------------------
sync_one() {
  local ep="$1"
  local id="$2"
  local src_dir="$SCRIPTS_DIR/$ep/recording"
  local dst_dir="$PIPELINE_ROOT/$id/recording"

  if [[ ! -d "$src_dir" ]]; then
    echo "[SKIP] $ep · 无 recording/ 目录"
    return 0
  fi

  # 收集所有 shot mp4
  local shots=()
  while IFS= read -r f; do
    shots+=("$f")
  done < <(find "$src_dir" -maxdepth 1 -name 'shot*.mp4' -type f | sort)

  if [[ ${#shots[@]} -eq 0 ]]; then
    echo "[SKIP] $ep · recording/ 内无 mp4"
    return 0
  fi

  echo "[SYNC] $ep → $id (${#shots[@]} shots)"

  mkdir -p "$dst_dir"

  # 删除旧 .ready，防止下游在 cp 期间误读
  rm -f "$dst_dir/.ready"

  # 拷 mp4：先 .tmp 再 mv（原子）
  for f in "${shots[@]}"; do
    local name
    name=$(basename "$f")
    local dst="$dst_dir/$name"
    cp "$f" "$dst.tmp"
    mv "$dst.tmp" "$dst"
    echo "  · $name"
  done

  # 写 manifest.json
  build_manifest "$ep" "$id" "$src_dir" "$dst_dir" "${shots[@]}"

  # touch .ready（必须最后）
  touch "$dst_dir/.ready"
  echo "  · .ready"
}

# ---------------------------------------------------------------
# 生成 manifest.json
#   {
#     "episode_id": "ml01",
#     "source_episode": "e01-概念距离",
#     "synced_at": "2026-05-04T...",
#     "shots": {
#       "shot05": {
#         "file": "shot05.mp4",
#         "duration_s": 78.000,
#         "video": {"width": 1198, "height": 600, "fps": 30}
#       }
#     }
#   }
# ---------------------------------------------------------------
build_manifest() {
  local ep="$1"
  local id="$2"
  local src_dir="$3"
  local dst_dir="$4"
  shift 4
  local shots=("$@")

  local manifest="$dst_dir/manifest.json"
  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  {
    printf '{\n'
    printf '  "episode_id": "%s",\n' "$id"
    printf '  "source_episode": "%s",\n' "$ep"
    printf '  "synced_at": "%s",\n' "$now"
    printf '  "shots": {\n'

    local n=${#shots[@]} i=0
    for f in "${shots[@]}"; do
      local name shot dur w h fps
      name=$(basename "$f")
      shot="${name%.mp4}"

      # ffprobe 读 duration / 分辨率 / fps
      dur=$(ffprobe -v error -show_entries format=duration \
                    -of default=noprint_wrappers=1:nokey=1 "$f")
      w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width \
                  -of default=noprint_wrappers=1:nokey=1 "$f")
      h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
                  -of default=noprint_wrappers=1:nokey=1 "$f")
      fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
                    -of default=noprint_wrappers=1:nokey=1 "$f" \
            | awk -F'/' '{ if ($2) printf "%.0f", $1/$2; else print $1 }')

      i=$((i + 1))
      local comma=","
      [[ $i -eq $n ]] && comma=""

      printf '    "%s": {\n' "$shot"
      printf '      "file": "%s",\n' "$name"
      printf '      "duration_s": %.3f,\n' "$dur"
      printf '      "video": {"width": %s, "height": %s, "fps": %s}\n' "$w" "$h" "$fps"
      printf '    }%s\n' "$comma"
    done

    printf '  }\n'
    printf '}\n'
  } > "$manifest.tmp"

  mv "$manifest.tmp" "$manifest"
  echo "  · manifest.json (${#shots[@]} shots)"
}

# ---------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------
main() {
  if [[ $# -eq 0 ]]; then usage; fi

  if [[ "$1" == "--all" ]]; then
    for entry in "${MAPPING[@]}"; do
      ep="${entry%%:*}"
      id="${entry##*:}"
      sync_one "$ep" "$id"
    done
    return
  fi

  if [[ $# -ne 2 ]]; then usage; fi
  sync_one "$1" "$2"
}

main "$@"
echo "[DONE] pipeline root = $PIPELINE_ROOT"
