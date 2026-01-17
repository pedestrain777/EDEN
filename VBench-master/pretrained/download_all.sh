#!/usr/bin/env bash
set -euo pipefail

# Bulk download helper for VBench evaluation weights.
#
# - Prefers the existing download.sh scripts when available (amt_model, raft_model, ...).
# - Supplements the remaining models whose directories only contain model_path.txt URLs.
# - Uses CACHE_ROOT / CACHE_VBENCH env vars when provided, otherwise defaults to ~/.cache and ~/.cache/vbench.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
CACHE_VBENCH="${CACHE_VBENCH:-$CACHE_ROOT/vbench}"

log() {
  printf '[download_all] %s\n' "$*" >&2
}

download_file() {
  local url="$1"
  local dest_dir="$2"
  local dest_name="${3:-}"

  mkdir -p "$dest_dir"

  if [[ -z "$dest_name" ]]; then
    dest_name="$(basename "${url%%\?*}")"
  fi

  local dest_path="${dest_dir%/}/$dest_name"

  if [[ -f "$dest_path" ]]; then
    log "skip (exists): $dest_path"
    return
  fi

  log "wget $url -> $dest_path"
  wget -c "$url" -O "$dest_path"
}

# 1) Directories that already provide download.sh
declare -a SCRIPT_DIRS=(
  "amt_model"
  "raft_model"
)

for subdir in "${SCRIPT_DIRS[@]}"; do
  if [[ -f "$PROJECT_DIR/$subdir/download.sh" ]]; then
    log "running $subdir/download.sh"
    (cd "$PROJECT_DIR/$subdir" && bash download.sh)
  else
    log "skip $subdir (no download.sh)"
  fi
done

# 2) Manual URLs specified via model_path.txt (or known requirements)

# Aesthetic score model
download_file \
  "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth" \
  "$CACHE_ROOT/aesthetic_model/emb_reader" \
  "sa_0_4_vit_l_14_linear.pth"

# Caption / Tag2Text model
download_file \
  "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
  "$CACHE_VBENCH/caption_model"

# CLIP checkpoints
download_file \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
  "$CACHE_VBENCH/clip_model"

download_file \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
  "$CACHE_VBENCH/clip_model"

# GRiT DenseCap detector
download_file \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth" \
  "$CACHE_VBENCH/grit_model"

# PIQA (MUSIQ) quality model
download_file \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
  "$CACHE_VBENCH/pyiqa_model"

# UMT model
download_file \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
  "$CACHE_VBENCH/umt_model"

# ViCLIP model
download_file \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
  "$CACHE_VBENCH/ViCLIP"

log "all downloads attempted."

