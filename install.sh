#!/usr/bin/env bash
# install.sh — Set up the lhmpp conda environment for LHM++ inference
#
# Usage:
#   bash install.sh            # full install (env + deps + models)
#   bash install.sh --skip-models  # env + deps only (skip weight download)
#   bash install.sh --only-models  # download weights into existing env
#
# Tested: Python 3.10, PyTorch 2.3.0, CUDA 12.1, Linux x86_64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lhmpp"
SKIP_MODELS=false
ONLY_MODELS=false

for arg in "$@"; do
  case $arg in
    --skip-models) SKIP_MODELS=true ;;
    --only-models) ONLY_MODELS=true ;;
  esac
done

log() { echo "[install.sh] $*"; }

# ── 1. Create conda environment ────────────────────────────────────────────────
if ! $ONLY_MODELS; then
  if conda env list | grep -qw "$ENV_NAME"; then
    log "Conda env '$ENV_NAME' already exists — skipping creation."
  else
    log "Creating conda env '$ENV_NAME' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
  fi

  # Activate inside the script via eval (works for bash/zsh)
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"

  # ── 2. PyTorch 2.3.0 + CUDA 12.1 ────────────────────────────────────────────
  log "Installing PyTorch 2.3.0 + CUDA 12.1..."
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

  # ── 3. xformers ──────────────────────────────────────────────────────────────
  log "Installing xformers..."
  pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

  # ── 4. requirements.txt (no strict deps to avoid solver fights) ───────────────
  log "Installing LHM++ requirements..."
  pip install -r "$SCRIPT_DIR/requirements.txt" --no-deps

  # ── 5. Remaining deps that --no-deps skips ───────────────────────────────────
  log "Installing auxiliary dependencies..."
  pip install \
    "rembg[cpu]" \
    lpips plyfile pyrender scipy imageio-ffmpeg \
    "open3d==0.19.0" \
    safetensors fvcore iopath \
    antlr4-python3-runtime==4.9.3 \
    regex filelock rich \
    "tokenizers>=0.19,<0.20" \
    typing_extensions

  # ── 6. spconv ────────────────────────────────────────────────────────────────
  log "Installing spconv-cu121..."
  pip install spconv-cu121

  # ── 7. torch_scatter ─────────────────────────────────────────────────────────
  log "Installing torch_scatter (PyTorch 2.3.0 + CUDA 12.1)..."
  pip install torch_scatter \
    --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html

  # ── 8. PyTorch3D ─────────────────────────────────────────────────────────────
  log "Installing PyTorch3D..."
  pip install --no-build-isolation --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

  # ── 9. diff-gaussian-rasterization ───────────────────────────────────────────
  log "Installing diff-gaussian-rasterization..."
  pip install --no-build-isolation \
    git+https://github.com/ashawkey/diff-gaussian-rasterization/

  # ── 10. simple-knn ───────────────────────────────────────────────────────────
  log "Installing simple-knn..."
  pip install --no-build-isolation \
    git+https://github.com/camenduru/simple-knn/

  # ── 11. pointops (build from source) ─────────────────────────────────────────
  log "Building and installing pointops..."
  pushd "$SCRIPT_DIR/lib/pointops" > /dev/null
  python setup.py install
  popd > /dev/null

  # ── 12. Smoke test ────────────────────────────────────────────────────────────
  log "Running import smoke test..."
  python - <<'EOF'
import torch, xformers, gsplat, spconv, torch_scatter, pytorch3d
import pointops_cuda, diff_gaussian_rasterization, simple_knn
import rembg, accelerate, omegaconf, einops, timm, transformers
print(f"[OK] torch={torch.__version__}, gsplat={gsplat.__version__}, xformers={xformers.__version__}")
EOF

fi  # end of --only-models skip

# ── 13. Download model weights ────────────────────────────────────────────────
if ! $SKIP_MODELS; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  cd "$SCRIPT_DIR"

  log "Downloading prior models (human_model_files, voxel_grid, BiRefNet, ...)..."
  python scripts/download_pretrained_models.py --prior

  log "Downloading LHM++ weights (LHMPP-700M, LHMPPS-700M)..."
  python scripts/download_pretrained_models.py --models

  log "Downloading motion videos (SMPL-X sequences for animation)..."
  python scripts/download_motion_video.py
fi

log "Done! Activate with: conda activate $ENV_NAME"
log "Run test inference: cd $SCRIPT_DIR && python scripts/test/test_app_case.py"
