#!/usr/bin/env bash
# Install omnivoice-server with the correct PyTorch CUDA variant for this machine.
# Usage: ./scripts/install.sh
set -euo pipefail

# ── Detect CUDA version from driver ──────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+") || true
fi

if [ -z "${CUDA_VER:-}" ]; then
    TORCH_INDEX="cpu"
    echo "→ No NVIDIA GPU detected, installing CPU build"
else
    MAJOR="${CUDA_VER%%.*}"
    MINOR="${CUDA_VER#*.}"; MINOR="${MINOR%%.*}"

    if   [ "$MAJOR" -ge 13 ];                        then TORCH_INDEX="cu130"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 6 ]; then TORCH_INDEX="cu126"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then TORCH_INDEX="cu124"
    elif [ "$MAJOR" -eq 12 ];                        then TORCH_INDEX="cu121"
    elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then TORCH_INDEX="cu118"
    else
        echo "→ CUDA $CUDA_VER too old for a supported PyTorch build, falling back to CPU"
        TORCH_INDEX="cpu"
    fi
    echo "→ Detected CUDA $CUDA_VER → PyTorch $TORCH_INDEX"
fi

# ── Install all non-torch deps ────────────────────────────────────────────────
echo "→ Running uv sync..."
uv sync

# ── Reinstall torch stack from the correct index ─────────────────────────────
if [ "$TORCH_INDEX" != "cpu" ]; then
    PYTORCH_URL="https://download.pytorch.org/whl/$TORCH_INDEX"
    echo "→ Installing torch+$TORCH_INDEX from $PYTORCH_URL"
    uv pip install torch torchaudio torchcodec \
        --index-url "$PYTORCH_URL" \
        --reinstall-package torch \
        --reinstall-package torchaudio \
        --reinstall-package torchcodec
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo "→ Verifying CUDA availability..."
uv run python -c "
import torch
print(f'   torch:          {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU:            {torch.cuda.get_device_name(0)}')
    print(f'   Device count:   {torch.cuda.device_count()}')
"

echo "✓ Install complete"
