# OmniVoice Server — Deployment Guide

OpenAI-compatible TTS server wrapping [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Setup

```bash
git clone https://github.com/khursanirevo/omnivoice-server.git
cd omnivoice-server
bash scripts/install.sh
uv run omnivoice-server --model Revolab/omnivoice
```

`scripts/install.sh` detects your GPU driver and installs the matching PyTorch CUDA variant automatically. Model downloads from HuggingFace on first run (~3GB).

## Quick Reference

| Item | Value |
|------|-------|
| Default port | 8880 |
| Health | `GET /live` · `GET /ready` · `GET /health` |
| Metrics | `GET /metrics/prometheus` |
| Auth | `OMNIVOICE_API_KEY` env var (empty = no auth) |
| Web UI | `http://HOST:8880/` |

## Usage

### Generate speech

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "omnivoice", "input": "Hello.", "voice": "alloy"}' \
  --output speech.wav
```

### Add a speaker profile

```bash
curl -X POST http://localhost:8880/v1/voices/profiles \
  -F "profile_id=anwar" \
  -F "ref_audio=@anwar.wav" \
  -F "ref_text=Exact transcript of the audio"
```

Then use it with `"voice": "anwar"` — embedding is cached to disk and survives restarts.

### Voice cloning (one-shot)

```bash
curl -X POST http://localhost:8880/v1/audio/speech/clone \
  -F "text=Hello." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Exact transcript" \
  --output out.wav
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `OMNIVOICE_HOST` | `127.0.0.1` | Bind host |
| `OMNIVOICE_PORT` | `8880` | Port |
| `OMNIVOICE_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `OMNIVOICE_MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace repo or local path |
| `OMNIVOICE_NUM_STEP` | `16` | Diffusion steps (8–32) |
| `OMNIVOICE_MAX_CONCURRENT` | `2` | Parallel inference slots |
| `OMNIVOICE_API_KEY` | `""` | Bearer token (empty = no auth) |
| `OMNIVOICE_COMPILE_MODE` | `none` | `max-autotune` for best GPU perf |
| `OMNIVOICE_COMPILE_CACHE_DIR` | `null` | Persist compiled kernels across restarts |
| `OMNIVOICE_QUANTIZATION` | `none` | `int8wo`, `int8dq`, `fp8wo`, `fp8dq` |
| `OMNIVOICE_RESPONSE_CACHE_MAX_GB` | `5.0` | Response cache size |
| `OMNIVOICE_LOG_LEVEL` | `info` | Log verbosity |

## Performance

Benchmarked at `num_step=16` on this machine (RTX 4090):

| Config | Latency | RTF |
|--------|---------|-----|
| CUDA (no compile) | ~500ms | ~0.1 |
| CUDA + `max-autotune` | ~99ms | ~0.02 |
| CPU | ~4.5s | ~0.9 |

For production: add `--compile-mode max-autotune --compile-cache-dir ./torch_compile_cache`. First boot compiles kernels (~2 min), all subsequent starts load from cache.

## Systemd

```bash
sudo cp omnivoice-server.service /etc/systemd/system/
# Edit User, Group, and ExecStart path in the service file to match your install
sudo systemctl daemon-reload
sudo systemctl enable --now omnivoice-server
sudo journalctl -u omnivoice-server -f
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA not detected | Run `bash scripts/install.sh` — re-detects and reinstalls correct torch variant |
| CUDA OOM | Lower `--max-concurrent` or use `--device cpu` |
| First request slow | Kernel compilation on first run. Use `--compile-cache-dir` to persist |
| 503 Queue Full | Raise `--max-queue-depth` or add inference slots |
| Auth failing | Check `OMNIVOICE_API_KEY` matches `Authorization: Bearer <key>` |
