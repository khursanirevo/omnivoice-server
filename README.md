# omnivoice-server

OpenAI-compatible HTTP server for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS.

## Setup

```bash
git clone https://github.com/maemreyo/omnivoice-server.git
cd omnivoice-server
bash scripts/install.sh        # auto-detects GPU driver, installs matching PyTorch
uv run omnivoice-server        # downloads model (~3GB) on first run
```

Server starts at `http://127.0.0.1:8880`. To use a custom or fine-tuned checkpoint:

```bash
uv run omnivoice-server --model your-org/your-model
```

## API

### Generate speech

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "omnivoice", "input": "Hello world.", "voice": "alloy"}' \
  --output speech.wav
```

### Voice presets

`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`

### Custom voice via instructions

```json
{ "input": "Hello.", "instructions": "female, british accent, young adult" }
```

Attributes: gender, age (child/teenager/young adult/middle-aged/elderly), pitch (very low/low/moderate/high/very high), accent (american/british/australian/indian/...), `whisper`.

### Voice cloning

Save a speaker profile (reusable, embedding cached to disk and survives restarts):

```bash
curl -X POST http://localhost:8880/v1/voices/profiles \
  -F "profile_id=my_speaker" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Exact transcript of the reference audio"   # required
```

Use it:

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "omnivoice", "input": "Hello.", "voice": "my_speaker"}' \
  --output out.wav
```

One-shot clone (no profile saved):

```bash
curl -X POST http://localhost:8880/v1/audio/speech/clone \
  -F "text=Hello." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Exact transcript of the reference audio"   # required
  --output out.wav
```

> `ref_text` must be the exact transcript of the reference audio. It is used to condition the speaker embedding and significantly affects clone quality.

### Streaming

```json
{ "input": "Long text...", "stream": true }
```

Returns chunked PCM (24kHz, 16-bit, mono). Set `"position_temperature": 0` for consistent voice across chunks.

## Configuration

All settings via env vars (`OMNIVOICE_` prefix) or CLI flags.

| Env Var | Default | Description |
|---------|---------|-------------|
| `OMNIVOICE_HOST` | `127.0.0.1` | Bind host |
| `OMNIVOICE_PORT` | `8880` | Port |
| `OMNIVOICE_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `OMNIVOICE_MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace repo or local path |
| `OMNIVOICE_NUM_STEP` | `16` | Diffusion steps (8–32) |
| `OMNIVOICE_MAX_CONCURRENT` | `2` | Parallel inference slots |
| `OMNIVOICE_API_KEY` | `""` | Bearer token (empty = no auth) |
| `OMNIVOICE_PROFILE_DIR` | `~/.local/share/omnivoice/profiles` | Speaker profiles directory |
| `OMNIVOICE_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `OMNIVOICE_COMPILE_MODE` | `none` | `none`, `default`, `reduce-overhead`, `max-autotune` |
| `OMNIVOICE_COMPILE_CACHE_DIR` | `null` | Persist compiled kernels across restarts |
| `OMNIVOICE_QUANTIZATION` | `none` | `fp8wo`, `fp8dq`, `int8wo`, `int8dq` |
| `OMNIVOICE_RESPONSE_CACHE_ENABLED` | `true` | Cache repeated identical requests |
| `OMNIVOICE_RESPONSE_CACHE_MAX_GB` | `5.0` | Max disk for response cache |

## Performance

Benchmarked at `num_step=16`:

| Hardware | Latency | RTF |
|----------|---------|-----|
| RTX 4090 (no compile) | ~500ms | ~0.1 |
| RTX 4090 + `max-autotune` | ~99ms | ~0.02 |
| CPU (Xeon) | ~4.5s | ~0.9 |

For production: `--compile-mode max-autotune` compiles Triton kernels on first boot (~2 min). Persist them with `--compile-cache-dir ./torch_compile_cache` so subsequent restarts load instantly.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/speech` | TTS with presets, instructions, or saved profile |
| `POST` | `/v1/audio/speech/clone` | One-shot voice clone |
| `GET` | `/v1/voices` | List voices and profiles |
| `POST` | `/v1/voices/profiles` | Save speaker profile |
| `GET` | `/v1/voices/profiles/{id}` | Get profile |
| `PATCH` | `/v1/voices/profiles/{id}` | Update profile audio or transcript |
| `DELETE` | `/v1/voices/profiles/{id}` | Delete profile |
| `GET` | `/live` | Liveness (always 200) |
| `GET` | `/ready` | Readiness (200 when model loaded, 503 during startup) |
| `GET` | `/health` | Full status JSON |
| `GET` | `/metrics/prometheus` | Prometheus metrics |

## Development

```bash
bash scripts/install.sh     # install runtime deps
uv sync --extra dev         # add dev deps (pytest, ruff, mypy)
uv run pytest
uv run ruff check omnivoice_server/
uv run mypy omnivoice_server/
```

## License

MIT — built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by k2-fsa.
