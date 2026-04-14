# probe_max_gpu_layers.sh

Universal `llama.cpp` GPU-layer probe script to find the highest stable `--n-gpu-layers` for any model.

## What it does
- Runs short inference probes across different `ngl` values.
- Uses binary search to find `max_stable_ngl`.
- Prints a recommended value (`max_stable_ngl - safe_margin`).
- Supports Docker backend and local `llama-cli`.
- Optional JSON summary output.

## File
- Script: `scripts/probe_max_gpu_layers.sh`

## Quick start (Docker)
```bash
cd /path/to/your/repo
chmod +x scripts/probe_max_gpu_layers.sh

scripts/probe_max_gpu_layers.sh \
  --model /hf-cache/models--ORG--MODEL/snapshots/<SNAPSHOT>/model.gguf \
  --backend docker \
  --max auto \
  --hard-max 120 \
  --ctx-size 2048 \
  --n-predict 32 \
  --timeout 180 \
  --safe-margin 1 \
  --json-out /tmp/ngl_probe.json
```

## Quick start (Local llama-cli)
```bash
scripts/probe_max_gpu_layers.sh \
  --backend local \
  --llama-cli /usr/local/bin/llama-cli \
  --model /models/model.gguf \
  --max 80
```

## Important flags
- `--n-predict`: number of generated tokens per probe (larger = slower but more realistic).
- `--ctx-size`: context size used during probes.
- `--fit on|off`: llama.cpp fit behavior (`off` by default for stricter limits).
- `--safe-margin`: subtracts from detected max for daily-use recommendation.
- `--json-out`: writes a compact JSON summary (not full raw logs).

## Typical output
```text
RESULT:
  max_stable_ngl=24
  recommended_ngl=23  (safe_margin=1)
  tested_ctx_size=2048
  tested_n_predict=32
```

## License
MIT. See `scripts/LICENSE_probe_max_gpu_layers_MIT`.
