#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  probe_max_gpu_layers.sh --model <path> [options]

Universal GPU-layer probe for llama.cpp models.
Works with:
  1) Docker backend (default): runs llama-cli inside a container
  2) Local backend: runs host llama-cli directly

Model path behavior (Docker backend):
  - If --model points to an existing host file, it is bind-mounted and used.
  - Otherwise it is treated as an in-container model path (e.g. /hf-cache/...).

Options:
  --model PATH              Model file path (required)
  --backend MODE            docker|local (default: docker)
  --llama-cli PATH          Host llama-cli path for local backend (default: llama-cli)
  --image IMAGE             Docker image (default: local/llama-cuda:latest)
  --hf-cache DIR            Host HF cache dir to mount to /hf-cache in Docker
                            (default: $HOME/.cache/huggingface/hub)
  --build                   Run "docker compose build llama" before probing
  --workdir DIR             Directory for docker compose build (default: current dir)

  --min N                   Minimum ngl to test (default: 0)
  --max N|auto              Maximum ngl bound (default: auto)
  --hard-max N              Upper cap when --max=auto (default: 512)
  --ctx-size N              Context size for probes (default: 1024)
  --n-predict N             Tokens to generate per probe (default: 8)
  --prompt TEXT             Probe prompt (default: "Reply with: ok")
  --timeout SEC             Per-probe timeout in seconds (default: 180)
  --retries N               Retries per ngl on failure (default: 1)
  --safe-margin N           Recommended value = max_stable - margin (default: 1)
  --fit MODE                on|off for llama.cpp fit behavior (default: off)
  --extra-arg ARG           Extra arg for llama-cli (repeatable)
  --json-out PATH           Write JSON summary to file
  --verbose                 Print full probe logs
  -h, --help                Show this help

Examples:
  probe_max_gpu_layers.sh \
    --model /hf-cache/models--Qwen--Qwen3-Coder-Next-GGUF/.../Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf \
    --max auto --hard-max 120 --ctx-size 1024 --n-predict 8

  probe_max_gpu_layers.sh \
    --backend local \
    --llama-cli /usr/local/bin/llama-cli \
    --model /models/foo.gguf \
    --max 80
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

MODEL=""
BACKEND="docker"
LLAMA_CLI="llama-cli"
IMAGE="local/llama-cuda:latest"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
DO_BUILD=0
WORKDIR="$(pwd)"

MIN_NGL=0
MAX_NGL="auto"
HARD_MAX=512
CTX_SIZE=1024
N_PREDICT=8
PROMPT="Reply with: ok"
TIMEOUT_SEC=180
RETRIES=1
SAFE_MARGIN=1
FIT_MODE="off"
JSON_OUT=""
VERBOSE=0
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --llama-cli) LLAMA_CLI="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --hf-cache) HF_CACHE_DIR="${2:-}"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --workdir) WORKDIR="${2:-}"; shift 2 ;;
    --min) MIN_NGL="${2:-}"; shift 2 ;;
    --max) MAX_NGL="${2:-}"; shift 2 ;;
    --hard-max) HARD_MAX="${2:-}"; shift 2 ;;
    --ctx-size) CTX_SIZE="${2:-}"; shift 2 ;;
    --n-predict) N_PREDICT="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --retries) RETRIES="${2:-}"; shift 2 ;;
    --safe-margin) SAFE_MARGIN="${2:-}"; shift 2 ;;
    --fit) FIT_MODE="${2:-}"; shift 2 ;;
    --extra-arg) EXTRA_ARGS+=("${2:-}"); shift 2 ;;
    --json-out) JSON_OUT="${2:-}"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[ -n "$MODEL" ] || die "--model is required"
[[ "$BACKEND" == "docker" || "$BACKEND" == "local" ]] || die "--backend must be docker|local"
is_int "$MIN_NGL" || die "--min must be integer"
is_int "$HARD_MAX" || die "--hard-max must be integer"
is_int "$CTX_SIZE" || die "--ctx-size must be integer"
is_int "$N_PREDICT" || die "--n-predict must be integer"
is_int "$TIMEOUT_SEC" || die "--timeout must be integer"
is_int "$RETRIES" || die "--retries must be integer"
is_int "$SAFE_MARGIN" || die "--safe-margin must be integer"
[[ "$FIT_MODE" == "on" || "$FIT_MODE" == "off" ]] || die "--fit must be on|off"

if [ "$MAX_NGL" != "auto" ]; then
  is_int "$MAX_NGL" || die "--max must be integer or auto"
fi

[ "$MIN_NGL" -le "$HARD_MAX" ] || die "--min must be <= --hard-max"

if [ "$DO_BUILD" -eq 1 ]; then
  [ "$BACKEND" = "docker" ] || die "--build is only valid with --backend docker"
  echo "Building Docker image..."
  (cd "$WORKDIR" && docker compose build llama >/dev/null)
fi

TMPDIR_PATH="$(mktemp -d)"
cleanup() {
  rm -rf "$TMPDIR_PATH"
}
trap cleanup EXIT

declare -A STATUS
declare -A REASON

run_probe_once() {
  local ngl="$1"
  local out_file="$2"
  local rc=0

  if [ "$BACKEND" = "docker" ]; then
    local model_in_container="$MODEL"
    local -a docker_args
    docker_args=(docker run --rm --gpus all --entrypoint llama-cli)

    if [ -f "$MODEL" ]; then
      local host_model
      host_model="$(readlink -f "$MODEL")"
      local host_model_dir
      host_model_dir="$(dirname "$host_model")"
      local host_model_base
      host_model_base="$(basename "$host_model")"
      docker_args+=(-v "$host_model_dir:/model-input:ro")
      model_in_container="/model-input/${host_model_base}"
    else
      docker_args+=(-v "$HF_CACHE_DIR:/hf-cache:ro")
    fi

    docker_args+=("$IMAGE" -m "$model_in_container" -ngl "$ngl" -c "$CTX_SIZE" -n "$N_PREDICT" --fit "$FIT_MODE" --single-turn -p "$PROMPT")
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
      docker_args+=("${EXTRA_ARGS[@]}")
    fi

    set +e
    timeout "${TIMEOUT_SEC}s" "${docker_args[@]}" >"$out_file" 2>&1
    rc=$?
    set -e
  else
    local -a local_args
    local_args=("$LLAMA_CLI" -m "$MODEL" -ngl "$ngl" -c "$CTX_SIZE" -n "$N_PREDICT" --fit "$FIT_MODE" --single-turn -p "$PROMPT")
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
      local_args+=("${EXTRA_ARGS[@]}")
    fi

    set +e
    timeout "${TIMEOUT_SEC}s" "${local_args[@]}" >"$out_file" 2>&1
    rc=$?
    set -e
  fi

  return "$rc"
}

classify_failure() {
  local out_file="$1"
  if rg -qi "cudaMalloc failed|out of memory|failed to allocate CUDA|unable to allocate CUDA|Failed to load the model" "$out_file"; then
    echo "oom_or_load_fail"
  elif rg -qi "forward compatibility was attempted on non supported HW|no usable GPU found|failed to initialize CUDA|invalid device function" "$out_file"; then
    echo "gpu_runtime_error"
  elif rg -qi "timed out|timeout" "$out_file"; then
    echo "timeout"
  else
    echo "other_error"
  fi
}

probe_ngl() {
  local ngl="$1"
  if [ -n "${STATUS[$ngl]:-}" ]; then
    return "${STATUS[$ngl]}"
  fi

  local attempt=0
  local rc=1
  local out_file="${TMPDIR_PATH}/ngl_${ngl}.log"
  local reason=""

  while [ "$attempt" -le "$RETRIES" ]; do
    if run_probe_once "$ngl" "$out_file"; then
      rc=0
      break
    fi
    rc=$?
    attempt=$((attempt + 1))
  done

  if [ "$rc" -eq 0 ]; then
    STATUS[$ngl]=0
    REASON[$ngl]="pass"
    echo "PASS ngl=$ngl"
    if [ "$VERBOSE" -eq 1 ]; then
      tail -n 30 "$out_file" | sed 's/^/  /'
    else
      rg -n "Prompt:|Generation:|memory breakdown|Exiting" "$out_file" | tail -n 5 | sed 's/^/  /' || true
    fi
    return 0
  fi

  reason="$(classify_failure "$out_file")"
  STATUS[$ngl]=1
  REASON[$ngl]="$reason"
  echo "FAIL ngl=$ngl rc=$rc reason=$reason"
  rg -n "cudaMalloc failed|out of memory|failed to load model|Failed to load the model|forward compatibility|no usable GPU|failed to initialize CUDA|invalid device function" "$out_file" | head -n 8 | sed 's/^/  /' || true
  return 1
}

echo "Probe config:"
echo "  backend=$BACKEND"
echo "  model=$MODEL"
echo "  min=$MIN_NGL max=$MAX_NGL hard_max=$HARD_MAX"
echo "  ctx_size=$CTX_SIZE n_predict=$N_PREDICT timeout=${TIMEOUT_SEC}s retries=$RETRIES"
echo "  safe_margin=$SAFE_MARGIN"
echo "  fit=$FIT_MODE"

best=-1
low="$MIN_NGL"
high=""

if probe_ngl "$MIN_NGL"; then
  best="$MIN_NGL"
else
  die "Minimum ngl=$MIN_NGL already fails; cannot determine usable range"
fi

if [ "$MAX_NGL" = "auto" ]; then
  high=$((MIN_NGL + 1))
  if [ "$high" -lt 1 ]; then
    high=1
  fi

  while [ "$high" -le "$HARD_MAX" ]; do
    if probe_ngl "$high"; then
      best="$high"
      low=$((high + 1))
      high=$((high * 2))
      if [ "$high" -gt "$HARD_MAX" ]; then
        high="$HARD_MAX"
        break
      fi
    else
      break
    fi
  done

  if [ "$low" -gt "$high" ]; then
    low="$high"
  fi
else
  high="$MAX_NGL"
  [ "$high" -le "$HARD_MAX" ] || die "--max must be <= --hard-max"
  low=$((MIN_NGL + 1))
fi

if [ "$high" -gt "$HARD_MAX" ]; then
  high="$HARD_MAX"
fi

# If high passes and equals hard max, we only know best >= hard max.
if [ "$MAX_NGL" = "auto" ] && [ "$high" -eq "$HARD_MAX" ]; then
  if probe_ngl "$high"; then
    best="$high"
    low=$((high + 1))
  fi
fi

while [ "$low" -le "$high" ]; do
  mid=$(( (low + high) / 2 ))
  if probe_ngl "$mid"; then
    best="$mid"
    low=$((mid + 1))
  else
    high=$((mid - 1))
  fi
done

[ "$best" -ge 0 ] || die "No passing ngl found"

recommended=$((best - SAFE_MARGIN))
if [ "$recommended" -lt 0 ]; then
  recommended=0
fi

echo
echo "RESULT:"
echo "  max_stable_ngl=$best"
echo "  recommended_ngl=$recommended  (safe_margin=$SAFE_MARGIN)"
echo "  tested_ctx_size=$CTX_SIZE"
echo "  tested_n_predict=$N_PREDICT"

if [ -n "$JSON_OUT" ]; then
  cat >"$JSON_OUT" <<EOF
{
  "model": "$(printf '%s' "$MODEL" | sed 's/"/\\"/g')",
  "backend": "$BACKEND",
  "max_stable_ngl": $best,
  "recommended_ngl": $recommended,
  "safe_margin": $SAFE_MARGIN,
  "fit": "$FIT_MODE",
  "ctx_size": $CTX_SIZE,
  "n_predict": $N_PREDICT,
  "timeout_sec": $TIMEOUT_SEC,
  "retries": $RETRIES
}
EOF
  echo "  json_out=$JSON_OUT"
fi
