#!/usr/bin/env bash
# generate_heat_experiment.sh
#
# Generates YAML configs for every (nx, shots) combination and stores them in:
#   heatconfigs/nx_<nx>/<shots>shots/<shots>shots.yaml
#
# Edit the arrays below, or pass as CLI args:
#   ./generate_heat_experiment.sh --nx "4 8 16" --shots "2048 4096 8192 16384"

set -euo pipefail

# ── Edit these to change the sweep ───────────────────────────────────────────
NX_VALUES=(4 8 16)
SHOTS_VALUES=(2048 4096 8192 16384)

# ── Parse optional CLI overrides ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nx)    read -ra NX_VALUES    <<< "$2"; shift 2 ;;
    --shots) read -ra SHOTS_VALUES <<< "$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Shared settings (mirrors heat_hybrid2.yaml) ───────────────────────────────
N=8
MAX_ITER=100
ALPHA=55
ARMIJO_C=1e-6
TAU=0.5
MIN_STEP=1e-8
MAX_BACKTRACKS=50
SPECTRAL_POINTS=16
DELTA=0.001
SCALING_SIZES="[4,8,16]"
SCALING_ITERATIONS=1

# ── Sweep ─────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#NX_VALUES[@]} * ${#SHOTS_VALUES[@]} ))
RUN=0

for NX in "${NX_VALUES[@]}"; do
  for SHOTS in "${SHOTS_VALUES[@]}"; do
    RUN=$(( RUN + 1 ))

    OUTPUT_DIR="heatconfigs/nx_${NX}/${SHOTS}shots"
    CONFIG_FILE="${OUTPUT_DIR}/${SHOTS}shots.yaml"

    mkdir -p "$OUTPUT_DIR"

    cat > "$CONFIG_FILE" <<YAML
experiment: heat

model:
  n: ${N}
  nx: ${NX}

optimizer:
  max_iter: ${MAX_ITER}
  alpha: ${ALPHA}
  use_backtracking: true
  armijo_c: ${ARMIJO_C}
  backtracking_tau: ${TAU}
  min_step: ${MIN_STEP}
  max_backtracks: ${MAX_BACKTRACKS}

solver:
  mode: hybrid

quantum:
  backend_mode: aer
  shots: ${SHOTS}
  spectral_points: ${SPECTRAL_POINTS}
  delta: ${DELTA}
  use_preconditioning: false

scaling:
  sizes: ${SCALING_SIZES}
  iterations: ${SCALING_ITERATIONS}

plots:
  show_solution: false
  show_scaling: false
  show_superimposed_histories: true
  output_dir: ${OUTPUT_DIR}
YAML

    echo "[${RUN}/${TOTAL}] Generated: ${CONFIG_FILE}"

  done
done

echo ""
echo "All ${TOTAL} configs generated in heatconfigs/"
