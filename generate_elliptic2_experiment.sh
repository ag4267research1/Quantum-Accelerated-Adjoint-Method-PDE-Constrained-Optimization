#!/usr/bin/env bash
# generate_elliptic2_experiment.sh
#
# Generates YAML configs for every (experiment_type, nx, shots) combination
# and stores them in:
#   elliptic2configs/<experiment_type>/nx_<nx>/<shots>shots/<shots>shots.yaml
#
# Edit the arrays below, or pass as CLI args:
#   ./generate_elliptic2_experiment.sh --exp "exp1 exp2 exp3 exp4 exp5" --nx "4 8 16" --shots "2048 4096 8192 16384"

set -euo pipefail

# ── Edit these to change the sweep ───────────────────────────────────────────
EXP_VALUES=(exp1 exp2 exp3 exp4 exp5)
NX_VALUES=(4 8 16)
SHOTS_VALUES=(2048 4096 8192 16384)

# ── Parse optional CLI overrides ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)   read -ra EXP_VALUES   <<< "$2"; shift 2 ;;
    --nx)    read -ra NX_VALUES    <<< "$2"; shift 2 ;;
    --shots) read -ra SHOTS_VALUES <<< "$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Shared settings (mirrors elliptic2_test.yaml) ─────────────────────────────
ALPHA=1e-2
MAX_ITER=200
OPT_ALPHA=70
ARMIJO_C=1e-6
TAU=0.5
MIN_STEP=1e-8
MAX_BACKTRACKS=50
SPECTRAL_POINTS=16
DELTA=0.001
SCALING_SIZES="[4,8,16]"

# ── Sweep ─────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#EXP_VALUES[@]} * ${#NX_VALUES[@]} * ${#SHOTS_VALUES[@]} ))
RUN=0

for EXP in "${EXP_VALUES[@]}"; do
  for NX in "${NX_VALUES[@]}"; do
    for SHOTS in "${SHOTS_VALUES[@]}"; do
      RUN=$(( RUN + 1 ))

      OUTPUT_DIR="elliptic2configs/${EXP}/nx_${NX}/${SHOTS}shots"
      CONFIG_FILE="${OUTPUT_DIR}/${SHOTS}shots.yaml"

      mkdir -p "$OUTPUT_DIR"

      cat > "$CONFIG_FILE" <<YAML
experiment: elliptic2

experiment_type: ${EXP}

alpha: ${ALPHA}

model:
  nx: ${NX}

optimizer:
  max_iter: ${MAX_ITER}
  alpha: ${OPT_ALPHA}
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

plots:
  show_solution: false
  show_scaling: false
  show_superimposed_histories: true
  output_dir: ${OUTPUT_DIR}
YAML

      echo "[${RUN}/${TOTAL}] Generated: ${CONFIG_FILE}"

    done
  done
done

echo ""
echo "All ${TOTAL} configs generated in elliptic2configs/"
