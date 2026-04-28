#!/usr/bin/env bash
#SBATCH --job-name=hpdeco_cpu
#SBATCH --cluster=chip-cpu
#SBATCH --account=pi_gobbert
#SBATCH --partition=match
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=hpdeco_cpu_%j.slurm.out
#SBATCH --error=hpdeco_cpu_%j.slurm.out

set -euo pipefail

# ============================================================
# Work from the folder where sbatch was submitted
# ============================================================

RUN_DIR="${SLURM_SUBMIT_DIR}"
cd "${RUN_DIR}"

export TMPDIR="${RUN_DIR}/tmp_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${RUN_DIR}/.cache"
export PIP_CACHE_DIR="${RUN_DIR}/.cache/pip"

mkdir -p "${TMPDIR}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${PIP_CACHE_DIR}"

RUN_LOG="${RUN_DIR}/hpdeco_cpu_${SLURM_JOB_ID}.run.log"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "============================================================"
echo "HPDECO CPU Slurm Job"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Node:          $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "Run dir:       ${RUN_DIR}"
echo "Tmp dir:       ${TMPDIR}"
echo "Start time:    $(date)"
echo "============================================================"
echo

module purge
module load Python/3.11.5-GCCcore-13.2.0

echo "Loaded modules:"
module list
echo

source /umbc/rs/pi_mhms379/common/research/HPDECO/envs/hpdeco_env/bin/activate

echo "Python:"
which python
python --version
echo

echo "CPU node info:"
hostname
lscpu | head -n 20
echo

PROJECT_DIR="/umbc/rs/pi_mhms379/common/research/HPDECO/src/Quantum-Accelerated-Adjoint-Method-PDE-Constrained-Optimization"

cd "${PROJECT_DIR}"

echo "Current directory:"
pwd
echo

echo "Running experiment..."
python run.py configs/ibm_run_heat_convergence.yaml

echo
echo "============================================================"
echo "Job finished successfully"
echo "End time: $(date)"
echo "============================================================"

rm -rf "${TMPDIR}"