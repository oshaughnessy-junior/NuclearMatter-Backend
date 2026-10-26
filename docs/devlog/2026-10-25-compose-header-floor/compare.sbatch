#!/bin/bash
# Run from the repo root: sbatch -M sporc -A <account> docs/devlog/2026-10-25-compose-header-floor/compare.sbatch
#SBATCH -p interactive --qos=qos_interactive --cpus-per-task=4 --mem=16G --time=01:30:00
#SBATCH -J nmb-compare -o docs/devlog/2026-10-25-compose-header-floor/compare_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1
PY=${PY:-/home/rossma/miniconda3/envs/genesis-gpu/bin/python}
echo "job $SLURM_JOB_ID HEAD $(git rev-parse HEAD) dirty=$(git status --porcelain -- src tests | wc -l) host $(hostname)"
"$PY" docs/devlog/2026-10-25-compose-header-floor/compare.py 2>&1 | grep -v cpu_aot_loader
echo "rc=${PIPESTATUS[0]}"
