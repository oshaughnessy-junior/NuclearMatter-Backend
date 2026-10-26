#!/bin/bash
# Run from the repo root: sbatch -M sporc -A <account> --array=0-12 docs/devlog/2026-10-25-compose-header-floor/tests.sbatch
#SBATCH -p interactive --qos=qos_interactive
#SBATCH --cpus-per-task=4 --mem=24G --time=02:00:00
#SBATCH -J nmb-tests -o docs/devlog/2026-10-25-compose-header-floor/tests_%A_%a.out
# one pytest process per test file; FILES order fixed
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
FILES=(tests/test_compose.py tests/test_model_diagnostic.py tests/test_bridge_prior.py
       tests/test_crust.py tests/test_core_stability.py tests/test_reference_surface.py
       tests/test_tidal_cs2_consistency.py tests/test_no_recompile.py tests/test_ejecta.py
       tests/test_lab_likelihood.py tests/test_cooling_geometry.py tests/test_cooling_whole.py
       tests/test_nscool_io.py)
F=${FILES[$SLURM_ARRAY_TASK_ID]}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_nmb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf $JAX_COMPILATION_CACHE_DIR' EXIT
PY=/home/rossma/miniconda3/envs/genesis-gpu/bin/python
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} HEAD $(git rev-parse HEAD) dirty=$(git status --porcelain -- src tests | wc -l) host $(hostname) file $F"
"$PY" -c "import nmbackend; print('nmbackend', nmbackend.__file__)"
"$PY" -m pytest -q -p no:cacheprovider "$F" 2>&1 | grep -v cpu_aot_loader
echo "rc=${PIPESTATUS[0]}"
