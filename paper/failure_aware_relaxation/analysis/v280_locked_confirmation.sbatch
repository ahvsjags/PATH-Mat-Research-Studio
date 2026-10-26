#!/bin/bash
#SBATCH --job-name=pathmat_v280_confirm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280/logs/v280_confirm_%j.out
#SBATCH --error=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280/logs/v280_confirm_%j.err

set -euo pipefail
ROOT=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280
PY=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runtime/venv/bin/python
RUN="$ROOT/runs/holdout/v280_failure_repair_fresh_b4b4_v1/scored"
cd "$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ -e "$RUN/run_completed.json" ]]; then
  echo "Locked V280 run already completed; refusing to rescore."
  exit 2
elif [[ -d "$RUN" ]]; then
  "$PY" material_generation/analysis/run_v280_confirmation.py --resume
else
  "$PY" material_generation/analysis/run_v280_confirmation.py --preflight-only
  "$PY" material_generation/analysis/run_v280_confirmation.py
fi
