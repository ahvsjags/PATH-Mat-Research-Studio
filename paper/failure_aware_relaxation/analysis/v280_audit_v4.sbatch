#!/bin/bash
#SBATCH --job-name=pathmat_v280_audit_v4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280/logs/v280_audit_v4_%j.out
#SBATCH --error=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280/logs/v280_audit_v4_%j.err

set -euo pipefail
ROOT=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V280
PY=/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runtime/venv/bin/python
cd "$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RUN="$ROOT/runs/holdout/v280_failure_repair_fresh_b4b4_v1/scored"
test -f "$RUN/run_completed.json"
"$PY" material_generation/analysis/audit_v280_confirmation_v4.py
