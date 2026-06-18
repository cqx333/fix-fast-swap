#!/usr/bin/env bash
#SBATCH --job-name=fastswap_perf
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Edit this path for the cluster installation if LMP_BIN is not passed by sbatch:
#   sbatch --export=ALL,LMP_BIN=/path/to/lmp,MPI_LAUNCHER=srun,NP_FLAG=-n submit_slurm.sh
LMP_BIN="${LMP_BIN:-lmp}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"
NP_FLAG="${NP_FLAG:--n}"
MAX_RANKS="${MAX_RANKS:-24}"
SUITE="${SUITE:-all}"

export MPI_LAUNCHER NP_FLAG MAX_RANKS SUITE

bash run_all.sh "${LMP_BIN}"
