#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash run_all.sh /absolute/path/lmp_mpi [--suite parallel,mdmc,timing,size3d] [--dry-run]\n' >&2
  exit 2
fi
LMP=$1
shift
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python scripts/workflow.py --lmp "$LMP" --expected-library-dir "$(dirname -- "$LMP")" "$@"
