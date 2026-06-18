#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LMP_BIN="${1:-${LMP_BIN:-lmp}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
NP_FLAG="${NP_FLAG:--np}"
MAX_RANKS="${MAX_RANKS:-24}"
SUITE="${SUITE:-all}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
APPEND_STATUS="${APPEND_STATUS:-0}"

echo "LAMMPS executable: ${LMP_BIN}"
echo "MPI launcher: ${MPI_LAUNCHER} ${NP_FLAG}"
echo "Suite: ${SUITE}"
echo "Max ranks: ${MAX_RANKS}"
echo "Append status: ${APPEND_STATUS}"

APPEND_ARGS=()
if [[ "${APPEND_STATUS}" == "1" ]]; then
  APPEND_ARGS+=(--append-status)
fi

"${PYTHON_BIN}" scripts/generate_inputs.py --suite "${SUITE}" --max-ranks "${MAX_RANKS}"
"${PYTHON_BIN}" scripts/run_workflow.py \
  --lmp "${LMP_BIN}" \
  --mpi-launcher "${MPI_LAUNCHER}" \
  --np-flag="${NP_FLAG}" \
  --suite "${SUITE}" \
  --max-ranks "${MAX_RANKS}" \
  --timeout-s "${TIMEOUT_S}" \
  "${APPEND_ARGS[@]}"
"${PYTHON_BIN}" scripts/analyze_results.py

echo "Done. Main summary: results/performance_summary.md"
