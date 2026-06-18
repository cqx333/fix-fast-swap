#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/lammps/source/root" >&2
  exit 1
fi

LAMMPS_ROOT="$1"
TARGET_DIR="${LAMMPS_ROOT}/src/MC"

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Could not find LAMMPS MC source directory: ${TARGET_DIR}" >&2
  exit 1
fi

cp src/MC/fix_fast_swap.cpp "${TARGET_DIR}/fix_fast_swap.cpp"
cp src/MC/fix_fast_swap.h "${TARGET_DIR}/fix_fast_swap.h"

echo "Installed fix_fast_swap.cpp and fix_fast_swap.h into ${TARGET_DIR}"
