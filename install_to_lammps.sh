#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! -d "$1/src/MC" ]]; then
  echo "Usage: bash install_to_lammps.sh /path/to/compatible/lammps" >&2
  exit 1
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for name in fix_fast_swap.cpp fix_fast_swap.h; do
  if [[ -f "$1/src/MC/$name" ]]; then
    cp -p -- "$1/src/MC/$name" "$1/src/MC/$name.before_20261020.$(date +%Y%m%d%H%M%S).bak"
  fi
  cp -- "$HERE/src/MC/$name" "$1/src/MC/$name"
done
echo "Sources copied with backups. Rebuild both the LAMMPS library and executable."
