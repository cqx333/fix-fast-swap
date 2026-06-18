#!/usr/bin/env bash
set -euo pipefail

LMP_BIN="${1:-/home/chenqx/LAMMPS-opt/build-new/lmp_mpi}"

echo "hostname: $(hostname)"
echo "date: $(date -Is)"
echo "LMP_BIN argument: ${LMP_BIN}"
echo "resolved LMP_BIN: $(readlink -f "${LMP_BIN}")"
echo

echo "== file timestamps =="
ls -l --time-style=full-iso "$(readlink -f "${LMP_BIN}")"
if ldd "${LMP_BIN}" | grep -i lammps >/tmp/lammps_libraries.$$; then
  cat /tmp/lammps_libraries.$$
  awk '{print $3}' /tmp/lammps_libraries.$$ | while read -r lib; do
    if [[ -n "${lib}" && -e "${lib}" ]]; then
      ls -l --time-style=full-iso "${lib}"
    fi
  done
fi
rm -f /tmp/lammps_libraries.$$
echo

echo "== hashes =="
sha256sum "$(readlink -f "${LMP_BIN}")"
ldd "${LMP_BIN}" | awk 'tolower($0) ~ /lammps|mpi/ {print $3}' | while read -r lib; do
  if [[ -n "${lib}" && -e "${lib}" ]]; then
    sha256sum "${lib}"
  fi
done
echo

echo "== loaded libraries =="
ldd "${LMP_BIN}" | grep -Ei 'lammps|mpi|stdc\+\+|gcc' || true
echo

echo "== fast/swap signature strings =="
for target in "$(readlink -f "${LMP_BIN}")" $(ldd "${LMP_BIN}" | awk 'tolower($0) ~ /lammps/ {print $3}'); do
  [[ -e "${target}" ]] || continue
  echo "-- ${target}"
  strings "${target}" | grep -E 'supports only explicitly listed|does not yet support pair_modify tail|pair->single|Fix fast/swap' | head -40 || true
done
echo

echo "== source tree search, if present =="
if [[ -d /home/chenqx/LAMMPS-opt ]]; then
  grep -RIn --exclude-dir=build --exclude-dir=.git \
    'supports only explicitly listed\|does not yet support pair_modify tail\|pair->single' \
    /home/chenqx/LAMMPS-opt 2>/dev/null | head -80 || true
fi
