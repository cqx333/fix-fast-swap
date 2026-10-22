"""Parse explicitly unnormalised revision logs; never guess legacy LJ normalisation."""
from __future__ import annotations

import math
import re

LOOP = re.compile(r"Loop time of\s+([\d.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms")


def production_text(text):
    lines = text.splitlines()
    begin = [i for i, s in enumerate(lines) if s.strip() == "REVISION_PRODUCTION_BEGIN"]
    if not begin:
        return text
    end = next((i for i in range(begin[-1] + 1, len(lines)) if lines[i].strip() == "REVISION_PRODUCTION_END"), len(lines))
    return "\n".join(lines[begin[-1] + 1:end])


def parse_thermo(text):
    header = None
    values = []
    for line in production_text(text).splitlines():
        parts = line.split()
        if parts and parts[0] == "Step" and "PotEng" in parts:
            header = parts
        elif line.startswith("Loop time"):
            header = None
        elif header and len(parts) == len(header):
            try:
                values.append(dict(zip(header, map(float, parts))))
            except ValueError:
                pass
    return values


def audit(text, spec):
    data = parse_thermo(text)
    loops = LOOP.findall(production_text(text))
    fallback = bool(re.search(r"parallel MC disabled|Safe interior too small", text, re.I))
    errors = re.findall(r"^ERROR[^\n]*", text, re.M)
    warnings = re.findall(r"^WARNING[^\n]*", text, re.M)
    result = {"fallback": fallback, "errors": errors, "warnings": warnings,
              "loop_s": float(loops[-1][0]) if loops else None, "thermo_samples": len(data)}
    if loops:
        _, np_, steps, atoms = loops[-1]
        result.update(observed_ranks=int(np_), observed_run_steps=int(steps), observed_natoms=int(atoms))
    if data and "v_epa" in data[0]:
        for row in data:
            if not math.isfinite(row["PotEng"]) or not math.isclose(row["PotEng"] / row["Atoms"], row["v_epa"], rel_tol=2e-14, abs_tol=2e-14):
                raise ValueError("Total/per-atom energy normalization mismatch")
            if row["Atoms"] != spec["natoms"]:
                raise ValueError("Unexpected atom count")
            counts = [v for k, v in row.items() if k.startswith("c_ctype[")]
            if not math.isclose(sum(counts), row["Atoms"], abs_tol=1e-8):
                raise ValueError("Type counts do not sum to N")
        attempts = data[-1]["f_mc[1]"] - data[0]["f_mc[1]"]
        accepts = data[-1]["f_mc[2]"] - data[0]["f_mc[2]"]
        result.update(actual_attempts=attempts, actual_sweeps=attempts / spec["natoms"], accepted_moves=accepts,
                      acceptance_fraction=accepts / attempts if attempts > 0 else None)
        if result["loop_s"] is not None and attempts > 0:
            result["time_per_actual_sweep_s"] = result["loop_s"] / result["actual_sweeps"]
    ghosts = re.findall(r"ghost atom cutoff\s*=\s*([\d.eE+-]+)", text)
    result["observed_ghost_cutoff"] = float(ghosts[-1]) if ghosts else None
    grids = re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", text)
    result["observed_processor_grid"] = [int(x) for x in grids[-1]] if grids else None
    if spec.get("expect_parallel"):
        if fallback:
            raise ValueError("A required parallel run fell back to serial selection")
        if result["observed_processor_grid"] != [1, 1, spec["ranks"]]:
            raise ValueError("Unexpected MPI decomposition")
        if not ghosts or spec["half_domain_z"] <= result["observed_ghost_cutoff"]:
            raise ValueError("Observed communication radius violates the independence margin")
    timer_rows = {}
    for line in production_text(text).splitlines():
        cells = [x.strip() for x in line.split("|")]
        if len(cells) == 6:
            try:
                timer_rows[cells[0]] = dict(zip(["min_s", "avg_s", "max_s", "variance_percent", "loop_percent"], map(float, cells[1:])))
            except ValueError:
                pass
    result["timer_breakdown"] = timer_rows
    return result
