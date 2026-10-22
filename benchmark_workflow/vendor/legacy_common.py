"""Structured raw-data handling and conservative correlated-data diagnostics."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import t as student_t
import generate_revision as gen
from log_data import parse_thermo, audit

gen.ROOT = ROOT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def csv_out(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["no_completed_rows"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def frames(path):
    with Path(path).open(encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Invalid dump header: {path}")
            step = int(f.readline())
            if f.readline().strip() not in ("ITEM: NUMBER OF ATOMS", "ITEM: NUMBER OF ENTRIES"):
                raise ValueError("Invalid dump count header")
            n = int(f.readline())
            header = f.readline().split()
            if header != ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"]:
                raise ValueError("Only orthogonal periodic snapshots are supported")
            bounds = [list(map(float, f.readline().split())) for _ in range(3)]
            columns = f.readline().split()[2:]
            rows = [list(map(float, f.readline().split())) for _ in range(n)]
            if any(len(r) != len(columns) for r in rows):
                raise ValueError("Truncated dump")
            yield step, columns, np.asarray(rows).reshape(n, len(columns)), bounds


def dump_to_data(source, target, case):
    records = list(frames(source))
    if len(records) != 1:
        raise ValueError("A continuation checkpoint must contain exactly one frame")
    step, columns, data, bounds = records[0]
    ids = data[:, columns.index("id")]
    types = data[:, columns.index("type")]
    if len(np.unique(ids)) != len(ids) or not np.all(ids == ids.astype(int)):
        raise ValueError("Invalid atom IDs")
    if not np.all(types == types.astype(int)) or not np.all((types >= 1) & (types <= case.ntypes)):
        raise ValueError("Invalid atom types")
    rows = ["Incremental checkpoint converted from a verified ASCII snapshot", "", f"{len(data)} atoms",
            f"{case.ntypes} atom types", ""]
    rows += [f"{lo:.17g} {hi:.17g} {axis}lo {axis}hi" for (lo, hi), axis in zip(bounds, "xyz")]
    rows += ["", "Masses", ""] + [f"{i} {m:.17g}" for i, m in enumerate(case.masses, 1)]
    rows += ["", "Atoms # atomic", ""]
    for r in data[np.argsort(ids)]:
        values = [int(r[columns.index("id")]), int(r[columns.index("type")])]
        values += [float(r[columns.index(axis)]) for axis in "xyz"]
        rows.append(f"{values[0]} {values[1]} " + " ".join(f"{v:.17g}" for v in values[2:]))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text("\n".join(rows) + "\n", encoding="ascii")
    return {"source_step": step, "natoms": len(data), "type_counts": [int(np.sum(types == i)) for i in range(1, case.ntypes + 1)],
            "source_sha256": sha(source), "data_sha256": sha(target)}


def correlated(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 4 or not np.all(np.isfinite(x)):
        return {"mean": float(np.mean(x)), "se": None, "ess": 0.0, "df": 1, "constant": True}
    mean = float(np.mean(x))
    constant = bool(np.ptp(x) <= 128 * np.finfo(float).eps * max(1, abs(mean)))
    if constant:
        return {"mean": mean, "se": 0.0, "ess": 0.0, "df": 1, "constant": True}
    y = x - mean
    size = 1 << (2 * len(x) - 1).bit_length()
    ft = np.fft.rfft(y, size)
    acf = np.fft.irfft(ft * ft.conjugate(), size)[:len(x)]
    acf /= acf[0]
    positive = 0.0
    for i in range(1, len(x) - 1, 2):
        pair = acf[i] + acf[i + 1]
        if pair <= 0:
            break
        positive += pair
    ess = min(float(len(x)), len(x) / max(1, 1 + 2 * positive))
    se = float(np.std(x, ddof=1) / math.sqrt(ess))
    df = max(1, int(ess) - 1)
    for width in (1, 2, 4, 8, 16, 32, 64, 128):
        blocks = len(x) // width
        if blocks < 4:
            continue
        means = x[-blocks * width:].reshape(blocks, width).mean(axis=1)
        block_se = float(np.std(means, ddof=1) / math.sqrt(blocks))
        if block_se > se:
            se, df = block_se, blocks - 1
    return {"mean": mean, "se": se, "ess": ess, "df": df, "constant": False}


def drift(values):
    half = len(values) // 2
    a, b = correlated(values[:half]), correlated(values[half:])
    if a["se"] is None or b["se"] is None:
        return {"delta": None, "ci95_halfwidth": None, "flag": True}
    delta = b["mean"] - a["mean"]
    width = float(student_t.ppf(.975, max(1, min(a["df"], b["df"]))) * math.hypot(a["se"], b["se"]))
    floor = 128 * np.finfo(float).eps * max(1, abs(a["mean"]), abs(b["mean"]))
    return {"delta": delta, "ci95_halfwidth": width, "flag": abs(delta) > max(width, floor)}


def converted(text, ntypes, origin=0.0):
    raw = parse_thermo(text)
    if len(raw) < 2:
        raise ValueError("Insufficient thermo snapshots")
    first_attempts = raw[0]["f_mc[1]"]
    result = []
    for r in raw:
        n = int(r["Atoms"])
        if not math.isclose(r["PotEng"] / n, r["v_epa"], rel_tol=2e-14, abs_tol=2e-14):
            raise ValueError("Energy normalization mismatch")
        counts = [int(r[f"c_ctype[{i}]"]) for i in range(1, ntypes + 1)]
        if sum(counts) != n:
            raise ValueError("Type counts do not sum to N")
        row = {"local_md_step": int(r["Step"]), "N": n, "U_total": r["PotEng"], "U_per_atom": r["v_epa"],
               "sweeps": origin + (r["f_mc[1]"] - first_attempts) / n, "attempts_local": int(r["f_mc[1]"]),
               "accepts_local": int(r["f_mc[2]"]), "temperature": r.get("Temp"), "pressure": r.get("Press"),
               "minimum_pair_distance": r.get("c_rmin")}
        row.update({f"x_type_{i}": counts[i-1] / n for i in range(1, ntypes + 1)})
        row.update({f"n_type_{i}": counts[i-1] for i in range(1, ntypes + 1)})
        result.append(row)
    return result
