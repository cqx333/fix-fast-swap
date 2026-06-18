from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NUM_RE = re.compile(r"^[\s+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def key_for(row: dict) -> tuple:
    return (
        row["suite"],
        row["case"],
        row["ranks"],
        row["natoms"],
        tuple(row["box_lattice"]),
        row["nsweeps"],
        row.get("seed"),
    )


def diag_text(row: dict) -> str:
    parts = []
    parts.extend(row.get("errors") or [])
    parts.extend(row.get("warnings") or [])
    parts.append(row.get("stderr_tail") or "")
    parts.append(row.get("stdout_tail") or "")
    return "\n".join(str(part) for part in parts)


def verdict(row: dict) -> str:
    expected = row.get("expected", "pass")
    if expected == "fast_reject" and row.get("kind") == "fast":
        if row.get("returncode") == 0:
            return "FAIL_UNEXPECTED_PASS"
        pattern = row.get("expected_error_pattern")
        if not pattern or pattern.lower() in diag_text(row).lower():
            return "PASS_EXPECTED_REJECT"
        return "FAIL_WRONG_ERROR"
    if row.get("returncode") != 0:
        return "FAIL"
    if row.get("kind") == "fast" and row.get("ranks", 1) > 1 and row.get("fallback_detected"):
        return "FAIL_FALLBACK"
    if row.get("time_per_sweep_s") is None:
        return "FAIL_NO_TIMER"
    return "PASS"


def log_path(row: dict) -> Path:
    path = Path(row.get("log", ""))
    return path if path.is_absolute() else ROOT / path


def parse_thermo(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    header: list[str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Step" and parts[1] == "PotEng":
            header = parts
            continue
        if header and NUM_RE.match(line):
            parts = line.split()
            if len(parts) != len(header):
                continue
            try:
                rows.append({name: float(value) for name, value in zip(header, parts)})
            except ValueError:
                continue
        if line.startswith("Loop time"):
            header = None
    return rows


def max_aligned_diff(rows_a: list[dict], rows_b: list[dict], column: str) -> float | None:
    if not rows_a or not rows_b or len(rows_a) != len(rows_b):
        return None
    value = 0.0
    for a, b in zip(rows_a, rows_b):
        if a.get("Step") != b.get("Step"):
            return None
        if column not in a or column not in b:
            return None
        value = max(value, abs(a[column] - b[column]))
    return value


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def stderr(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    mu = mean(vals)
    var = sum((value - mu) ** 2 for value in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def stddev(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    mu = mean(vals)
    var = sum((value - mu) ** 2 for value in vals) / (len(vals) - 1)
    return math.sqrt(var)


def stationary_stats(rows: list[dict], ntypes: int, natoms: int) -> dict:
    if len(rows) < 4:
        return {"ok": False}
    window = rows[max(1, len(rows) // 4):]
    e = [row["PotEng"] / natoms for row in window if "PotEng" in row]
    comps = []
    for itype in range(1, ntypes + 1):
        col = f"c_ctype[{itype}]"
        comps.append([row[col] / natoms for row in window if col in row])
    if not e or any(not vals for vals in comps):
        return {"ok": False}
    return {
        "ok": True,
        "n_samples": len(window),
        "mean_e_per_atom": mean(e),
        "se_e_per_atom": stderr(e),
        "mean_x": [mean(vals) for vals in comps],
        "se_x": [stderr(vals) for vals in comps],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_strong_scaling(rows: list[dict]) -> None:
    strong = [row for row in rows if row.get("suite") == "strong_scaling"]
    if not strong:
        return

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in strong:
        grouped[(row["case"], row["kind"], row["natoms"], tuple(row["box_lattice"]), row["nsweeps"])].append(row)

    summary_rows: list[dict] = []
    for key, group in sorted(grouped.items()):
        case, kind, natoms, box_lattice, nsweeps = key
        baseline = next((row for row in group if row["ranks"] == 1 and row["verdict"] == "PASS"), None)
        baseline_t = baseline.get("time_per_sweep_s") if baseline else None
        for row in sorted(group, key=lambda item: item["ranks"]):
            time_per_sweep = row.get("time_per_sweep_s")
            speedup = None
            efficiency = None
            if baseline_t and time_per_sweep and time_per_sweep > 0:
                speedup = baseline_t / time_per_sweep
                efficiency = speedup / row["ranks"]
            summary_rows.append(
                {
                    "case": case,
                    "kind": kind,
                    "ranks": row["ranks"],
                    "natoms": natoms,
                    "box_lattice": "x".join(map(str, box_lattice)),
                    "nsweeps": nsweeps,
                    "time_per_sweep_s": time_per_sweep,
                    "speedup_vs_rank1": speedup,
                    "parallel_efficiency_vs_rank1": efficiency,
                    "verdict": row["verdict"],
                    "fallback_detected": row.get("fallback_detected"),
                }
            )

    write_csv(RESULTS / "strong_scaling_summary.csv", summary_rows)
    md = ["# Strong Scaling Summary", ""]
    md.append("Fixed-size LJ/cut test. Speedup and efficiency are computed relative to the MPI-launched rank-1 run of the same fix.")
    md.append("")
    md.append("| case | kind | ranks | atoms | sweeps | time/sweep (s) | speedup vs P=1 | efficiency | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        time_s = f"{row['time_per_sweep_s']:.6g}" if row["time_per_sweep_s"] is not None else ""
        speedup_s = f"{row['speedup_vs_rank1']:.6g}" if row["speedup_vs_rank1"] is not None else ""
        eff_s = f"{row['parallel_efficiency_vs_rank1']:.6g}" if row["parallel_efficiency_vs_rank1"] is not None else ""
        md.append(
            f"| {row['case']} | {row['kind']} | {row['ranks']} | {row['natoms']} | {row['nsweeps']} | "
            f"{time_s} | {speedup_s} | {eff_s} | {row['verdict']} |"
        )
    md.append("")
    (RESULTS / "strong_scaling_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_timeseries_exports(rows: list[dict]) -> None:
    timeseries_dir = RESULTS / "timeseries"
    manifest_rows: list[dict] = []
    for row in rows:
        thermo_rows = parse_thermo(log_path(row))
        if not thermo_rows:
            continue
        timeseries_dir.mkdir(parents=True, exist_ok=True)
        ntypes = int(row.get("ntypes") or 0)
        natoms = int(row.get("natoms") or 0)
        out_path = timeseries_dir / f"{row['id']}.csv"
        base_fields = [
            "id",
            "suite",
            "analysis_role",
            "case",
            "kind",
            "ranks",
            "natoms",
            "seed",
            "sample_index",
            "step",
            "sweep",
            "pe",
            "e_per_atom",
            "mc_attempt",
            "mc_accept",
            "acceptance_ratio",
        ]
        count_fields = [f"n_type_{itype}" for itype in range(1, ntypes + 1)]
        comp_fields = [f"x_type_{itype}" for itype in range(1, ntypes + 1)]
        out_rows: list[dict] = []
        for idx, thermo in enumerate(thermo_rows):
            step = thermo.get("Step")
            pe = thermo.get("PotEng")
            attempt = thermo.get("f_mc[1]")
            accept = thermo.get("f_mc[2]")
            out_row = {
                "id": row["id"],
                "suite": row.get("suite"),
                "analysis_role": row.get("analysis_role"),
                "case": row.get("case"),
                "kind": row.get("kind"),
                "ranks": row.get("ranks"),
                "natoms": natoms,
                "seed": row.get("seed"),
                "sample_index": idx,
                "step": step,
                "sweep": step,
                "pe": pe,
                "e_per_atom": (pe / natoms) if pe is not None and natoms else None,
                "mc_attempt": attempt,
                "mc_accept": accept,
                "acceptance_ratio": (accept / attempt) if attempt and attempt > 0 and accept is not None else None,
            }
            for itype in range(1, ntypes + 1):
                count = thermo.get(f"c_ctype[{itype}]")
                out_row[f"n_type_{itype}"] = count
                out_row[f"x_type_{itype}"] = (count / natoms) if count is not None and natoms else None
            out_rows.append(out_row)
        write_csv(out_path, out_rows)
        manifest_rows.append(
            {
                "id": row["id"],
                "suite": row.get("suite"),
                "analysis_role": row.get("analysis_role"),
                "case": row.get("case"),
                "kind": row.get("kind"),
                "ranks": row.get("ranks"),
                "natoms": natoms,
                "box_lattice": "x".join(map(str, row.get("box_lattice", []))),
                "nsweeps": row.get("nsweeps"),
                "thermo_every": row.get("thermo_every"),
                "seed": row.get("seed"),
                "n_samples": len(out_rows),
                "verdict": row.get("verdict"),
                "csv": out_path.relative_to(ROOT).as_posix(),
            }
        )
    write_csv(RESULTS / "timeseries_manifest.csv", manifest_rows)


def write_timing_repeats(rows: list[dict]) -> None:
    repeat_rows = [row for row in rows if row.get("analysis_role") == "timing_repeat"]
    if not repeat_rows:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in repeat_rows:
        grouped[key_for(row)][row["kind"]] = row

    raw_rows: list[dict] = []
    for key, pair in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][3], item[0][6])):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_t = atom.get("time_per_sweep_s") if atom else None
        fast_t = fast.get("time_per_sweep_s") if fast else None
        speedup = atom_t / fast_t if atom_t and fast_t and fast_t > 0 else None
        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "atom_time_per_sweep_s": atom_t,
                "fast_time_per_sweep_s": fast_t,
                "speedup_atom_over_fast": speedup,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "fast_fallback_detected": fast.get("fallback_detected") if fast else None,
            }
        )
    write_csv(RESULTS / "timing_repeats_raw.csv", raw_rows)

    aggregate: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw_rows:
        if row["atom_time_per_sweep_s"] is None or row["fast_time_per_sweep_s"] is None:
            continue
        key = (row["case"], row["ranks"], row["natoms"], row["box_lattice"], row["nsweeps"])
        aggregate[key].append(row)

    summary_rows: list[dict] = []
    for key, vals in sorted(aggregate.items(), key=lambda item: (item[0][1], item[0][2])):
        case, ranks, natoms, box_lattice, nsweeps = key
        atom_vals = [row["atom_time_per_sweep_s"] for row in vals]
        fast_vals = [row["fast_time_per_sweep_s"] for row in vals]
        speed_vals = [row["speedup_atom_over_fast"] for row in vals if row["speedup_atom_over_fast"] is not None]
        summary_rows.append(
            {
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": box_lattice,
                "nsweeps": nsweeps,
                "n_repeats": len(vals),
                "atom_time_mean_s": mean(atom_vals),
                "atom_time_std_s": stddev(atom_vals),
                "fast_time_mean_s": mean(fast_vals),
                "fast_time_std_s": stddev(fast_vals),
                "speedup_mean": mean(speed_vals),
                "speedup_std": stddev(speed_vals),
                "speedup_min": min(speed_vals) if speed_vals else None,
                "speedup_max": max(speed_vals) if speed_vals else None,
            }
        )
    write_csv(RESULTS / "timing_repeats_summary.csv", summary_rows)


def write_size_scaling(rows: list[dict]) -> None:
    size_rows = [row for row in rows if row.get("suite") == "size_scaling"]
    if not size_rows:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in size_rows:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    for key, pair in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][3])):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_t = atom.get("time_per_sweep_s") if atom else None
        fast_t = fast.get("time_per_sweep_s") if fast else None
        speedup = atom_t / fast_t if atom_t and fast_t and fast_t > 0 else None
        out.append(
            {
                "scaling_mode": "serial_size" if ranks == 1 else "parallel_size",
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "atom_time_per_sweep_s": atom_t,
                "fast_time_per_sweep_s": fast_t,
                "speedup_atom_over_fast": speedup,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "fast_fallback_detected": fast.get("fallback_detected") if fast else None,
            }
        )
    write_csv(RESULTS / "size_scaling_summary.csv", out)


def write_nonlj_parallel_scaling(rows: list[dict]) -> None:
    nonlj_rows = [row for row in rows if row.get("suite") == "nonlj_parallel_scaling"]
    if not nonlj_rows:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in nonlj_rows:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    for key, pair in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2])):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_t = atom.get("time_per_sweep_s") if atom else None
        fast_t = fast.get("time_per_sweep_s") if fast else None
        speedup = atom_t / fast_t if atom_t and fast_t and fast_t > 0 else None
        out.append(
            {
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "atom_time_per_sweep_s": atom_t,
                "fast_time_per_sweep_s": fast_t,
                "speedup_atom_over_fast": speedup,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "fast_fallback_detected": fast.get("fallback_detected") if fast else None,
            }
        )
    write_csv(RESULTS / "nonlj_parallel_scaling_summary.csv", out)

    md = ["# Non-LJ Parallel Scaling Summary", ""]
    md.append("Fixed-size 6144-atom tests for representative non-LJ/cut pair styles.")
    md.append("")
    md.append("| case | ranks | atoms | sweeps | atom/sweep (s) | fast/sweep (s) | speedup | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in out:
        atom_t = f"{row['atom_time_per_sweep_s']:.6g}" if row["atom_time_per_sweep_s"] is not None else ""
        fast_t = f"{row['fast_time_per_sweep_s']:.6g}" if row["fast_time_per_sweep_s"] is not None else ""
        speed = f"{row['speedup_atom_over_fast']:.6g}" if row["speedup_atom_over_fast"] is not None else ""
        verdicts = f"{row['atom_verdict']}/{row['fast_verdict']}"
        md.append(
            f"| {row['case']} | {row['ranks']} | {row['natoms']} | {row['nsweeps']} | "
            f"{atom_t} | {fast_t} | {speed} | {verdicts} |"
        )
    md.append("")
    (RESULTS / "nonlj_parallel_scaling_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_unequal_cutoff_scaling(rows: list[dict]) -> None:
    unequal_rows = [row for row in rows if row.get("suite") == "unequal_cutoff_scaling"]
    if not unequal_rows:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in unequal_rows:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    for key, pair in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][3])):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_t = atom.get("time_per_sweep_s") if atom else None
        fast_t = fast.get("time_per_sweep_s") if fast else None
        speedup = atom_t / fast_t if atom_t and fast_t and fast_t > 0 else None
        out.append(
            {
                "scaling_mode": "serial_unequal_cutoff" if ranks == 1 else "parallel_unequal_cutoff",
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "atom_time_per_sweep_s": atom_t,
                "fast_time_per_sweep_s": fast_t,
                "speedup_atom_over_fast": speedup,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "fast_fallback_detected": fast.get("fallback_detected") if fast else None,
            }
        )
    write_csv(RESULTS / "unequal_cutoff_scaling_summary.csv", out)

    md = ["# Unequal-Cutoff Scaling Summary", ""]
    md.append("This suite isolates the exact maximum-cutoff path used when swappable type pairs have unequal pair cutoffs.")
    md.append("")
    md.append("| mode | ranks | atoms | box | sweeps | atom/sweep (s) | fast/sweep (s) | speedup | verdict |")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|---|")
    for row in out:
        atom_t = f"{row['atom_time_per_sweep_s']:.6g}" if row["atom_time_per_sweep_s"] is not None else ""
        fast_t = f"{row['fast_time_per_sweep_s']:.6g}" if row["fast_time_per_sweep_s"] is not None else ""
        speed = f"{row['speedup_atom_over_fast']:.6g}" if row["speedup_atom_over_fast"] is not None else ""
        verdicts = f"{row['atom_verdict']}/{row['fast_verdict']}"
        md.append(
            f"| {row['scaling_mode']} | {row['ranks']} | {row['natoms']} | {row['box_lattice']} | "
            f"{row['nsweeps']} | {atom_t} | {fast_t} | {speed} | {verdicts} |"
        )
    md.append("")
    (RESULTS / "unequal_cutoff_scaling_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_exact_consistency(rows: list[dict]) -> None:
    exact = [row for row in rows if row.get("analysis_role") == "exact_consistency"]
    if not exact:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in exact:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    for key, pair in sorted(grouped.items()):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_rows = parse_thermo(log_path(atom)) if atom else []
        fast_rows = parse_thermo(log_path(fast)) if fast else []
        pe = max_aligned_diff(atom_rows, fast_rows, "PotEng")
        counter = None
        for col in ("f_mc[1]", "f_mc[2]"):
            diff = max_aligned_diff(atom_rows, fast_rows, col)
            counter = diff if counter is None else (None if diff is None else max(counter, diff))
        counts = None
        ntypes = atom.get("ntypes", 0) if atom else 0
        for itype in range(1, ntypes + 1):
            diff = max_aligned_diff(atom_rows, fast_rows, f"c_ctype[{itype}]")
            counts = diff if counts is None else (None if diff is None else max(counts, diff))
        ok = (
            atom and fast
            and atom.get("verdict") == "PASS"
            and fast.get("verdict") == "PASS"
            and pe is not None and pe <= 1.0e-8
            and counts == 0.0
            and counter == 0.0
        )
        out.append(
            {
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "thermo_rows": min(len(atom_rows), len(fast_rows)),
                "max_abs_pe_diff": pe,
                "max_abs_type_count_diff": counts,
                "max_abs_mc_counter_diff": counter,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "consistency_verdict": "PASS" if ok else "FAIL",
            }
        )
    write_csv(RESULTS / "exact_consistency_summary.csv", out)
    md = ["# Exact Consistency Summary", ""]
    md.append("Single-rank runs use identical proposal schedules. The fast/swap input also enables internal local-vs-global energy-difference self-checks.")
    md.append("")
    md.append("| case | atoms | sweeps | thermo rows | max abs dPE | max type diff | max MC counter diff | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in out:
        pe = "" if row["max_abs_pe_diff"] is None else f"{row['max_abs_pe_diff']:.3e}"
        td = "" if row["max_abs_type_count_diff"] is None else f"{row['max_abs_type_count_diff']:.0f}"
        mc = "" if row["max_abs_mc_counter_diff"] is None else f"{row['max_abs_mc_counter_diff']:.0f}"
        md.append(
            f"| {row['case']} | {row['natoms']} | {row['nsweeps']} | {row['thermo_rows']} | "
            f"{pe} | {td} | {mc} | {row['consistency_verdict']} |"
        )
    md.append("")
    (RESULTS / "exact_consistency_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_parallel_stat_consistency(rows: list[dict]) -> None:
    stat_rows = [row for row in rows if row.get("analysis_role") == "statistical_consistency"]
    if not stat_rows:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in stat_rows:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    aggregate: dict[str, list[dict]] = defaultdict(list)
    for key, pair in sorted(grouped.items()):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_stats = stationary_stats(parse_thermo(log_path(atom)), atom.get("ntypes", 0), natoms) if atom else {"ok": False}
        fast_stats = stationary_stats(parse_thermo(log_path(fast)), fast.get("ntypes", 0), natoms) if fast else {"ok": False}
        ok_base = atom and fast and atom.get("verdict") == "PASS" and fast.get("verdict") == "PASS" and atom_stats.get("ok") and fast_stats.get("ok")
        de = dx = tol_e = tol_x = None
        verdict_stat = "FAIL"
        if ok_base:
            de = abs(atom_stats["mean_e_per_atom"] - fast_stats["mean_e_per_atom"])
            dx = max(abs(a - b) for a, b in zip(atom_stats["mean_x"], fast_stats["mean_x"]))
            se_e = math.sqrt(atom_stats["se_e_per_atom"] ** 2 + fast_stats["se_e_per_atom"] ** 2)
            se_x = max(math.sqrt(a ** 2 + b ** 2) for a, b in zip(atom_stats["se_x"], fast_stats["se_x"]))
            tol_e = max(0.05, 3.0 * se_e)
            tol_x = max(0.03, 3.0 * se_x)
            verdict_stat = "PASS" if de <= tol_e and dx <= tol_x else "WARN"
        row = {
            "case": case,
            "seed": seed,
            "ranks": ranks,
            "natoms": natoms,
            "box_lattice": "x".join(map(str, box_lattice)),
            "nsweeps": nsweeps,
            "atom_mean_e_per_atom": atom_stats.get("mean_e_per_atom"),
            "fast_mean_e_per_atom": fast_stats.get("mean_e_per_atom"),
            "abs_mean_e_per_atom_diff": de,
            "energy_tolerance": tol_e,
            "max_abs_mean_composition_diff": dx,
            "composition_tolerance": tol_x,
            "atom_verdict": atom.get("verdict") if atom else "MISSING",
            "fast_verdict": fast.get("verdict") if fast else "MISSING",
            "statistical_verdict": verdict_stat,
        }
        out.append(row)
        if verdict_stat in {"PASS", "WARN"}:
            aggregate[case].append(row)
    write_csv(RESULTS / "parallel_statistical_consistency_summary.csv", out)

    aggregate_rows = []
    for case, vals in sorted(aggregate.items()):
        de_vals = [row["abs_mean_e_per_atom_diff"] for row in vals if row["abs_mean_e_per_atom_diff"] is not None]
        dx_vals = [row["max_abs_mean_composition_diff"] for row in vals if row["max_abs_mean_composition_diff"] is not None]
        aggregate_rows.append(
            {
                "case": case,
                "n_seeds": len(vals),
                "mean_abs_e_per_atom_diff": mean(de_vals),
                "max_abs_e_per_atom_diff": max(de_vals) if de_vals else None,
                "mean_abs_composition_diff": mean(dx_vals),
                "max_abs_composition_diff": max(dx_vals) if dx_vals else None,
                "aggregate_verdict": "PASS" if vals and all(row["statistical_verdict"] == "PASS" for row in vals) else "WARN",
            }
        )
    write_csv(RESULTS / "parallel_statistical_consistency_aggregate.csv", aggregate_rows)

    md = ["# Parallel Statistical Consistency Summary", ""]
    md.append("Parallel runs are compared statistically because proposal ownership and random-number consumption can differ.")
    md.append("")
    md.append("| case | seed | ranks | abs d<E/N> | tol E | max abs d<x> | tol x | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in out:
        de_s = "" if row["abs_mean_e_per_atom_diff"] is None else f"{row['abs_mean_e_per_atom_diff']:.3g}"
        te_s = "" if row["energy_tolerance"] is None else f"{row['energy_tolerance']:.3g}"
        dx_s = "" if row["max_abs_mean_composition_diff"] is None else f"{row['max_abs_mean_composition_diff']:.3g}"
        tx_s = "" if row["composition_tolerance"] is None else f"{row['composition_tolerance']:.3g}"
        md.append(f"| {row['case']} | {row['seed']} | {row['ranks']} | {de_s} | {te_s} | {dx_s} | {tx_s} | {row['statistical_verdict']} |")
    md.append("")
    (RESULTS / "parallel_statistical_consistency_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_boundary_guards(rows: list[dict]) -> None:
    boundary = [row for row in rows if row.get("analysis_role") == "expected_rejection"]
    if not boundary:
        return
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in boundary:
        grouped[key_for(row)][row["kind"]] = row
    out = []
    for key, pair in sorted(grouped.items()):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        ok = atom and fast and atom.get("verdict") == "PASS" and fast.get("verdict") == "PASS_EXPECTED_REJECT"
        out.append(
            {
                "case": case,
                "natoms": natoms,
                "expected_error_pattern": fast.get("expected_error_pattern") if fast else None,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "guard_verdict": "PASS" if ok else "FAIL",
                "notes": fast.get("notes") if fast else "",
            }
        )
    write_csv(RESULTS / "boundary_guard_summary.csv", out)


def build_performance_summary(rows: list[dict]) -> list[dict]:
    perf = [row for row in rows if row.get("analysis_role", "performance") == "performance"]
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in perf:
        grouped[key_for(row)][row["kind"]] = row

    summary_rows = []
    for key, pair in sorted(grouped.items()):
        suite, case, ranks, natoms, box_lattice, nsweeps, seed = key
        atom = pair.get("atom")
        fast = pair.get("fast")
        atom_t = atom.get("time_per_sweep_s") if atom else None
        fast_t = fast.get("time_per_sweep_s") if fast else None
        speedup = atom_t / fast_t if atom_t and fast_t and fast_t > 0 else None
        summary_rows.append(
            {
                "suite": suite,
                "case": case,
                "ranks": ranks,
                "natoms": natoms,
                "box_lattice": "x".join(map(str, box_lattice)),
                "nsweeps": nsweeps,
                "atom_time_per_sweep_s": atom_t,
                "fast_time_per_sweep_s": fast_t,
                "speedup_atom_over_fast": speedup,
                "atom_verdict": atom.get("verdict") if atom else "MISSING",
                "fast_verdict": fast.get("verdict") if fast else "MISSING",
                "fast_fallback_detected": fast.get("fallback_detected") if fast else None,
            }
        )
    return summary_rows


def main() -> None:
    status_path = RESULTS / "run_status.json"
    if not status_path.exists():
        raise SystemExit(f"Missing {status_path}; run scripts/run_workflow.py first.")
    rows = json.loads(status_path.read_text(encoding="utf-8"))
    for row in rows:
        row["verdict"] = verdict(row)

    flat_csv = RESULTS / "run_status_flat.csv"
    fieldnames = [
        "id",
        "suite",
        "case",
        "description",
        "kind",
        "ranks",
        "natoms",
        "box_lattice",
        "nsweeps",
        "ncycle",
        "seed",
        "analysis_role",
        "expected",
        "returncode",
        "verdict",
        "loop_s",
        "time_per_sweep_s",
        "fallback_detected",
        "log",
    ]
    with flat_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    summary_rows = build_performance_summary(rows)
    write_csv(RESULTS / "performance_summary.csv", summary_rows)

    md = ["# Performance Summary", ""]
    md.append("Each row compares the original global-energy `atom/swap` path with `fast/swap` for the same performance case.")
    md.append("Validation and expected-rejection tests are reported in their own summary files.")
    md.append("")
    md.append("| suite | case | ranks | atoms | sweeps | atom/sweep (s) | fast/sweep (s) | speedup | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        speedup = row["speedup_atom_over_fast"]
        speed = f"{speedup:.6g}" if speedup is not None else ""
        atom_t = f"{row['atom_time_per_sweep_s']:.6g}" if row["atom_time_per_sweep_s"] is not None else ""
        fast_t = f"{row['fast_time_per_sweep_s']:.6g}" if row["fast_time_per_sweep_s"] is not None else ""
        verdicts = f"{row['atom_verdict']}/{row['fast_verdict']}"
        md.append(
            f"| {row['suite']} | {row['case']} | {row['ranks']} | {row['natoms']} | {row['nsweeps']} | "
            f"{atom_t} | {fast_t} | {speed} | {verdicts} |"
        )
    md.append("")
    failed = [row for row in rows if row["verdict"] not in {"PASS", "PASS_EXPECTED_REJECT"}]
    if failed:
        md.append("## Failed Or Excluded Runs")
        md.append("")
        for row in failed:
            md.append(f"- `{row['id']}`: {row['verdict']}; log `{row.get('log')}`")
            for err in row.get("errors", []):
                md.append(f"  - {err}")
            for warn in row.get("warnings", []):
                if "parallel MC disabled" in warn or "Safe interior" in warn:
                    md.append(f"  - {warn}")
        md.append("")
    (RESULTS / "performance_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_strong_scaling(rows)
    write_size_scaling(rows)
    write_unequal_cutoff_scaling(rows)
    write_nonlj_parallel_scaling(rows)
    write_exact_consistency(rows)
    write_parallel_stat_consistency(rows)
    write_boundary_guards(rows)
    write_timeseries_exports(rows)
    write_timing_repeats(rows)

    print(f"Wrote {flat_csv}")
    print(f"Wrote {RESULTS / 'performance_summary.csv'}")
    print(f"Wrote {RESULTS / 'performance_summary.md'}")
    for name in (
        "strong_scaling_summary.csv",
        "size_scaling_summary.csv",
        "unequal_cutoff_scaling_summary.csv",
        "unequal_cutoff_scaling_summary.md",
        "nonlj_parallel_scaling_summary.csv",
        "nonlj_parallel_scaling_summary.md",
        "exact_consistency_summary.csv",
        "parallel_statistical_consistency_summary.csv",
        "boundary_guard_summary.csv",
        "timeseries_manifest.csv",
        "timing_repeats_raw.csv",
        "timing_repeats_summary.csv",
    ):
        if (RESULTS / name).exists():
            print(f"Wrote {RESULTS / name}")


if __name__ == "__main__":
    main()
