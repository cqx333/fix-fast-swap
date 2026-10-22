"""Export raw trajectories, fluctuation diagnostics and provenance-aware timings."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
from scipy.stats import t as student_t
from legacy_common import correlated
from log_data import parse_thermo
from workflow import load, save, sha


def csv_out(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["no_completed_rows"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def samples(record, log, family):
    raw = parse_thermo(log.read_text(errors="replace"))
    if not raw:
        raise ValueError("No production thermo records: " + str(log))
    first = raw[0]["f_mc[1]"]
    rows = []
    for row in raw:
        if family == "mdmc" and int(row["Step"]) % 100:
            continue
        value = {"sweeps": record.get("start_sweep", 0) + (row["f_mc[1]"] - first) / record["natoms"],
                 "local_md_step": row["Step"], "U_total": row["PotEng"], "U_per_atom": row["v_epa"],
                 "attempts": row["f_mc[1]"], "accepts": row["f_mc[2]"], "natoms": record["natoms"]}
        for i in range(1, record["ntypes"] + 1):
            value["x_type_" + str(i)] = row[f"c_ctype[{i}]"] / record["natoms"]
            value["n_type_" + str(i)] = row[f"c_ctype[{i}]"]
        for key in ("Temp", "Press", "Volume", "c_rmin"):
            if key in row:
                value[key] = row[key]
        rows.append(value)
    return rows


def bootstrap_variances(chains, width, draws=800, stream=0):
    rng = np.random.default_rng(20261020 + width + stream)
    result = np.zeros(draws)
    for x in chains:
        for start in range(0, draws, 100):
            count = min(100, draws - start)
            indices = (rng.integers(0, len(x), size=(count, math.ceil(len(x) / width), 1)) + np.arange(width)) % len(x)
            values = x[indices.reshape(count, -1)[:, :len(x)]]
            result[start:start + count] += values.var(axis=1, ddof=1) / len(chains)
    return result


def compare(chains, case, ranks, family, ntypes, n):
    rows, flags, details = [], [], []
    seeds = sorted(set(seed for kind, seed in chains if kind == "fast") & set(seed for kind, seed in chains if kind == "atom"))
    if len(seeds) < 3:
        return rows, [f"Incomplete statistical replica set: {family}/{case}/p{ranks}"], details
    for discard in (.25, .5):
        for obs in ["U_per_atom"] + ["x_type_" + str(i) for i in range(2, ntypes + 1)]:
            data, stats = {}, {}
            for kind in ("atom", "fast"):
                data[kind], stats[kind] = [], []
                for seed in seeds:
                    values = chains[kind, seed]
                    limit = values[-1]["sweeps"] * discard
                    x = np.array([r[obs] for r in values if r["sweeps"] >= limit])
                    st = correlated(x)
                    if st["se"] is None:
                        raise ValueError("Too few samples for uncertainty analysis")
                    data[kind].append(x)
                    stats[kind].append(st)
                    a, b = correlated(x[:len(x)//2]), correlated(x[len(x)//2:])
                    drift = b["mean"] - a["mean"]
                    half = float(student_t.ppf(.975, max(1, min(a["df"], b["df"]))) * math.hypot(a["se"] or 0, b["se"] or 0))
                    details.append({"family": family, "case": case, "ranks": ranks, "natoms": n, "kind": kind,
                                    "seed": seed, "discard": discard, "observable": obs, **st,
                                    "variance": float(x.var(ddof=1)), "samples": len(x),
                                    "tail_drift": drift, "drift_ci95_halfwidth": half,
                                    "drift_review": abs(drift) > half})
            differences = np.array([b["mean"] - a["mean"] for a, b in zip(stats["atom"], stats["fast"])])
            within = math.sqrt(sum(s["se"]**2 for kind in stats for s in stats[kind])) / len(seeds)
            between = float(differences.std(ddof=1) / math.sqrt(len(seeds)))
            half = float(student_t.ppf(.975, len(seeds)-1) * max(within, between))
            delta = float(differences.mean())
            row = {"family": family, "case": case, "ranks": ranks, "natoms": n, "discard": discard,
                   "observable": obs, "replicas": len(seeds), "mean_difference_fast_minus_atom": delta,
                   "mean_ci95_halfwidth": half, "atom_count_equivalent_mean_difference": delta*n if obs.startswith("x_type") else "",
                   "minimum_ESS": min(s["ess"] for kind in stats for s in stats[kind]),
                   "reference_status": "historical atom reference; fluctuation discrepancy requires review, not automatic fast failure"}
            if case == "coul_dsf" and obs == "U_per_atom":
                # Fixed equal charges and frozen coordinates make this energy type-independent.
                scale = max(1., *(abs(s["mean"]) for kind in stats for s in stats[kind]))
                floor = 128 * np.finfo(float).eps * scale
                expected_constant = all(s["constant"] for kind in stats for s in stats[kind]) and abs(delta) <= floor
                row["expected_type_independent_constant_energy"] = expected_constant
                if expected_constant:
                    row["variance_ratio_fast_over_atom"] = None
                    row["interpretation"] = "Constant fixed-charge energy; variance ratio and energy ESS are not informative"
                    rows.append(row)
                    continue
            if abs(delta) > half:
                flags.append(f"Mean difference needs review: {family}/{case}/p{ranks}/{obs}/discard{discard}")
            if row["minimum_ESS"] < 50:
                flags.append(f"Low ESS: {family}/{case}/p{ranks}/{obs}/discard{discard}")
            va = float(np.mean([x.var(ddof=1) for x in data["atom"]]))
            vb = float(np.mean([x.var(ddof=1) for x in data["fast"]]))
            row.update(atom_variance=va, fast_variance=vb, variance_ratio_fast_over_atom=vb/va if va else None)
            for width in (8, 32):
                aa, bb = bootstrap_variances(data["atom"], width), bootstrap_variances(data["fast"], width, stream=10000)
                valid = aa > 0
                if valid.sum() >= 700:
                    lo, hi = map(float, np.quantile(bb[valid] / aa[valid], [.025, .975]))
                    row[f"variance_ratio_ci95_block{width}_low"] = lo
                    row[f"variance_ratio_ci95_block{width}_high"] = hi
                    if not lo <= 1 <= hi:
                        flags.append(f"Variance difference needs reference review: {family}/{case}/p{ranks}/{obs}/discard{discard}/block{width}")
                else:
                    row[f"variance_ratio_ci95_block{width}_low"] = None
                    row[f"variance_ratio_ci95_block{width}_high"] = None
                    flags.append(f"Near-zero reference variance: {family}/{case}/p{ranks}/{obs}")
            rows.append(row)
    return rows, flags, details


def analyze(output):
    output = Path(output)
    manifest = load(ROOT / "manifest.json")
    status = load(output / "run_status.json") if (output / "run_status.json").exists() else {}
    references = load(ROOT / "baseline/records.json")
    complete = {k: v for k, v in status.items() if v.get("state") == "complete"}
    out = output / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    invalid = []
    for key, record in list(complete.items()):
        if any(not (ROOT / p).is_file() or sha(ROOT / p) != h for p, h in record["output_sha256"].items()):
            invalid.append(key)
            del complete[key]
    chains = defaultdict(lambda: defaultdict(list))
    specs = {}
    seen_reference_keys = set()
    raw_timing = []
    for task in manifest["tasks"]:
        if task["id"] not in complete:
            continue
        record = complete[task["id"]]
        if task["family"] in ("timing", "size3d"):
            reference = references[task["reference_keys"][0]]
            raw_timing.append({"id": task["id"], "family": task["family"], "case": task["case"], "ranks": task["ranks"],
                "natoms": task["natoms"], "box": "x".join(map(str, task["box"])), "repeat": task["repeat"],
                "fast_time_per_sweep_s": record["time_per_actual_sweep_s"],
                "atom_time_per_sweep_s": reference["time_per_actual_sweep_s"],
                "speedup_paired_repeat": reference["time_per_actual_sweep_s"] / record["time_per_actual_sweep_s"],
                "fast_loop_s": record["loop_s"], "atom_loop_s": reference["loop_s"],
                "fast_acceptance_fraction": record["acceptance_fraction"],
                "atom_acceptance_fraction": reference["acceptance_fraction"],
                "reference_reused": True, "fast_host": record["source_host"],
                "reference_host": reference.get("source_host", "node05"),
                "initial_state_sha256": reference.get("initial_data_sha256"),
                "fast_log": record["log"], "atom_log": reference["archived_log"]})
            continue
        condition = (task["family"], task["case"], task["ranks"], tuple(task["box"]))
        specs[condition] = task
        values = samples(task, ROOT / record["log"], task["family"])
        chains[condition]["fast", task["seed"]] += values
        for refkey in task["reference_keys"]:
            if refkey in seen_reference_keys:
                continue
            seen_reference_keys.add(refkey)
            ref = references[refkey]
            chains[condition]["atom", task["seed"]] += samples(ref, ROOT / ref["archived_log"], task["family"])

    comparisons, flags, details = [], [], []
    for condition, by_seed in chains.items():
        family, case, ranks, box = condition
        task = specs[condition]
        # Never compare a short, partially completed chain to a full reference.
        expected = 600 if family == "mdmc" else task["requested_sweeps"]
        keep = {}
        for (kind, seed), values in by_seed.items():
            values = sorted({r["sweeps"]: r for r in values}.values(), key=lambda r: r["sweeps"])
            name = f"{family}__{case}__p{ranks}__s{seed}__{kind}"
            csv_out(out / "timeseries" / (name + ".csv"), values)
            if values[-1]["sweeps"] >= expected:
                keep[kind, seed] = values
        rows, warnings, diag = compare(keep, case, ranks, family, task["ntypes"], task["natoms"])
        comparisons += rows
        flags += warnings
        details += diag

    groups = defaultdict(list)
    for row in raw_timing:
        groups[tuple(row[k] for k in ("family", "case", "ranks", "natoms", "box"))].append(row)
    timing = []
    for key, rows in groups.items():
        family, case, ranks, n, box = key
        aa = np.array([r["atom_time_per_sweep_s"] for r in rows])
        bb = np.array([r["fast_time_per_sweep_s"] for r in rows])
        item = dict(family=family, case=case, ranks=ranks, natoms=n, box=box, repeats=len(rows),
                    atom_mean_time_per_sweep_s=float(aa.mean()), fast_mean_time_per_sweep_s=float(bb.mean()),
                    same_rank_speedup_ratio_of_means=float(aa.mean()/bb.mean()))
        if len(rows) > 1:
            item.update(fast_time_sd=float(bb.std(ddof=1)), fast_time_CV=float(bb.std(ddof=1)/bb.mean()),
                        fast_time_ci95_halfwidth=float(student_t.ppf(.975, len(rows)-1)*bb.std(ddof=1)/math.sqrt(len(rows))))
        timing.append(item)
    csv_out(out / "timing_raw.csv", raw_timing)
    csv_out(out / "timing_summary.csv", timing)
    csv_out(out / "statistical_comparisons.csv", comparisons)
    csv_out(out / "replica_diagnostics.csv", details)
    missing = [t["id"] for t in manifest["tasks"] if t["id"] not in complete]
    summary = {"completed_valid_new_tasks": len(complete), "planned_new_tasks": len(manifest["tasks"]),
        "missing_or_failed": missing, "hash_invalid_tasks": invalid, "statistical_review_flags": sorted(set(flags)),
        "decision": "INCOMPLETE" if missing else "REVIEW",
        "reference_fluctuation_discrepancy_already_known": True,
        "no_automatic_ensemble_equivalence_claim": True,
        "old_parallel_fast_samples_never_pooled": True,
        "reused_rng_regression_summary": "reused_rng_regression/regression_summary.json",
        "note": "REVIEW is deliberate even if pairwise flags are empty: finite tests and old-reference uncertainty require scientific review."}
    save(out / "summary.json", summary)
    print("Analysis:", out, "valid completed:", len(complete), "missing:", len(missing), "review flags:", len(set(flags)), flush=True)
    return summary


if __name__ == "__main__":
    analyze(ROOT / "results")
