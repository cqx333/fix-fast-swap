"""Reproducible phase-application analysis without modifying simulation output."""
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT.parent
RAW = WORK / "CPC-test_revision_unified_20261012/results"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
(DATA / "snapshots").mkdir(exist_ok=True)
module_path = HERE / "workflow_phase_diagram.py"
spec = importlib.util.spec_from_file_location("application_workflow", module_path)
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)
TEMPS = [.59, .60, .61, .62, .63, .64]
FIELDS = ["md_step", "N", "n_B", "c", "U_per_atom", "H_per_atom", "temperature", "pressure", "volume",
          "mean_csp", "ordered_fraction", "attempts_increment", "accepts_increment"]
SEEDS = [2145927, 3719241, 5817329]
CACHED = "--cached-only" in sys.argv
CACHE_SOURCES = json.loads((DATA / "inputs/case_sources.json").read_text(encoding="utf-8")) if CACHED else {}


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DATA / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dump(name, value):
    (DATA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source(T, case):
    temp = f"T{T:.2f}"
    for root in (RAW / "rich_lowB_20261016" / temp, RAW / temp):
        path = root / "trajectories" / case
        if path.exists():
            return path
    raise FileNotFoundError(case)


def stats_for(case, a, stage, fraction):
    d = {key: a[:, i] for i, key in enumerate(FIELDS)}
    use = d["md_step"] > (1-fraction) * d["md_step"][-1]
    mc = use & (stage == 1)
    sc = wf.correlated_stats(d["c"][mc])
    su = wf.correlated_stats(d["U_per_atom"][use])
    ids = ((d["md_step"][use] - 1) // 40000).astype(int)
    cycles = np.array([d["c"][use][ids == v].mean() for v in np.unique(ids)])
    cs = wf.correlated_stats(cycles)
    sem = max(sc["sem"], cs["sem"])
    df = min(sc["df"], cs["df"])
    early = wf.correlated_stats(d["c"][mc][:int(mc.sum())//2])
    late = wf.correlated_stats(d["c"][mc][int(mc.sum())//2:])
    delta, half, _ = wf.equivalence((late["mean"], late["sem"], late["df"]),
                                  (early["mean"], early["sem"], early["df"]), 1)
    accepts, attempts = int(d["accepts_increment"][use].sum()), int(d["attempts_increment"][use].sum())
    result = {"case": case["id"], "T": case["T"], "phase": case["phase"], "mu": case["mu"],
              "start": case["start"], "seed": case["seed"], "fraction": fraction, "N": 6912,
              "mean_c": float(d["c"][use].mean()), "sem_c": sem, "df_c": df,
              "ci95_c": float(student_t.ppf(.975, df)*sem), "mean_MC_only_c": sc["mean"],
              "mc_ESS": sc["n_effective"], "cycle_ESS": cs["n_effective"],
              "mean_U": su["mean"], "sem_U": su["sem"], "mean_CSP": float(d["mean_csp"][use].mean()),
              "mean_ordered_fraction": float(d["ordered_fraction"][use].mean()),
              "mean_T": float(d["temperature"][use].mean()), "mean_P": float(d["pressure"][use].mean()),
              "attempts": attempts, "accepts": accepts, "acceptance": accepts/attempts,
              "half_tail_drift_B": delta*6912, "half_tail_drift_ci90_B": half*6912,
              "cycle_block_width": max(1, math.ceil(max(cs["tau"], sc["tau"]/20)))}
    return result, cycles


trajectories, windows, inventory, integrity = [], [], [], []
trace, cycle_data, saved_states = {}, {}, []
for T in TEMPS:
    merged = DATA / "inputs" / f"T{T:.2f}" if CACHED else RAW / ("rich_lowB_20261016/" if T >= .61 else "") / f"T{T:.2f}"
    config = json.loads((merged / "effective_config.json").read_text(encoding="utf-8"))
    cfg_states = {wf.state_id(s["phase"], T, s["mu"]): s for s in config["states"]}
    supplied = read(merged / "analysis/trajectory_statistics.csv")
    flags = {r["state"]: r for r in read(merged / "analysis/state_convergence.csv")}
    replaced = json.loads((merged / "rich_repair.json").read_text(encoding="utf-8"))["replacement_case_ids"] if T >= .61 else []
    for s in supplied:
        case_id = s["case"]
        folder = Path(CACHE_SOURCES[case_id]) if CACHED else source(T, case_id)
        case_path = DATA / "inputs/cases" / (case_id+".json") if CACHED else folder / "case.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["id"] = case_id
        cache = DATA / "snapshots" / (case_id + ".npz")
        if cache.exists():
            z = np.load(cache)
            a, stages = z["values"], z["stage"]
        else:
            if CACHED:
                raise FileNotFoundError("Missing cached observations: " + str(cache))
            rows = [r for c in sorted(folder.glob("cycle_*")) for r in read(c / "snapshots.csv")]
            a = np.array([[float(r[k]) for k in FIELDS] for r in rows], dtype=float)
            stages = np.array([r["stage"] == "mc" for r in rows], dtype=np.uint8)
            np.savez_compressed(cache, values=a, stage=stages)
        if len(a) != 2400 or not np.all(np.diff(a[:, 0]) > 0) or int(a[-1, 0]) != 2400000:
            integrity.append({"case": case_id, "problem": "snapshot schedule"})
        if not np.all(np.isfinite(a)) or np.max(abs(a[:, 3]-a[:, 2]/a[:, 1])) > 1e-12:
            integrity.append({"case": case_id, "problem": "numeric data"})
        old_stats = {}
        for fraction in (.2, .3, .5):
            result, cycles = stats_for(case, a, stages, fraction)
            result["is_replacement"] = case_id in replaced
            result["actual_initial_c"] = float(s["actual_c_start"])
            result["source"] = str(folder)
            windows.append(result)
            if fraction == .2:
                if abs(result["mean_c"]-float(s["mean_c"])) > 1e-12:
                    integrity.append({"case": case_id, "problem": "provided composition differs"})
                trajectories.append(result)
                cycle_data[case_id] = cycles
                old_stats = result
        trace[case_id] = (case, a, stages)
    for sid, state in cfg_states.items():
        group = [r for r in trajectories if r["T"] == T and r["phase"] == state["phase"] and r["mu"] == state["mu"]]
        saved_states.append({"T": T, "phase": state["phase"], "mu": state["mu"], "state": sid,
                             "starts": sorted({r["start"] for r in group}), "replicas": len(group),
                             "old_gate_passed": flags[sid]["statistical_gate_passed"],
                             "old_ESS_flag": flags[sid]["enough_effective_samples"],
                             "old_drift_flag": flags[sid]["tail_drift_equivalent"],
                             "legacy_c": state.get("legacy_c"), "initial_compositions": state["initial_compositions"]})
    inventory.append({"T": T, "trajectories": len(supplied), "replacement": len(replaced),
                      "mu": sorted({r["mu"] for r in trajectories if r["T"] == T}),
                      "source": str(merged)})
    print("Read temperature", T, "trajectories", len(supplied), flush=True)

save("trajectory_statistics.csv", trajectories)
save("window_statistics.csv", windows)
dump("snapshot_schema.json", {"columns": FIELDS, "stage_1": "post-NVT MC block", "stage_0": "NPT snapshot",
                              "energy_units": "epsilon_AA per atom", "T_units": "epsilon_AA/k_B"})
dump("state_inventory.json", saved_states)
dump("inventory.json", {"temperatures": inventory, "issues": integrity})
assert not integrity, integrity

# Reference enthalpies are independent pure-A NPT trajectories.
reference, reference_series, href = [], {}, {}
reference_inputs = read(DATA / "reference_enthalpy.csv") if CACHED else [{"source": str(p)} for p in sorted((RAW / "T0.59/reference").iterdir()) if p.is_dir()]
for entry in reference_inputs:
    folder = Path(entry["source"])
    if CACHED:
        phase, T, seed = entry["phase"], float(entry["T"]), int(entry["seed"])
    else:
        phase, temperature, seed = folder.name.split("_")
        T, seed = float(temperature[1:]), int(seed[4:])
    reference_id = f"{phase}_T{T:.2f}_seed{seed}"
    if CACHED:
        values = np.load(DATA / "snapshots" / (reference_id+"_enthalpy.npz"))["H"]
    else:
        status = json.loads((folder / "status.json").read_text())
        assert status["state"] == "complete"
        for name, digest in status["sha256"].items():
            assert wf.sha(folder / name) == digest, (folder, name)
        reference_rows = read(folder / "snapshots.csv")
        assert all(int(r["n_B"]) == 0 for r in reference_rows)
        values = np.array([float(r["H_per_atom"]) for r in reference_rows])
    reference_series[phase, T, seed] = values
    st = wf.correlated_stats(values)
    reference.append({"phase": phase, "T": T, "seed": seed, "H": st["mean"], "sem_H": st["sem"], "df_H": st["df"],
                      "ESS": st["n_effective"], "source": str(folder), "block_width": math.ceil(st["tau"])})
    np.savez_compressed(DATA / "snapshots" / (reference_id + "_enthalpy.npz"), H=values)
assert len(reference) == 42
ref_points = []
for phase in ("solid", "liquid"):
    for T in np.arange(59, 66)/100:
        g = [r for r in reference if r["phase"] == phase and r["T"] == T]
        mean, sem, df = wf.aggregate_replicas(g, "H", "sem_H", "df_H")
        ref_points.append({"phase": phase, "T": T, "H": mean, "sem_H": sem, "df_H": df,
                           "ci95_H": float(student_t.ppf(.975, df)*sem)})
    g = [r for r in ref_points if r["phase"] == phase]
    href[phase] = np.polynomial.Polynomial.fit([r["T"] for r in g], [r["H"] for r in g], 1).convert().coef
save("reference_enthalpy.csv", reference)
save("enthalpy_mean_points.csv", ref_points)


def state_means(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["phase"], r["mu"], r["start"]].append(r)
    starts = defaultdict(dict)
    for (phase, mu, start), group in groups.items():
        starts[phase, mu][start] = wf.aggregate_replicas(group)
    return starts


def fits_for(T, rows, degree=3, start=None, only_gate=False):
    means = state_means(rows)
    fits = {}
    for phase in ("solid", "liquid"):
        points = []
        for (p, mu), starts in means.items():
            if p != phase or (start is not None and start not in starts):
                continue
            state = next(r for r in saved_states if r["T"] == T and r["phase"] == p and r["mu"] == mu)
            if only_gate and state["old_gate_passed"] != "True":
                continue
            c = starts[start][0] if start else float(np.mean([v[0] for v in starts.values()]))
            points.append((c, mu))
        fits[phase] = wf.fit_mu(T, *zip(*sorted(points)), degree)
    return fits


def offset(T, coeffs):
    result = 0.0
    for power, coefficient in enumerate(coeffs):
        if power == 0:
            result += coefficient*(1-T/.65)
        elif power == 1:
            result += coefficient*T*math.log(.65/T)
        else:
            result += coefficient*T*(.65**(power-1)-T**(power-1))/(power-1)
    return float(result)


boundaries, fit_models, sensitivity, comparisons, state_table = [], {}, [], [], []
delta_h = href["liquid"] - href["solid"]
for T in TEMPS:
    rows = [r for r in trajectories if r["T"] == T]
    fits = fits_for(T, rows)
    off = offset(T, delta_h)
    b = wf.common_tangent(T, fits["solid"], fits["liquid"], off)
    assert b["endpoint_coverage"], (T, b)
    boundaries.append({"T": T, "offset": off, **b})
    fit_models[str(T)] = fits
    means = state_means(rows)
    for (phase, mu), starts in sorted(means.items()):
        g = [r for r in rows if r["phase"] == phase and r["mu"] == mu]
        state_table.append({"T": T, "phase": phase, "mu": mu, "mean_c": float(np.mean([v[0] for v in starts.values()])),
                            "poor_mean_c": starts["poor"][0], "rich_mean_c": starts["rich"][0],
                            "pooled_acceptance": sum(r["accepts"] for r in g)/sum(r["attempts"] for r in g),
                            "minimum_MC_ESS": min(r["mc_ESS"] for r in g), "replicas": len(g)})
        for left, right in (("poor", "rich"), ("poor", "middle"), ("middle", "rich")):
            if left not in starts or right not in starts:
                continue
            a, z = starts[left], starts[right]
            delta, half, _ = wf.equivalence(a, z, 1)
            var = a[1]**2+z[1]**2
            denom = a[1]**4/a[2]+z[1]**4/z[2]
            df = var**2/denom if denom else 1
            half95 = float(student_t.ppf(.975, max(df, 1))*math.sqrt(var))
            comparisons.append({"T": T, "phase": phase, "mu": mu, "left": left, "right": right,
                                "difference_B": delta*6912, "ci95_half_B": half95*6912,
                                "zero_in_ci95": abs(delta) <= half95})
    variants = [("quadratic_excess", rows, 2, None, False), ("quartic_excess", rows, 4, None, False),
                ("poor_start_only", rows, 3, "poor", False), ("rich_start_only", rows, 3, "rich", False),
                ("automatic_gate", rows, 3, None, True),
                ("tail30", [r for r in windows if r["T"] == T and r["fraction"] == .3], 3, None, False),
                ("tail50", [r for r in windows if r["T"] == T and r["fraction"] == .5], 3, None, False)]
    for label, rr, degree, start, gate in variants:
        result = {"T": T, "variant": label}
        try:
            ff = fits_for(T, rr, degree, start, gate)
            result.update(wf.common_tangent(T, ff["solid"], ff["liquid"], off))
            result.update(delta_cs_B=(result["c_s"]-b["c_s"])*6912, delta_cl_B=(result["c_l"]-b["c_l"])*6912)
        except ValueError as error:
            result["error"] = str(error)
        sensitivity.append(result)
    quad_h = {}
    for phase in ("solid", "liquid"):
        g = [r for r in ref_points if r["phase"] == phase]
        quad_h[phase] = np.polynomial.Polynomial.fit([r["T"] for r in g], [r["H"] for r in g], 2).convert().coef
    result = {"T": T, "variant": "quadratic_enthalpy"}
    result.update(wf.common_tangent(T, fits["solid"], fits["liquid"], offset(T, quad_h["liquid"]-quad_h["solid"])))
    result.update(delta_cs_B=(result["c_s"]-b["c_s"])*6912, delta_cl_B=(result["c_l"]-b["c_l"])*6912)
    sensitivity.append(result)
save("state_composition_and_acceptance.csv", state_table)
save("initial_state_comparison.csv", comparisons)
save("fit_sensitivity.csv", sensitivity)
dump("fitted_models.json", fit_models)

# Seed clusters are resampled jointly; moving cycle blocks preserve NPT holds.
rng = np.random.default_rng(20261018)
reference_lookup = {(r["phase"], r["T"], r["seed"]): r for r in reference}
grouped = defaultdict(list)
for r in trajectories:
    grouped[r["T"], r["phase"], r["mu"], r["start"]].append(r)
samples, failures = [], []
for replicate in range(500):
    seeds = rng.choice(SEEDS, len(SEEDS), replace=True).tolist()
    boot_h = {}
    for phase in ("solid", "liquid"):
        points = []
        for T in np.arange(59, 66)/100:
            hs = []
            for seed in seeds:
                ref = reference_lookup[phase, T, seed]
                hs.append(wf.block_resample(reference_series[phase, T, seed], ref["block_width"], rng).mean())
            points.append(np.mean(hs))
        boot_h[phase] = np.polynomial.Polynomial.fit(np.arange(59, 66)/100, points, 1).convert().coef
    for T in TEMPS:
        by_state = defaultdict(list)
        for (temp, phase, mu, start), group in grouped.items():
            if temp != T:
                continue
            available = {r["seed"]: r for r in group}
            selected = [available[s] for s in seeds if s in available] if len(group) > 1 else group
            if not selected:
                selected = [group[int(rng.integers(len(group)))]]
            values = [wf.block_resample(cycle_data[r["case"]], r["cycle_block_width"], rng).mean() for r in selected]
            by_state[phase, mu].append(float(np.mean(values)))
        try:
            ff = {}
            for phase in ("solid", "liquid"):
                points = [(np.mean(v), mu) for (p, mu), v in by_state.items() if p == phase]
                ff[phase] = wf.fit_mu(T, *zip(*points), 3)
            result = wf.common_tangent(T, ff["solid"], ff["liquid"], offset(T, boot_h["liquid"]-boot_h["solid"]))
            if not result["endpoint_coverage"]:
                raise ValueError("endpoint outside resampled range")
            samples.append({"replicate": replicate, "T": T, **result})
        except ValueError as error:
            failures.append({"replicate": replicate, "T": T, "reason": str(error)})
    if (replicate+1) % 50 == 0:
        print("Bootstrap", replicate+1, "/500", "failed temperature replicates", len(failures), flush=True)
save("bootstrap_samples.csv", samples)
save("bootstrap_failures.csv", failures)
for b in boundaries:
    values = [r for r in samples if r["T"] == b["T"]]
    b["valid_bootstraps"] = len(values)
    for field in ("c_s", "c_l", "mu_eq"):
        lo, hi = np.quantile([r[field] for r in values], [.025, .975])
        b[field+"_lo"], b[field+"_hi"] = float(lo), float(hi)
save("phase_boundaries.csv", boundaries)
dump("analysis_summary.json", {"trajectories": len(trajectories), "states": len(state_table),
     "repairs": sum(r["is_replacement"] for r in trajectories), "integrity_issues": integrity,
     "bootstrap_replicates": 500, "bootstrap_failed_temperature_replicates": len(failures),
     "reference_runs": len(reference), "enthalpy_coefficients": {k: v.tolist() for k, v in href.items()},
     "minimum_MC_half_domain_margin": min(float((a[stage == 1, FIELDS.index('volume')]**(1/3)/4-3.8).min()) for _, a, stage in trace.values()),
     "uncertainty_scope": "Conditional on Tm=0.65, fixed pressure, retained window and fitted functional forms. No Tm, finite-size, or potential-model uncertainty included."})
print("Application numerical analysis complete", flush=True)
