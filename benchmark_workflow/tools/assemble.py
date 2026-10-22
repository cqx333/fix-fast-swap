"""Assemble a self-contained, fast-parallel-only recomputation package."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OLD = WORK / "CPC-test_revision_unified_20261012/results/collected_all_node05"
THREE = WORK / "CPC-test_3d_size_supplement_20261019"
WORKFLOW = WORK / "CPC-test_revision_unified_20261012"
FIXED = WORK / "results_rng_fixed_20261020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(source, target, expected=None):
    if expected and sha(source) != expected:
        raise ValueError("Historical file hash mismatch: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.relative_to(ROOT).as_posix()


def main():
    for name in ("models.py", "generate_revision.py", "log_data.py"):
        copy_file(WORKFLOW / "vendor" / name, ROOT / "vendor" / name)
    copy_file(THREE / "vendor/generate_revision.py", ROOT / "vendor/geometry3d.py")
    copy_file(WORKFLOW / "scripts/common.py", ROOT / "vendor/legacy_common.py")
    copy_file(WORKFLOW / "tables/pair_lj.table", ROOT / "tables/pair_lj.table")
    copy_file(WORKFLOW / "COPYING", ROOT / "COPYING")
    for name in ("fix_fast_swap.cpp", "fix_fast_swap.h"):
        copy_file(WORK / "fast_swap_rng_patch_20261020" / name, ROOT / "src/MC" / name)
    sys.path.insert(0, str(ROOT / "vendor"))
    import generate_revision as gen
    import geometry3d

    core = load(OLD / "results/run_status.json")
    extra = load(OLD / "incremental/run_status.json")
    three = load(THREE / "results/run_status.json")
    records = {}
    copied = {}

    def historical(dataset, entry, relative, target=None):
        source = {"core": OLD, "incremental": OLD, "three_d": THREE}[dataset]
        expected = entry.get("output_sha256", {}).get(relative)
        if relative == entry.get("log"):
            expected = expected or entry.get("log_sha256")
        if relative == entry.get("input"):
            expected = expected or entry.get("input_sha256")
        if not expected:
            raise ValueError("Missing historical digest: " + relative)
        key = dataset + ":" + relative
        if key not in copied:
            destination = ROOT / (target or ("baseline/" + dataset + "/" + relative))
            copied[key] = copy_file(source / relative, destination, expected)
        return copied[key]

    def keep(dataset, entry):
        key = dataset + ":" + entry["id"]
        if key in records:
            return key
        record = copy.deepcopy(entry)
        record["archived_log"] = historical(dataset, entry, entry["log"])
        record["reused_without_simulation"] = True
        record["reference_status"] = "historical evidence; not automatically a distribution gold standard"
        if entry.get("input"):
            record["archived_input"] = historical(dataset, entry, entry["input"])
        records[key] = record
        return key

    tasks = []
    excluded = []
    for entry in core.values():
        if entry["kind"] == "fast" and entry["ranks"] > 1 and entry["stage"] in ("parallel", "timing"):
            if entry["stage"] == "timing" and entry.get("group_membership") == ["size"]:
                excluded.append({"id": entry["id"], "reason": "superseded flat LJ size scan; use the 3D scan"})
                continue
            task = copy.deepcopy(entry)
            task["family"] = "parallel" if entry["stage"] == "parallel" else "timing"
            task["renderer"] = "legacy"
            task["reference_keys"] = [keep("core", core[entry["id"].rsplit("__", 1)[0] + "__atom"])]
            if entry["stage"] == "parallel":
                if entry["case"] in ("ternary_lj_cut", "zbl") and entry["ranks"] == 24:
                    task["requested_sweeps"] = 1200
                    task["reference_keys"] += [keep("incremental", v) for v in extra.values()
                        if v.get("previous_id") == task["reference_keys"][0].split(":", 1)[1]
                        and v.get("stage") == "extension"]
                task["type_dump_every"] = 25
            else:
                prep = core[entry["preparation_id"]]
                source = entry["initial_data_path"]
                task["initial_data_path"] = historical("core", prep, source, "baseline/assets/" + Path(source).name)
                task["asset_dependencies"] = [task["initial_data_path"]]
                task["minimum_loop_seconds"] = 30
            tasks.append(task)
        elif entry["ranks"] == 1 or entry["kind"] == "atom" or entry.get("expected_fallback"):
            keep("core", entry)
            excluded.append({"id": entry["id"], "reason": "unchanged atom/swap or rank-1/guard path; reuse"})

    for entry in three.values():
        if entry["stage"] == "timing" and entry["kind"] == "fast":
            task = copy.deepcopy(entry)
            task.update(family="size3d", renderer="three_d")
            task["reference_keys"] = [keep("three_d", three[entry["id"].rsplit("__", 1)[0] + "__atom"])]
            prep = three[entry["preparation_id"]]
            source = entry["initial_data_path"]
            task["initial_data_path"] = historical("three_d", prep, source, "baseline/assets/" + Path(source).name)
            task["asset_dependencies"] = [task["initial_data_path"]]
            tasks.append(task)
        elif entry["kind"] == "atom" or entry["ranks"] == 1:
            keep("three_d", entry)

    for entry in extra.values():
        if entry.get("stage") == "dynamic" and entry.get("kind") == "fast" and entry["ranks"] > 1:
            task = copy.deepcopy(entry)
            task.update(family="mdmc", renderer="mobile", requested_sweeps=entry["sweeps"])
            task["reference_keys"] = [keep("incremental", extra[entry["id"].replace("__fast__", "__atom__")])]
            source_input = OLD / "incremental/inputs" / ("in." + entry["id"])
            input_key = "incremental/inputs/" + source_input.name
            expected = entry["output_sha256"][input_key]
            copy_file(source_input, ROOT / "templates" / source_input.name, expected)
            task["template"] = "templates/" + source_input.name
            if entry["start_sweep"] == 0:
                pid = f"prepare__{entry['case']}__s{entry['seed']}"
                source = "incremental/raw/" + pid + ".restart"
                task["restart_input"] = historical("incremental", extra[pid], source, "baseline/assets/" + pid + ".restart")
                task["asset_dependencies"] = [task["restart_input"]]
                task["depends_on"] = []
            else:
                previous_id = entry["id"].rsplit("__from", 1)[0] + "__from" + str(entry["start_sweep"] - 100)
                task["depends_on"] = [previous_id]
                task["restart_input"] = "@RAW@/" + previous_id + ".restart"
            tasks.append(task)
        elif entry.get("ranks") == 1 or entry.get("kind") == "atom":
            keep("incremental", entry)

    removed_keys = ("command", "output_sha256", "state", "fingerprint", "pass_", "log", "input", "input_sha256",
                    "log_sha256", "elapsed_s", "adaptive_intervals", "calibration_file", "actual_attempts", "actual_sweeps",
                    "time_per_actual_sweep_s", "timer_breakdown", "loop_s", "accepted_moves", "acceptance_fraction")
    for task in tasks:
        command = task.get("command", [])
        task["historical_launcher"] = command[:command.index("-np")] if "-np" in command else ["mpirun"]
        for key in removed_keys:
            task.pop(key, None)
        task["kind"] = "fast"
        task["expect_parallel"] = True
        task.setdefault("asset_dependencies", [])
        task.setdefault("depends_on", [])
        task["initial_state_policy"] = "restart contaminated parallel chains from the original state, never append old fast samples"
    tasks.sort(key=lambda t: ({"parallel": 0, "mdmc": 1, "timing": 2, "size3d": 3}[t["family"]],
                             t["case"], t["natoms"], t["ranks"], t["seed"], t.get("start_sweep", 0), t.get("repeat", 0)))
    for i, task in enumerate(tasks, 1):
        task["number"] = i
        if task["renderer"] != "mobile":
            text = (geometry3d if task["renderer"] == "three_d" else gen).render_mc(task)
            path = ROOT / "templates" / ("in." + task["id"])
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(text.encode("ascii"))
            task["template"] = path.relative_to(ROOT).as_posix()

    for path in FIXED.iterdir():
        if path.is_file():
            copy_file(path, ROOT / "reused_rng_regression" / path.name)
    counts = dict(collections.Counter(t["family"] for t in tasks))
    manifest = {"version": "rng-recompute-20261020-v1", "tasks": tasks, "task_counts": counts,
        "maximum_ranks": 24, "simulation_kinds": ["fast"], "minimum_simulation_ranks": 2,
        "applications": [], "reference_distribution_review_required": True,
        "old_parallel_fast_samples_reusable": False,
        "already_completed_regression": "reused_rng_regression/regression_summary.json",
        "reused_records": len(records), "reference_records_file": "baseline/records.json",
        "timing_policy": "all fast production loops >=30 seconds; 3 legacy repeats, 5 size3d repeats; no atom timings rerun",
        "scope_note": "Original atom and rank-1 evidence retained with old hashes; no fabricated new-build attribution."}
    save(ROOT / "manifest.json", manifest)
    save(ROOT / "baseline/records.json", records)
    save(ROOT / "baseline/excluded.json", excluded)
    save(ROOT / "baseline/source_datasets.json", {
        "core_status_sha256": sha(OLD / "results/run_status.json"),
        "incremental_status_sha256": sha(OLD / "incremental/run_status.json"),
        "three_d_status_sha256": sha(THREE / "results/run_status.json"),
        "three_d_host": load(THREE / "results/metadata/environment.json")["host"],
        "current_rng_library_sha256": load(FIXED / "regression_summary.json")["runtime"]["library_sha256"]})
    for label, source in (("core", OLD / "results/metadata"), ("three_d", THREE / "results/metadata")):
        for path in source.iterdir():
            if path.is_file() and (path.name.startswith(("environment", "affinity", "CMake", "lammps_flags"))):
                copy_file(path, ROOT / "baseline" / label / "metadata" / path.name)
    with (ROOT / "task_list.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["number", "id", "family", "case", "ranks", "natoms", "requested_sweeps", "seed", "repeat"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tasks)
    immutable = {}
    for directory in ("baseline", "templates", "tables", "src", "vendor", "reused_rng_regression"):
        immutable.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / directory).rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts})
    immutable["manifest.json"] = sha(ROOT / "manifest.json")
    save(ROOT / "input_integrity.json", immutable)
    print(json.dumps({"new_tasks": len(tasks), "counts": counts, "reused_records": len(records),
                      "immutable_files": len(immutable)}, indent=2))


if __name__ == "__main__":
    main()
