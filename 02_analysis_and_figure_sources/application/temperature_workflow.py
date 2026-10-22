#!/usr/bin/env python3
"""Temperature-isolated phase sampling with a fixed complementary mu grid."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import sys

import workflow_phase_diagram as wf


def grid_assignment(states):
    groups = {}
    for state in states:
        groups.setdefault((state["T"], state["phase"]), []).append(state)
    assignment = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda state: state["mu"])
        selected = set(range(0, len(ordered), 2))
        middle = len(ordered) // 2
        if len(ordered) >= 5 and middle not in selected:
            # Keep the original low/middle/high checks without increasing the
            # half-grid count. The exchanged point remains available in B.
            replace = min(selected - {0, len(ordered) - 1},
                          key=lambda index: (abs(index - middle), index))
            selected.remove(replace)
            selected.add(middle)
        for index, state in enumerate(ordered):
            sid = wf.state_id(state["phase"], state["T"], state["mu"])
            assignment[sid] = "A" if index in selected else "B"
    return assignment


def export_grid(cfg, output):
    assignment = grid_assignment(cfg["states"])
    rows = []
    for state in sorted(cfg["states"], key=lambda state: (state["T"], state["phase"], state["mu"])):
        sid = wf.state_id(state["phase"], state["T"], state["mu"])
        dense = state.get("dense_validation", False)
        rows.append({"T": state["T"], "phase": state["phase"], "delta_mu": state["mu"],
                     "grid": assignment[sid], "dense_validation": dense,
                     "trajectories": (len(cfg["dense_starts"]) * cfg["dense_seed_count"] if dense
                                      else len(cfg["ordinary_starts"]) * cfg["ordinary_seed_count"]),
                     "state": sid})
    wf.atomic_json(output / "mu_grid_split.json", assignment)
    wf.write_csv(output / "mu_grid_split.csv", rows)
    lines = ["# Fixed complementary chemical-potential grid", "",
             "A is the default. B is its disjoint complement; A + B recovers the original grid.",
             "Values are in epsilon_AA. Solid and liquid retain their historical initial compositions.", "",
             "| Temperature | Phase | A (default) | B (supplement later) |",
             "|---|---|---|---|"]
    for T, phase in sorted({(s["T"], s["phase"]) for s in cfg["states"]}):
        values = {}
        for half in ("A", "B"):
            values[half] = ", ".join(f"{s['mu']:.1f}" for s in sorted(cfg["states"], key=lambda s: s["mu"])
                if s["T"] == T and s["phase"] == phase
                and assignment[wf.state_id(phase, T, s["mu"])] == half)
        lines.append(f"| {T:.2f} | {phase} | {values['A']} | {values['B']} |")
    (output / "MU_GRID.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved the complete A/B grid: {output / 'mu_grid_split.csv'}")


def parse_layout(layout):
    match = re.fullmatch(r"([1-9][0-9]*)[xX]([1-9][0-9]*)", layout)
    if not match:
        raise ValueError("--layout must be ranks x simultaneous jobs, for example 8x3")
    ranks, workers = map(int, match.groups())
    if ranks * workers > 24:
        raise ValueError("The layout must not exceed 24 total MPI ranks")
    return ranks, workers


def parse_cpu_list(text):
    cpus = []
    for part in text.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            cpus.append(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            cpus.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError("Invalid --cpu-list; use OS CPU IDs, for example 0-23")
    if not cpus or min(cpus) < 0 or len(set(cpus)) != len(cpus):
        raise ValueError("CPU IDs must be nonnegative and distinct")
    return cpus


def cpu_topology(cpu):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        return (int((root / "physical_package_id").read_text()),
                int((root / "core_id").read_text()))
    except (OSError, ValueError):
        return None


def allocate_cpus(ranks, workers, cpu_list=None, allow_smt=False):
    if not hasattr(os, "sched_getaffinity"):
        if workers > 1 or cpu_list:
            raise ValueError("Concurrent pinned jobs require Linux; use --layout 1x1 for local checking")
        return []
    allowed = set(os.sched_getaffinity(0))
    candidates = parse_cpu_list(cpu_list) if cpu_list else sorted(allowed)
    if not set(candidates).issubset(allowed):
        raise ValueError("--cpu-list includes CPUs outside this process's permitted CPU set")
    topology = {cpu: cpu_topology(cpu) for cpu in candidates}
    if not cpu_list:
        candidates.sort(key=lambda cpu: (topology[cpu] or (0, cpu), cpu))
    if not allow_smt:
        physical, unique = set(), []
        for cpu in candidates:
            core = topology[cpu] or ("unknown", cpu)
            if core not in physical:
                unique.append(cpu)
                physical.add(core)
        candidates = unique
    needed = ranks * workers
    if len(candidates) < needed:
        raise ValueError(f"Layout needs {needed} distinct cores, but only {len(candidates)} are available. "
                         "Choose a smaller --layout or explicitly allow SMT with --allow-smt.")
    return [candidates[index * ranks:(index + 1) * ranks] for index in range(workers)]


def selected_states(cfg, T, half, assignment, text_filter=None):
    selected = [copy.deepcopy(state) for state in cfg["states"] if state["T"] == T and
                (half == "all" or assignment[wf.state_id(state["phase"], T, state["mu"])] == half)]
    if text_filter:
        matched = {wf.state_id(case["phase"], T, case["mu"])
                   for case in wf.make_cases(dict(cfg, states=selected))
                   if text_filter in wf.case_id(case)}
        selected = [state for state in selected if wf.state_id(state["phase"], T, state["mu"]) in matched]
    return selected


def selected_case_ids(cfg, states, args):
    selected = {wf.state_id(s["phase"], s["T"], s["mu"]) for s in states}
    return {wf.case_id(case) for case in wf.make_cases(cfg)
            if wf.state_id(case["phase"], case["T"], case["mu"]) in selected
            and (args.phase == "both" or case["phase"] == args.phase)
            and (not args.starts or case["start"] in args.starts)
            and (not args.filter or args.filter in wf.case_id(case))}


def temperature_directory(root, T):
    root = Path(root).resolve()
    return root if root.name == f"T{T:.2f}" else root / f"T{T:.2f}"


def load_rich_map(path, N):
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--rich-c-map must be a JSON object")
    if "compositions" in data:
        if data.get("N", N) != N:
            raise ValueError("The rich concentration map uses a different atom count")
        data = data["compositions"]
    if not isinstance(data, dict) or not data:
        raise ValueError("--rich-c-map must contain state-id to concentration mappings")
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"Invalid rich concentration for {key}")
    return data


def source_configuration_identity(cfg):
    identity = copy.deepcopy(cfg)
    identity.pop("cycles", None)
    return wf.digest(identity)


def repair_source_check(source, manifest):
    if not source.is_dir():
        raise ValueError(f"Old temperature results are missing: {source}; supply --reuse-from")
    if source_configuration_identity(wf.load_config(source / "effective_config.json")) != manifest["source_configuration_identity"]:
        raise ValueError("Old source configuration changed; use a new repair output root")
    if wf.sha(source / "simulation_protocol.json") != manifest["source_protocol_sha256"]:
        raise ValueError("Old source protocol changed")
    environment = json.loads((source / "environment.json").read_text())
    if environment["run_identity"] != manifest["source_run_identity"]:
        raise ValueError("Old source binary/library/ranks/potential identity changed")


def repair_configuration(base, states, source, output, args):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Repair output and reused temperature archive must be separate, non-nested directories")
    wf.check_protocol(base, source)
    source_cfg = wf.load_config(source / "effective_config.json")
    cfg = copy.deepcopy(source_cfg)
    selected = {wf.state_id(s["phase"], s["T"], s["mu"]) for s in states}
    by_id = {wf.state_id(s["phase"], s["T"], s["mu"]): s for s in cfg["states"]}
    missing = selected - set(by_id)
    if missing:
        raise ValueError(f"Selected states have no old temperature configuration: {sorted(missing)}")
    N = 4 * math.prod(cfg["box_cells"])
    values = load_rich_map(args.rich_c_map, N)
    definitions = []
    for sid in sorted(selected):
        state = by_id[sid]
        if values is not None and sid not in values:
            raise ValueError(f"Rich concentration map has no entry for {sid}")
        new_c = float(values[sid] if values is not None else args.rich_c)
        old_c = state["initial_compositions"]["rich"]
        poor_c = state["initial_compositions"]["poor"]
        if not poor_c < new_c < old_c or (new_c - poor_c)*N < 1 - 1e-9:
            raise ValueError(f"{sid}: new rich concentration must exceed poor by at least one atom and be below old rich")
        definitions.append({"state": sid, "old_rich_c": old_c, "new_rich_c": new_c,
                            "planned_B_atoms": round(new_c*N)})
        state["initial_compositions"]["rich"] = new_c
    cfg["cycles"] = args.cycles
    replacements = [case for case in wf.make_cases(cfg)
                    if wf.state_id(case["phase"], case["T"], case["mu"]) in selected
                    and case["start"] == "rich" and case["fix"] == "fast"]
    if not replacements:
        raise ValueError("Old configuration contains no rich trajectories for the requested states")
    if args.filter and any(args.filter not in wf.case_id(case) for case in replacements):
        raise ValueError("A repair replaces all rich seeds of a state; use a mu/state filter rather than a seed filter")
    manifest = {"schema": "rich-repair-1", "temperature": states[0]["T"],
                "source_temperature_directory": str(source),
                "source_config_sha256": wf.sha(source / "effective_config.json"),
                "source_configuration_identity": source_configuration_identity(source_cfg),
                "source_protocol_sha256": wf.sha(source / "simulation_protocol.json"),
                "source_run_identity": json.loads((source / "environment.json").read_text())["run_identity"],
                "replacement_case_ids": sorted(wf.case_id(case) for case in replacements),
                "initial_state_changes": definitions,
                "rule": "Reused poor/liquid/reference data are read-only; selected old rich cases are excluded even before replacements complete."}
    existing = output / "rich_repair.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        # A later extension may change only the old cycle target. Preserve
        # the original file hash for provenance and validate every other field.
        manifest["source_config_sha256"] = previous["source_config_sha256"]
        if previous != manifest:
            raise ValueError("Repair scope/concentrations/source changed; use a new --output. Only --cycles may increase in place.")
        saved = wf.load_config(output / "effective_config.json")
        if {(wf.case_id(c), c["c0"]) for c in wf.make_cases(saved)} != {(wf.case_id(c), c["c0"]) for c in wf.make_cases(cfg)}:
            raise ValueError("Saved repair configuration has different case definitions")
        wf.check_protocol(saved, source)
        saved["cycles"] = max(saved["cycles"], args.cycles)
        cfg = saved
    elif any(path.name != "workflow.lock" for path in output.iterdir()):
        raise ValueError("Repair output already contains unrelated files; choose a new --output")
    repair_source_check(source, manifest)
    wf.atomic_json(existing, manifest)
    wf.atomic_json(output / "effective_config.json", cfg)
    wf.check_protocol(cfg, output, create=True)
    return cfg, manifest


def repair_trajectory_sources(cfg, output, source, manifest):
    replacement_ids = set(manifest["replacement_case_ids"])
    expected = {wf.case_id(case) for case in wf.make_cases(cfg)}
    if not replacement_ids.issubset(expected):
        raise ValueError("Repair manifest includes unknown case IDs")
    # Route every selected rich case to the new root, including missing or
    # incomplete cases. Never substitute the old melted rich trajectory.
    return {case: (output if case in replacement_ids else source) / "trajectories" / case
            for case in sorted(expected)}


def analyze_repair(cfg, output, source, manifest):
    repair_source_check(source, manifest)
    wf.check_protocol(cfg, source)
    new_environment = output / "environment.json"
    if new_environment.exists() and json.loads(new_environment.read_text())["run_identity"] != manifest["source_run_identity"]:
        raise ValueError("New and reused binary/library/ranks/potential identities differ")
    sources = repair_trajectory_sources(cfg, output, source, manifest)
    wf.atomic_json(output / "analysis" / "rich_repair_provenance.json", {
        "source_temperature_directory": str(source), "replacement_case_ids": manifest["replacement_case_ids"],
        "source_config_sha256_at_repair_creation": manifest["source_config_sha256"],
        "source_config_sha256_at_analysis": wf.sha(source / "effective_config.json"),
        "excluded_old_rich_paths": {case: str(source / "trajectories" / case)
                                   for case in manifest["replacement_case_ids"]},
        "initial_state_changes": manifest["initial_state_changes"]})
    return wf.analyze(cfg, output, trajectory_sources=sources, reference_output=source)


def run_rich_repair(base, T, args, assignment):
    output = args.output.resolve() / f"T{T:.2f}"
    source = temperature_directory(args.reuse_from, T)
    states = [s for s in selected_states(base, T, args.mu_grid, assignment, args.filter)
              if s["phase"] == "solid"]
    if getattr(args, "only_mapped_rich", False):
        values = load_rich_map(args.rich_c_map, 4 * math.prod(base["box_cells"]))
        known = {wf.state_id(s["phase"], s["T"], s["mu"]) for s in base["states"]}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown states in rich map: {sorted(unknown)}")
        states = [s for s in states if wf.state_id(s["phase"], s["T"], s["mu"]) in values]
    if not states:
        raise ValueError(f"No solid states selected at T={T:.2f}")
    if not source.is_dir():
        raise ValueError(f"Reused source directory is missing: {source}")
    output.mkdir(parents=True, exist_ok=True)
    # Lock the old temperature against concurrent extension while its raw
    # files are read; no source configuration, analysis, or trajectory is edited.
    with wf.output_lock(source), wf.output_lock(output):
        cfg, manifest = repair_configuration(base, states, source, output, args)
        if args.auto_extend and cfg["cycles"] > args.max_cycles:
            raise ValueError("--max-cycles is below the saved repair target; do not silently exceed it")
        args.selected_case_ids = set(manifest["replacement_case_ids"])
        args.selected_states = {wf.state_id(s["phase"], T, s["mu"]) for s in states}
        args.reuse_temperature_output = source
        args.stage = "sample"
        write_plan(cfg, output, args, states, assignment)
        plan_path = output / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan.update(reference_jobs=0, prepared_jobs=0,
                    reused_reference_jobs=len(wf.PHASES)*len(cfg["reference_temperatures"])*cfg["dense_seed_count"],
                    reference_scope=f"Read-only references from {source}",
                    reused_trajectory_count=len(wf.make_cases(cfg))-len(args.selected_case_ids))
        wf.atomic_json(plan_path, plan)
        wf.write_csv(output / "new_rich_trajectories.csv", [
            {"id": wf.case_id(case), "T": case["T"], "mu": case["mu"], "seed": case["seed"],
             "new_c0": case["c0"], "planned_B_atoms": round(4*math.prod(cfg["box_cells"])*case["c0"])}
            for case in wf.make_cases(cfg) if wf.case_id(case) in args.selected_case_ids])
        export_grid(dict(base, states=[s for s in base["states"] if s["T"] == T]), output)
        print(f"Rich repair: {len(args.selected_case_ids)} new trajectories; no prepared/enthalpy jobs; old selected rich excluded.", flush=True)
        if args.command == "plan":
            print(f"Only {len(args.selected_case_ids)} new rich trajectories; old poor/liquid/enthalpy reused.")
            return
        wf.run_workflow(args, cfg, output)
        report = analyze_repair(cfg, output, source, manifest)
        while args.auto_extend and cfg["cycles"] < args.max_cycles:
            flagged = args.selected_states.intersection(report["states_requiring_extension_or_review"])
            if not flagged:
                break
            args.selected_states = flagged
            cfg["cycles"] = min(cfg["cycles"] + 20, args.max_cycles)
            wf.run_workflow(args, cfg, output)
            report = analyze_repair(cfg, output, source, manifest)


def temperature_config(base, states, output, cycles, include_reference=False):
    cfg = copy.deepcopy(base)
    existing = output / "effective_config.json"
    previous = wf.load_config(existing) if existing.exists() else None
    all_states = {}
    if previous:
        wf.check_protocol(cfg, output)
        temperatures = {state["T"] for state in states}
        if any(state["T"] not in temperatures for state in previous["states"]):
            raise ValueError("Existing output includes other temperatures; do not mix old flat archives here")
        all_states.update({wf.state_id(s["phase"], s["T"], s["mu"]): s for s in previous["states"]})
        cfg["include_atom_swap_reference"] |= previous["include_atom_swap_reference"]
    for state in states:
        sid = wf.state_id(state["phase"], state["T"], state["mu"])
        if sid in all_states and state != all_states[sid]:
            raise ValueError(f"State definition changed for {sid}; preserve the fixed grid and initial states")
        all_states[sid] = state
    cfg["states"] = sorted(all_states.values(), key=lambda s: (s["T"], s["phase"], s["mu"]))
    cfg["cycles"] = cycles
    cfg["minimum_cycles"] = 60
    cfg["include_atom_swap_reference"] |= include_reference
    return cfg


def write_plan(cfg, output, args, states, assignment):
    wf.plan(cfg, output, workers=args.workers)
    path = output / "plan.json"
    plan = json.loads(path.read_text())
    current = dict(cfg, states=states)
    cases = wf.make_cases(current)
    if args.filter:
        cases = [case for case in cases if args.filter in wf.case_id(case)]
    if getattr(args, "selected_case_ids", None) is not None:
        cases = [case for case in cases if wf.case_id(case) in args.selected_case_ids]
    production_steps = plan["production_md_steps_per_trajectory"]
    plan.update(maximum_concurrent_lammps_jobs=args.workers, ranks_per_job=args.ranks,
                total_cpu_limit=args.workers * args.ranks, layout=args.layout,
                current_mu_grid=args.mu_grid, current_invocation_trajectories=len(cases),
                current_invocation_target_md_steps=len(cases) * production_steps,
                minimum_production_cycles=cfg["minimum_cycles"],
                reference_scope="Configured pure-A reference grid, local to this folder",
                reference_temperatures=cfg["reference_temperatures"],
                current_mu_points=sorted({state["mu"] for state in states}),
                cumulative_mu_grids=sorted({assignment[wf.state_id(s["phase"], s["T"], s["mu"])]
                                            for s in cfg["states"]}),
                sampling_diagnostics="Conservative review flags; failure is not proof of nonequilibration")
    wf.atomic_json(path, plan)
    print(f"Layout {args.layout}; selected grid {args.mu_grid}; {len(cases)} trajectories; "
          f"target {cfg['cycles']} cycles each. Output: {output}")


def run_temperature(base, T, args, assignment):
    output = args.output.resolve() / f"T{T:.2f}"
    states = selected_states(base, T, args.mu_grid, assignment, args.filter)
    if args.phase != "both":
        states = [s for s in states if s["phase"] == args.phase]
    if not states:
        raise ValueError(f"No selected states at T={T:.2f}; check --mu-grid and --filter")
    output.mkdir(parents=True, exist_ok=True)
    with wf.output_lock(output):
        cfg = temperature_config(base, states, output, args.cycles, args.include_reference)
        args.selected_case_ids = selected_case_ids(cfg, states, args)
        if not args.selected_case_ids:
            raise ValueError("No trajectories match the phase/start/filter selection")
        write_plan(cfg, output, args, states, assignment)
        # Preserve the complete preassigned grid, even when only A has run.
        all_temperature_states = [state for state in base["states"] if state["T"] == T]
        export_grid(dict(base, states=all_temperature_states), output)
        if args.command == "plan":
            return
        args.selected_states = {wf.state_id(s["phase"], T, s["mu"]) for s in states}
        wf.run_workflow(args, cfg, output)
        if args.stage in ("all", "sample"):
            report = wf.analyze(cfg, output)
            while args.auto_extend and cfg["cycles"] < args.max_cycles:
                flagged = args.selected_states.intersection(report["states_requiring_extension_or_review"])
                if not flagged:
                    break
                args.selected_states = flagged
                cfg["cycles"] = min(cfg["cycles"] + 20, args.max_cycles)
                args.stage = "sample"
                wf.run_workflow(args, cfg, output)
                report = wf.analyze(cfg, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plan", "analyze", "grid"))
    parser.add_argument("--lmp")
    parser.add_argument("--config", type=Path, default=wf.ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=wf.ROOT / "results")
    parser.add_argument("--temperature", "--temperatures", type=float, nargs="+", dest="temperatures")
    parser.add_argument("--mu-grid", choices=("A", "B", "all"), default="A")
    parser.add_argument("--layout", default="8x3", help="MPI ranks per job x simultaneous mu jobs")
    parser.add_argument("--mpi-launcher", default="mpirun")
    parser.add_argument("--np-flag", default="-np")
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--filter", help="Optional trajectory-id substring within the selected grid")
    parser.add_argument("--phase", choices=("both", "solid", "liquid"), default="both")
    parser.add_argument("--start", dest="starts", choices=("poor", "middle", "rich"), nargs="+")
    parser.add_argument("--reuse-from", type=Path, help="Old result root (or one T folder); read-only data reuse")
    rich = parser.add_mutually_exclusive_group()
    rich.add_argument("--rich-c", type=float, help="New solid rich concentration for the selected temperatures/states")
    rich.add_argument("--rich-c-map", type=Path, help="JSON state-specific new solid rich concentrations")
    parser.add_argument("--only-mapped-rich", action="store_true",
                        help="Repair only states listed in --rich-c-map; reuse other rich states")
    parser.add_argument("--stage", choices=("all", "prepare", "reference", "sample"), default="all")
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--cpu-list", help="Restrict to specified OS CPU IDs, e.g. 0-23")
    parser.add_argument("--allow-smt", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--auto-extend", action="store_true", help="Optional; off by default")
    parser.add_argument("--max-cycles", type=int, default=80)
    args = parser.parse_args(argv)
    args.scheduling_version = "temperature-application-1.3-rich-repair"
    repair_requested = args.rich_c is not None or args.rich_c_map is not None
    if args.only_mapped_rich and args.rich_c_map is None:
        parser.error("--only-mapped-rich requires --rich-c-map")
    if repair_requested:
        if args.command not in ("plan", "run") or not args.temperatures or args.phase != "solid" or args.starts != ["rich"] or args.reuse_from is None:
            parser.error("Rich repair requires plan/run, explicit --temperature, --phase solid --start rich and --reuse-from")
        if args.stage not in ("all", "sample") or args.include_reference:
            parser.error("Rich repair only samples new fast/swap rich trajectories; do not request other stages/comparators")
        if args.rich_c is not None and (not math.isfinite(args.rich_c) or not 0 < args.rich_c < 1):
            parser.error("--rich-c must be strictly between 0 and 1")
    elif args.reuse_from is not None and args.command != "analyze":
        parser.error("--reuse-from on plan/run requires --rich-c or --rich-c-map")
    if args.cycles <= 0 or args.timeout <= 0:
        parser.error("--cycles and --timeout must be positive")
    if args.auto_extend and (args.max_cycles < args.cycles or args.filter or args.stage not in ("all", "sample")):
        parser.error("--auto-extend needs unfiltered all/sample and --max-cycles >= --cycles")
    args.ranks, args.workers = parse_layout(args.layout)
    base = wf.load_config(args.config)
    assignment = grid_assignment(base["states"])
    if args.command == "grid":
        args.output.mkdir(parents=True, exist_ok=True)
        export_grid(base, args.output)
        return
    available = sorted({state["T"] for state in base["states"]})
    requested = sorted(set(args.temperatures)) if args.temperatures else available
    if any(T not in available for T in requested):
        parser.error(f"Choose temperatures from {available}")
    if args.command == "analyze":
        found = False
        for T in requested:
            output = args.output.resolve() / f"T{T:.2f}"
            if not (output / "effective_config.json").exists():
                if args.temperatures:
                    raise ValueError(f"No completed run configuration: {output}")
                continue
            found = True
            manifest_file = output / "rich_repair.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                source = (temperature_directory(args.reuse_from, T) if args.reuse_from else
                          Path(manifest["source_temperature_directory"]))
                with wf.output_lock(source), wf.output_lock(output):
                    analyze_repair(wf.load_config(output / "effective_config.json"), output, source, manifest)
            else:
                if args.reuse_from:
                    raise ValueError("This output has no rich repair manifest")
                with wf.output_lock(output):
                    wf.analyze(wf.load_config(output / "effective_config.json"), output)
        if not found:
            raise ValueError("No temperature folders found; old flat archives use workflow_phase_diagram.py analyze")
        return
    if args.filter:
        requested = [T for T in requested if selected_states(base, T, args.mu_grid, assignment, args.filter)]
        if not requested:
            raise ValueError("--filter matched no trajectories in the selected mu grid")
    args.cpu_groups = (allocate_cpus(args.ranks, args.workers, args.cpu_list, args.allow_smt)
                       if args.command == "run" else [])
    print(f"Temperatures: {requested}; results saved independently, without node labels.", flush=True)
    for T in requested:
        # Extension of one temperature must not change the next temperature's
        # initial cycle target or requested run stage.
        if repair_requested:
            run_rich_repair(base, T, copy.copy(args), assignment)
        else:
            run_temperature(base, T, copy.copy(args), assignment)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, wf.subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
