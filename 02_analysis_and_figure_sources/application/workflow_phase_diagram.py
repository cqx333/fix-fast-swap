#!/usr/bin/env python3
"""Unified, resumable MD/SGC-MC phase-diagram application workflow.

Run this Python process once, not once per MPI rank. Raw data, counters and
checkpoints remain separate from analysis. The temperature entry point can
schedule independent chemical-potential groups concurrently.
"""
import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import itertools
from importlib.metadata import version as package_version
import json
import math
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np
from scipy.optimize import least_squares
from scipy.special import xlogy
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
VERSION = "phase-application-1.1"
PHASES = ("solid", "liquid")
PROTOCOL_KEYS = ("box_cells", "fcc_number_density", "pressure", "timestep",
                 "prepare_temperature", "melt_temperature", "prepare_relax_steps", "melt_steps",
                 "state_equil_steps", "md_steps_per_mc_block", "mc_blocks_per_cycle",
                 "npt_steps_per_cycle", "snapshot_every", "reference_equil_steps", "reference_sample_steps",
                 "csp_threshold")
COLUMNS = ["stage", "md_step", "N", "n_B", "c", "U_per_atom", "H_per_atom", "H_thermo_per_atom",
           "temperature", "pressure", "volume", "mean_csp", "ordered_fraction",
           "attempts_increment", "accepts_increment"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def output_lock(output):
    path = output / "workflow.lock"
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ValueError(f"Output is locked: {path}. If a previous process crashed, first confirm all MPI jobs stopped, then remove ONLY this lock file.")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "host": platform.node(), "time": time.time()}, stream)
        yield
    finally:
        path.unlink()


def check_protocol(cfg, output, create=False):
    actual = {key: cfg[key] for key in PROTOCOL_KEYS}
    path = output / "simulation_protocol.json"
    if path.exists() and json.loads(path.read_text()) != actual:
        raise ValueError("Physical MD/MC protocol differs from this output archive; use a new --output")
    if create:
        atomic_json(path, actual)


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def seed_for(*parts):
    # Stable across machines, job order, additions to the grid, and resumes.
    return 1 + int(digest(list(parts))[:15], 16) % 899999999


def state_id(phase, temperature, mu):
    return f"{phase}_T{temperature:.2f}_mu{mu:+.3f}"


def case_id(case):
    return (state_id(case["phase"], case["T"], case["mu"]) +
            f"_start{case['start']}_seed{case['seed']}_{case['fix']}")


def initialize_bundle(original):
    if (ROOT / "config.json").exists():
        raise ValueError("config.json exists; initialization will not overwrite this bundle")
    original = Path(original).resolve()
    legacy = ROOT / "legacy_data"
    legacy.mkdir(exist_ok=True)
    for directory in ("potential", "src"):
        target = ROOT / directory if directory == "potential" else ROOT / "src" / "legacy"
        shutil.copytree(original / directory, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = ROOT / "original_scripts"
    scripts.mkdir(exist_ok=True)
    for source in original.glob("workflow_*.py"):
        shutil.copy2(source, scripts / source.name)
    states, source_records = [], []
    semi = original / "semi"
    for folder in sorted(semi.iterdir()):
        match = re.fullmatch(r"([0-9.]+)_(solid|liquid)", folder.name)
        if not match or not folder.is_dir():
            continue
        T, phase = float(match[1]), match[2]
        for source in sorted(folder.iterdir()):
            if not source.is_file() or not re.fullmatch(r"-?\d+(\.\d+)?", source.name):
                continue
            values = np.loadtxt(source, comments="#", ndmin=2)
            if values.shape[1] != 2 or len(values) < 10:
                raise ValueError(f"Invalid historical composition data: {source}")
            if np.any(np.diff(values[:, 0]) <= 0) or np.any((values[:, 1] < 0) | (values[:, 1] > 1)):
                raise ValueError(f"Invalid historical steps/compositions: {source}")
            tail = values[int(0.8 * len(values)):, 1]
            mu = float(source.name)
            item = {"phase": phase, "T": T, "mu": mu,
                    "legacy_c": float(np.mean(tail)), "legacy_md_steps": int(values[-1, 0])}
            states.append(item)
            destination = legacy / folder.name / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            log = source.with_name(source.name + ".log")
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            first_commands = [line.strip() for line in text.splitlines()
                              if line.startswith(("pair_style ", "pair_coeff ", "pair_modify "))]
            source_records.append({**item, "path": str(source.relative_to(original)),
                                   "sha256": sha(source), "log_sha256": sha(log) if log.exists() else None,
                                   "pair_commands": first_commands,
                                   "historical_fix": "atom/swap" if "fix mc all atom/swap" in text else "unknown"})
    if not states:
        raise ValueError("No historical phase/T/mu composition files found")
    for name in ("solid.H", "liquid.H", "phase_diagram.txt", "solid_snapshot.dump", "liquid_snapshot.dump"):
        if (semi / name).exists():
            shutil.copy2(semi / name, legacy / name)
    for name in ("solid.log", "liquid.log"):
        if (semi / name).exists():
            shutil.copy2(semi / name, scripts / ("historical_" + name))
    # Temperature/phase low, middle, high mu states get the denser validation.
    temperatures = sorted({s["T"] for s in states})
    representative_T = {temperatures[0], temperatures[len(temperatures) // 2], temperatures[-1]}
    for phase, T in itertools.product(PHASES, representative_T):
        group = sorted((s for s in states if s["phase"] == phase and s["T"] == T), key=lambda s: s["mu"])
        for index in {0, len(group) // 2, len(group) - 1}:
            group[index]["dense_validation"] = True
    for s in states:
        s.setdefault("dense_validation", False)
        old_c = s["legacy_c"]
        s["initial_compositions"] = {"poor": max(0.0, old_c - 0.10),
                                     "middle": old_c,
                                     "rich": min(0.90, old_c + 0.10)}
    config = {
        "version": VERSION, "temperature_unit": "epsilon_AA/k_B", "energy_unit": "epsilon_AA/atom",
        "melting_temperature": 0.65, "melting_temperature_sigma": 0.0,
        "melting_reference": "Author-confirmed external two-phase coexistence result; not recalculated here",
        "box_cells": [12, 12, 12], "fcc_number_density": 0.9, "pressure": 0.0,
        "timestep": 0.005, "prepare_temperature": 0.6, "melt_temperature": 0.9,
        "prepare_relax_steps": 10000, "melt_steps": 50000, "state_equil_steps": 10000,
        "md_steps_per_mc_block": 1000, "mc_blocks_per_cycle": 20,
        "npt_steps_per_cycle": 20000, "snapshot_every": 1000,
        "cycles": 20, "reference_equil_steps": 10000, "reference_sample_steps": 100000,
        "reference_temperatures": sorted(set(temperatures + [0.65])),
        "base_seeds": [2145927, 3719241, 5817329], "ordinary_seed_count": 1,
        "dense_seed_count": 3, "ordinary_starts": ["poor", "rich"],
        "dense_starts": ["poor", "middle", "rich"], "include_atom_swap_reference": False,
        "reference_comparison_states": [state_id("solid", 0.60, -2.2), state_id("liquid", 0.60, -2.2)],
        "tail_fraction": 0.20, "minimum_effective_samples": 10.0,
        "composition_equivalence_margin": 0.01, "polynomial_degree": 3,
        "bootstrap_replicates": 200, "structure_dump_every_cycles": 5,
        "csp_threshold": 2.0,
        "author_confirmations": {"historical_grid_confirmed": True,
                                 "melting_reference_confirmed": True,
                                 "phase_labels_confirmed": False,
                                 "enthalpy_linear_model_confirmed": False},
        "states": sorted(states, key=lambda s: (s["T"], s["phase"], s["mu"]))}
    atomic_json(ROOT / "config.json", config)
    atomic_json(ROOT / "historical_inventory.json", source_records)
    write_csv(ROOT / "historical_state_grid.csv", states)
    print(f"Recovered {len(states)} actual historical state points; {sum(s['dense_validation'] for s in states)} dense states.")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["version"] = VERSION
    cfg.setdefault("minimum_cycles", cfg["cycles"])
    cfg.setdefault("composition_equivalence_relative_margin", 0.25)
    cfg.setdefault("composition_equivalence_atom_floor", 2.0)
    ints = ["prepare_relax_steps", "melt_steps", "state_equil_steps", "md_steps_per_mc_block",
            "mc_blocks_per_cycle", "npt_steps_per_cycle", "snapshot_every", "cycles",
            "reference_equil_steps", "reference_sample_steps"]
    for key in ints:
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["npt_steps_per_cycle"] % cfg["snapshot_every"]:
        raise ValueError("NPT cycle must be divisible by snapshot_every")
    if cfg["reference_sample_steps"] % cfg["snapshot_every"]:
        raise ValueError("Reference sampling must be divisible by snapshot_every")
    if cfg["md_steps_per_mc_block"] < 2:
        raise ValueError("At least two MD steps are required per MC block")
    if not 0 < cfg["tail_fraction"] <= 1 or cfg["melting_temperature"] <= 0 or cfg["timestep"] <= 0:
        raise ValueError("Invalid tail fraction, melting temperature or timestep")
    if max(cfg["ordinary_seed_count"], cfg["dense_seed_count"]) > len(cfg["base_seeds"]):
        raise ValueError("Not enough independent base seeds")
    ids = [state_id(s["phase"], s["T"], s["mu"]) for s in cfg["states"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate phase/T/mu states")
    for s in cfg["states"]:
        if s["phase"] not in PHASES or s["T"] <= 0 or not math.isfinite(s["mu"]):
            raise ValueError("Invalid phase, temperature or chemical potential")
        for value in s["initial_compositions"].values():
            if not 0 <= value <= 1:
                raise ValueError("Invalid initial composition")
    return cfg


def make_cases(cfg):
    cases = []
    for state in cfg["states"]:
        dense = state.get("dense_validation", False)
        seeds = cfg["base_seeds"][:cfg["dense_seed_count"] if dense else cfg["ordinary_seed_count"]]
        starts = cfg["dense_starts"] if dense else cfg["ordinary_starts"]
        for start, seed in itertools.product(starts, seeds):
            cases.append({**state, "start": start, "c0": state["initial_compositions"][start],
                          "seed": seed, "fix": "fast"})
        if cfg["include_atom_swap_reference"] and state_id(state["phase"], state["T"], state["mu"]) in cfg["reference_comparison_states"]:
            for seed in cfg["base_seeds"][:cfg["dense_seed_count"]]:
                cases.append({**state, "start": "middle", "c0": state["initial_compositions"]["middle"],
                              "seed": seed, "fix": "atom"})
            # Ensure a matched fast comparator exists, even for ordinary states.
            for seed in cfg["base_seeds"][:cfg["dense_seed_count"]]:
                case = {**state, "start": "middle", "c0": state["initial_compositions"]["middle"],
                        "seed": seed, "fix": "fast"}
                if not any(case_id(c) == case_id(case) for c in cases):
                    cases.append(case)
    return cases


def quoted(path):
    return '"' + Path(path).resolve().as_posix() + '"'


def base_input(cfg, restart=None):
    lines = ["units lj", "atom_style atomic", "boundary p p p"]
    if restart:
        lines.append("read_restart " + quoted(restart))
    else:
        nx, ny, nz = cfg["box_cells"]
        lines += [f"lattice fcc {cfg['fcc_number_density']} orient x 1 0 0 orient y 0 1 0",
                  f"region box block 0 {nx} 0 {ny} 0 {nz}", "create_box 2 box", "create_atoms 1 box"]
    lines += ["include " + quoted(ROOT / "potential" / "force.txt"),
              f"timestep {cfg['timestep']}", f"variable p_external equal {cfg['pressure']:.17g}",
              "neighbor 0.8 bin", "neigh_modify every 1 delay 0 check yes",
              "pair_modify tail no", "variable b atom type==2", "compute nb all reduce sum v_b",
              "compute csp all centro/atom fcc", "compute mcsp all reduce ave c_csp",
              f"variable ordered atom c_csp<{cfg['csp_threshold']}",
              "compute ofrac all reduce ave v_ordered",
              "thermo_style custom step atoms temp pe enthalpy press vol c_nb c_mcsp c_ofrac",
              "thermo_modify norm no format float %.17g", f"thermo {cfg['snapshot_every']}"]
    return lines


def thermostat(cfg, kind, T, name="integrate", prepare=False):
    command = f"fix {name} all {kind} temp {T:.16g} {T:.16g} $(100*dt)"
    if kind == "npt":
        multiplier = 500 if prepare else 1000
        command += f" iso {cfg['pressure']:.16g} {cfg['pressure']:.16g} $({multiplier}*dt)"
    return command


def snapshot(stage, file="snapshots.csv", attempts="0", accepts="0"):
    terms = [stage, "$(step:%.0f)", "$(count(all):%.0f)", "$(c_nb:%.0f)",
             "$(c_nb/count(all):%.17g)", "$(pe/count(all):%.17g)",
             "$((pe+ke+v_p_external*vol)/count(all):%.17g)", "$(enthalpy/count(all):%.17g)",
             "$(temp:%.17g)", "$(press:%.17g)", "$(vol:%.17g)", "$(c_mcsp:%.17g)",
             "$(c_ofrac:%.17g)", attempts, accepts]
    return f'print "{",".join(terms)}" append {file} screen no'


def dump_structure(path="structure.dump"):
    return (f"write_dump all custom {path} id type x y z c_csp "
            "modify sort id format float %.17g")


def prepare_input(cfg, phase, seed):
    lines = base_input(cfg)
    if phase == "liquid":
        schedule = [(cfg["prepare_temperature"], cfg["prepare_relax_steps"]),
                    (cfg["melt_temperature"], cfg["melt_steps"]),
                    (cfg["prepare_temperature"], cfg["prepare_relax_steps"])]
    else:
        schedule = [(cfg["prepare_temperature"], cfg["prepare_relax_steps"])]
    for index, (T, steps) in enumerate(schedule):
        lines += [f"velocity all create {T} {seed_for(seed, phase, 'prepare', index)} mom yes rot no dist gaussian",
                  thermostat(cfg, "npt", T, prepare=True), f"run {steps}", "unfix integrate"]
    lines += ["run 0", dump_structure(), "write_restart prepared.restart"]
    return "\n".join(lines) + "\n"


def initialize_input(cfg, case, prepared):
    lines = base_input(cfg, prepared)
    lines += ["set group all type 1"]
    if case["c0"] == 1:
        lines.append("set group all type 2")
    elif case["c0"] > 0:
        identity = [case[k] for k in ("phase", "T", "mu", "start", "seed")]
        lines.append(f"set group all type/ratio 2 {case['c0']:.17g} {seed_for(*identity, 'labels')}")
    identity = [case[k] for k in ("phase", "T", "mu", "start", "seed")]
    lines += [f"velocity all create {case['T']} {seed_for(*identity, 'velocity')} mom yes rot no dist gaussian",
              "reset_timestep 0", "run 0", f'print "{",".join(COLUMNS)}" file snapshots.csv screen no',
              snapshot("initial_unrelaxed"), thermostat(cfg, "npt", case["T"]),
              f"run {cfg['state_equil_steps']}", "unfix integrate", "reset_timestep 0", "run 0",
              snapshot("start"), dump_structure(), "write_restart initial.restart"]
    return "\n".join(lines) + "\n"


def cycle_input(cfg, case, restart, cycle, structure):
    lines = base_input(cfg, restart)
    lines.append(f'print "{",".join(COLUMNS)}" file snapshots.csv screen no')
    for block in range(cfg["mc_blocks_per_cycle"]):
        seed = seed_for(case["phase"], case["T"], case["mu"], case["start"], case["seed"], cycle, block, "mc")
        lines.append(thermostat(cfg, "nvt", case["T"]))
        if case["fix"] == "fast":
            command = (f"fix mc all fast/swap 1 $(count(all):%.0f) {case['T']} {seed} "
                       f"types 2 1 2 mu 0.0 {case['mu']}")
        else:
            command = (f"fix mc all atom/swap 1 $(count(all):%.0f) {seed} {case['T']} "
                       f"semi-grand yes ke no types 1 2 mu 0.0 {case['mu']}")
        # Exactly one invocation at the first step of a fixed-volume MD block.
        # Removing the fix prevents extra invocations or any NPT/MC overlap.
        lines += [command, "run 1", "variable da equal $(f_mc[1]:%.0f)",
                  "variable dc equal $(f_mc[2]:%.0f)", "unfix mc",
                  f"run {cfg['md_steps_per_mc_block'] - 1}",
                  snapshot("mc", attempts="$(v_da:%.0f)", accepts="$(v_dc:%.0f)"),
                  "variable da delete", "variable dc delete", "unfix integrate"]
    lines.append(thermostat(cfg, "npt", case["T"]))
    for _ in range(cfg["npt_steps_per_cycle"] // cfg["snapshot_every"]):
        lines += [f"run {cfg['snapshot_every']}", snapshot("npt")]
    lines += ["unfix integrate"]
    if structure:
        lines.append(dump_structure())
    lines.append("write_restart checkpoint.restart")
    return "\n".join(lines) + "\n"


def enthalpy_input(cfg, phase, T, seed, prepared):
    lines = base_input(cfg, prepared)
    lines += [f"velocity all create {T} {seed_for(phase, T, seed, 'enthalpy')} mom yes rot no dist gaussian",
              thermostat(cfg, "npt", T), f"run {cfg['reference_equil_steps']}",
              "unfix integrate", "reset_timestep 0", "run 0",
              f'print "{",".join(COLUMNS)}" file snapshots.csv screen no',
              thermostat(cfg, "npt", T)]
    for _ in range(cfg["reference_sample_steps"] // cfg["snapshot_every"]):
        lines += [f"run {cfg['snapshot_every']}", snapshot("enthalpy")]
    lines += ["unfix integrate", dump_structure(), "write_restart reference.restart"]
    return "\n".join(lines) + "\n"


class Runner:
    def __init__(self, args, cfg, output):
        self.args, self.cfg, self.output = args, cfg, output
        lmp = shutil.which(args.lmp) if args.lmp else None
        if not lmp:
            raise ValueError("Specify --lmp with an existing LAMMPS executable")
        self.lmp = str(Path(lmp).resolve())
        if not 1 <= args.ranks <= 24:
            raise ValueError("MPI rank count must be between 1 and 24")
        self.env = os.environ.copy()
        self.env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                         "LAMMPS_POTENTIALS": str(ROOT / "potential")})
        self.cpu_groups = getattr(args, "cpu_groups", [])
        self.cpu_group = self.cpu_groups[0] if self.cpu_groups else None
        self.stop_event = threading.Event()
        self.processes = set()
        self.process_lock = threading.Lock()
        if self.cpu_group:
            # The rank wrapper pins after launcher binding; Intel MPI must not
            # replace that affinity during MPI_Init in the LAMMPS executable.
            self.env["I_MPI_PIN"] = "0"
        if args.mpi_launcher == "direct":
            if args.ranks != 1:
                raise ValueError("direct launcher is only valid for one rank")
            self.prefix = []
        else:
            self.prefix = shlex.split(args.mpi_launcher)
            if not self.prefix or not shutil.which(self.prefix[0]):
                raise ValueError("MPI launcher not found; activate the correct MPI environment")
            self.prefix += [args.np_flag, str(args.ranks)]
        metadata = {"executable": self.lmp, "executable_sha256": sha(self.lmp), "ranks": args.ranks,
                    "launcher": self.prefix, "hostname": platform.node(), "platform": platform.platform(),
                    "python": sys.version,
                    "analysis_packages": {name: package_version(name) for name in ("numpy", "scipy", "matplotlib")},
                    "runtime_environment": {name: self.env.get(name) for name in
                        ("PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "I_MPI_ROOT", "I_MPI_FABRICS", "I_MPI_OFI_PROVIDER", "I_MPI_PIN", "FI_PROVIDER")},
                    "version": VERSION, "potential_files": {str(p.relative_to(ROOT)): sha(p)
                        for p in sorted((ROOT / "potential").glob("*")) if p.is_file()}}
        metadata["concurrent_jobs"] = getattr(args, "workers", 1)
        metadata["cpu_groups"] = self.cpu_groups
        metadata["scheduling_version"] = getattr(args, "scheduling_version", "legacy-sequential")
        metadata["workflow_source_sha256"] = sha(ROOT / "workflow_phase_diagram.py")
        if self.cpu_group:
            metadata["rank_wrapper_sha256"] = sha(ROOT / "rank_entry.py")
        if os.name != "nt" and shutil.which("ldd"):
            result = subprocess.run(["ldd", self.lmp], capture_output=True, text=True, env=self.env, timeout=30)
            metadata["ldd"] = result.stdout + result.stderr
            libs = re.findall(r"=>\s+(/\S+)\s+\(", result.stdout)
            metadata["loaded_library_hashes"] = {p: sha(p) for p in libs if Path(p).is_file()}
            if result.returncode or "not found" in result.stdout:
                raise ValueError("Unresolved shared libraries; see ldd output before running")
        binary_identity = {key: metadata.get(key) for key in
                           ("executable_sha256", "loaded_library_hashes", "ranks", "launcher", "potential_files", "version")}
        self.identity = digest(binary_identity)
        old = output / "environment.json"
        if old.exists() and json.loads(old.read_text())["run_identity"] != self.identity:
            raise ValueError("Binary/library/ranks/potential changed. Use a NEW --output; do not mix runs.")
        help_result = subprocess.run(self.command(["-h"]), capture_output=True, text=True,
                                     errors="replace", env=self.env, timeout=120)
        (output / "lammps_help.txt").write_text(help_result.stdout + help_result.stderr, encoding="utf-8")
        if help_result.returncode or "fast/swap" not in help_result.stdout:
            raise ValueError("LAMMPS/MPI preflight failed or fast/swap is absent; inspect lammps_help.txt")
        metadata.update(run_identity=self.identity, checked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(old, metadata)

    def command(self, arguments):
        executable = [self.lmp]
        if self.cpu_group:
            executable = [sys.executable, str(ROOT / "rank_entry.py"),
                          "--cpus", ",".join(map(str, self.cpu_group)), "--", self.lmp]
        return self.prefix + executable + arguments

    def for_slot(self, index):
        worker = copy.copy(self)
        worker.env = self.env.copy()
        worker.cpu_group = self.cpu_groups[index] if self.cpu_groups else None
        return worker

    @staticmethod
    def terminate(process):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                import signal
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        except ProcessLookupError:
            pass

    def abort_all(self):
        self.stop_event.set()
        with self.process_lock:
            for process in list(self.processes):
                self.terminate(process)

    def execute(self, folder, text, products):
        if self.stop_event.is_set():
            raise RuntimeError("Workflow stopped; resume from completed checkpoints")
        folder.mkdir(parents=True, exist_ok=True)
        # Relative paths also let Windows fopen access a Unicode workspace
        # through an ASCII input file, and keep inputs portable after copying.
        def relative_path(match):
            value = match.group(1)
            if Path(value).is_absolute():
                return '"' + Path(os.path.relpath(value, folder)).as_posix() + '"'
            return match.group(0)
        text = re.sub(r'"([^"\n]+)"', relative_path, text)
        fingerprint = digest({"identity": self.identity, "input": text})
        status_path = folder / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("fingerprint") != fingerprint:
                raise ValueError(f"Changed physics/input for existing job: {folder}. Use a new output root.")
            if status.get("state") == "complete":
                for name, expected in status["sha256"].items():
                    path = folder / name
                    if not path.is_file() or sha(path) != expected:
                        raise ValueError(f"Completed output missing/corrupt: {path}; do not silently reuse")
                return status
        (folder / "in.lammps").write_text(text, encoding="ascii")
        command = self.command(["-in", "in.lammps", "-log", "log.lammps", "-screen", "none"])
        start = time.perf_counter()
        atomic_json(status_path, {"state": "running", "fingerprint": fingerprint, "command": command})
        print(f"  running {folder.relative_to(self.output)}", flush=True)
        try:
            with (folder / "stdout.txt").open("w", encoding="utf-8") as stream:
                with self.process_lock:
                    if self.stop_event.is_set():
                        raise RuntimeError("Workflow stopped before segment launch")
                    process = subprocess.Popen(command, cwd=folder, env=self.env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                    self.processes.add(process)
                try:
                    returncode = process.wait(timeout=self.args.timeout)
                except BaseException:
                    self.terminate(process)
                    raise
                finally:
                    with self.process_lock:
                        self.processes.discard(process)
            if returncode:
                raise RuntimeError(f"LAMMPS exited {returncode}; inspect {folder / 'log.lammps'}")
            for product in products:
                if not (folder / product).is_file():
                    raise RuntimeError(f"Missing expected output: {folder / product}")
            if "snapshots.csv" in products:
                validate_snapshots(folder / "snapshots.csv", self.args.ranks)
            log = (folder / "log.lammps").read_text(errors="replace")
            if any(term in log.lower() for term in ("lost atoms", "non-numeric", "nan ", "error:")):
                raise RuntimeError(f"Unstable/error log in {folder}")
            hashes = {name: sha(folder / name) for name in products + ["log.lammps", "in.lammps", "stdout.txt"]}
            status = {"state": "complete", "fingerprint": fingerprint, "sha256": hashes,
                      "elapsed_seconds": time.perf_counter() - start,
                      "loop_seconds_sum": sum(float(v) for v in re.findall(r"Loop time of ([0-9.eE+-]+) on", log)),
                      "fallback_messages": log.count("parallel MC disabled permanently"),
                      "command": command}
            atomic_json(status_path, status)
            return status
        except BaseException as exc:
            atomic_json(status_path, {"state": "failed", "fingerprint": fingerprint, "error": str(exc),
                                      "elapsed_seconds": time.perf_counter() - start})
            raise


def validate_snapshots(path, ranks=24):
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No snapshots: {path}")
    if not set(COLUMNS).issubset(rows[0]):
        raise ValueError(f"Snapshot schema differs from this workflow version: {path}")
    for row in rows:
        numeric = [float(row[key]) for key in COLUMNS[1:]]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError(f"Non-finite snapshot in {path}")
        N, nb = int(row["N"]), int(row["n_B"])
        attempted, accepted = int(row["attempts_increment"]), int(row["accepts_increment"])
        if N <= 0 or not 0 <= nb <= N or abs(float(row["c"]) - nb / N) > 1e-12:
            raise ValueError(f"Inconsistent type counts in {path}")
        if not 0 <= accepted <= attempted:
            raise ValueError(f"Invalid acceptance counters in {path}")
        if row["stage"] == "mc" and not N <= attempted < N + ranks:
            raise ValueError(f"Expected one global attempted sweep (with bounded overshoot): {path}")
        if row["stage"] != "mc" and attempted:
            raise ValueError(f"Unexpected MC during NPT/reference stage: {path}")
    return rows


def run_workflow(args, cfg, output):
    check_protocol(cfg, output, create=True)
    runner = Runner(args, cfg, output)
    atomic_json(output / "effective_config.json", cfg)
    cases = make_cases(cfg)
    if args.filter:
        cases = [c for c in cases if args.filter in case_id(c)]
        if not cases:
            raise ValueError("--filter matched no trajectories")
    if getattr(args, "selected_states", None) is not None:
        cases = [c for c in cases if state_id(c["phase"], c["T"], c["mu"]) in args.selected_states]
    if getattr(args, "selected_case_ids", None) is not None:
        cases = [c for c in cases if case_id(c) in args.selected_case_ids]
    if not cases and args.stage in ("all", "sample"):
        raise ValueError("No selected trajectories")
    seeds = set(c["seed"] for c in cases)
    if args.stage in ("all", "reference", "prepare"):
        seeds.update(cfg["base_seeds"][:cfg["dense_seed_count"]])
    seeds = sorted(seeds)
    prepared = {}
    reuse = getattr(args, "reuse_temperature_output", None)
    if reuse is not None:
        reuse = Path(reuse)
        check_protocol(cfg, reuse)
        old_environment = json.loads((reuse / "environment.json").read_text())
        if old_environment["run_identity"] != runner.identity:
            raise ValueError("Reused data have different binary/library/ranks/potential; do not combine them")
    for phase, seed in itertools.product(PHASES, seeds):
        folder = (reuse if reuse is not None else output) / "prepared" / f"{phase}_seed{seed}"
        if reuse is not None:
            status = json.loads((folder / "status.json").read_text())
            if status.get("state") != "complete":
                raise ValueError(f"Reused prepared structure is incomplete: {folder}")
            for name, expected in status["sha256"].items():
                if not (folder / name).is_file() or sha(folder / name) != expected:
                    raise ValueError(f"Reused prepared output missing/corrupt: {folder / name}")
        else:
            runner.execute(folder, prepare_input(cfg, phase, seed), ["prepared.restart", "structure.dump"])
        prepared[phase, seed] = folder / "prepared.restart"
    if reuse is not None and args.stage != "sample":
        raise ValueError("Reusing prepared/reference data is only supported for sample runs")
    if args.stage in ("all", "reference"):
        for phase, T, seed in itertools.product(PHASES, cfg["reference_temperatures"],
                                                cfg["base_seeds"][:cfg["dense_seed_count"]]):
            folder = output / "reference" / f"{phase}_T{T:.2f}_seed{seed}"
            runner.execute(folder, enthalpy_input(cfg, phase, T, seed, prepared[phase, seed]),
                           ["snapshots.csv", "reference.restart", "structure.dump"])
    if args.stage not in ("all", "sample"):
        return
    def run_case(index, case, worker):
        planned_seeds = [seed_for(case["phase"], case["T"], case["mu"], case["start"], case["seed"], cycle, block, "mc")
                         for cycle in range(1, cfg["cycles"] + 1) for block in range(cfg["mc_blocks_per_cycle"])]
        if len(planned_seeds) != len(set(planned_seeds)):
            raise ValueError(f"Child-seed collision within {case_id(case)}; do not reuse the stream")
        print(f"[{index + 1}/{len(cases)}] {case_id(case)}", flush=True)
        folder = output / "trajectories" / case_id(case)
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / "case.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            if any(old[k] != case[k] for k in ("phase", "T", "mu", "start", "c0", "seed", "fix")):
                raise ValueError(f"Physical case metadata changed: {folder}")
        atomic_json(meta, case)
        init = folder / "initial"
        worker.execute(init, initialize_input(cfg, case, prepared[case["phase"], case["seed"]]),
                       ["snapshots.csv", "initial.restart", "structure.dump"])
        restart = init / "initial.restart"
        for cycle in range(1, cfg["cycles"] + 1):
            cycle_folder = folder / f"cycle_{cycle:04d}"
            structure = cycle % cfg["structure_dump_every_cycles"] == 0 or cycle == 1
            products = ["snapshots.csv", "checkpoint.restart"] + (["structure.dump"] if structure else [])
            worker.execute(cycle_folder, cycle_input(cfg, case, restart, cycle, structure), products)
            restart = cycle_folder / "checkpoint.restart"
    groups = {}
    for index, case in enumerate(cases):
        groups.setdefault(case["mu"], []).append((index, case))
    workers = getattr(args, "workers", 1)
    if not 1 <= workers or workers * args.ranks > 24:
        raise ValueError("Concurrent jobs times MPI ranks must be between 1 and 24")
    if workers > 1 and len(getattr(args, "cpu_groups", [])) != workers:
        raise ValueError("Concurrent MPI jobs require disjoint CPU groups; use temperature_workflow.py")
    slots = queue.Queue()
    for slot in range(workers):
        slots.put(runner.for_slot(slot))

    def run_group(group):
        worker = slots.get()
        try:
            for index, case in group:
                run_case(index, case, worker)
        finally:
            slots.put(worker)

    # A slot runs one mu group at a time. Long multi-start groups go first to
    # avoid leaving a single expensive group at the end of the queue.
    ordered = sorted(groups.values(), key=lambda group: (-len(group), group[0][1]["mu"]))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(run_group, group) for group in ordered]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        runner.abort_all()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    print("All selected trajectories completed. Starting numerical/structural diagnostics.", flush=True)


def correlated_stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 4 or not np.all(np.isfinite(values)):
        raise ValueError("At least four finite samples required")
    variance = float(np.var(values, ddof=1))
    tau = 1.0
    if variance > 1e-28:
        centered = values - values.mean()
        fft = np.fft.rfft(centered, n=2 * n)
        acov = np.fft.irfft(fft * fft.conjugate(), n=2 * n)[:n] / np.arange(n, 0, -1)
        rho = acov / acov[0]
        previous = float("inf")
        for lag in range(1, n - 1, 2):
            pair = float(rho[lag] + rho[lag + 1])
            if pair <= 0:
                break
            pair = min(pair, previous)
            tau += 2 * pair
            previous = pair
    tau = min(float(n), max(1.0, tau))
    sem = math.sqrt(variance * tau / n)
    blocks = []
    for width in (1, 2, 4, 8, 16, 32):
        nb = n // width
        if nb < 4:
            continue
        means = values[:nb * width].reshape(nb, width).mean(axis=1)
        candidate = float(np.std(means, ddof=1) / math.sqrt(nb))
        sem = max(sem, candidate)
        blocks.append((width, nb, candidate))
    df = max(1.0, n / tau - 1)
    # Do not describe zero variance as a large effective sample size.
    neff = n / tau if variance > 1e-28 else 0.0
    return {"mean": float(values.mean()), "sem": sem, "tau": tau,
            "n_effective": neff, "n_samples": n, "df": df,
            "ci95_halfwidth": float(student_t.ppf(0.975, df) * sem),
            "blocks": blocks, "constant": variance <= 1e-28}


def aggregate_replicas(items, key="mean_c", semkey="sem_c", dfkey="df_c"):
    values = np.array([item[key] for item in items], dtype=float)
    within_variance = sum(item[semkey] ** 2 for item in items) / len(items) ** 2
    between_variance = float(np.var(values, ddof=1) / len(items)) if len(items) > 1 else 0.0
    variance = max(within_variance, between_variance)
    if len(items) > 1 and between_variance >= within_variance:
        df = len(items) - 1
    else:
        denominator = sum((item[semkey] ** 2 / len(items) ** 2) ** 2 / item[dfkey] for item in items)
        df = within_variance ** 2 / denominator if denominator > 0 else 1.0
    sem = math.sqrt(variance)
    return float(np.mean(values)), sem, max(1.0, df)


def equivalence(a, b, margin):
    # Two one-sided tests represented by a 90% Welch interval, not a
    # failure to reject a zero-difference null hypothesis.
    delta = a[0] - b[0]
    variance = a[1] ** 2 + b[1] ** 2
    denom = a[1] ** 4 / a[2] + b[1] ** 4 / b[2]
    df = variance ** 2 / denom if denom > 0 else 1.0
    half = float(student_t.ppf(0.95, max(1.0, df)) * math.sqrt(variance))
    return delta, half, abs(delta) + half < margin


def composition_margin(cfg, state, N):
    old = state.get("legacy_c")
    if old is None:
        return cfg["composition_equivalence_margin"]
    return min(cfg["composition_equivalence_margin"],
               max(cfg["composition_equivalence_atom_floor"] / N,
                   cfg["composition_equivalence_relative_margin"] * old))


def load_trajectory(folder, cfg):
    case = json.loads((folder / "case.json").read_text())
    initial_status = folder / "initial" / "status.json"
    if not initial_status.exists() or json.loads(initial_status.read_text()).get("state") != "complete":
        return case, [], None, None
    initial = json.loads(initial_status.read_text())
    for name, expected in initial["sha256"].items():
        if not (folder / "initial" / name).exists() or sha(folder / "initial" / name) != expected:
            raise ValueError(f"Initial-state hash mismatch: {folder / 'initial' / name}")
    initial_rows = validate_snapshots(folder / "initial" / "snapshots.csv")
    actual_initial_c = float(initial_rows[-1]["c"])
    rows, attempts, accepts = [], 0, 0
    for cycle_number, cycle in enumerate(sorted(folder.glob("cycle_*")), start=1):
        if cycle.name != f"cycle_{cycle_number:04d}":
            raise ValueError(f"Missing cycle/checkpoint in {folder}")
        status_path = cycle / "status.json"
        if not status_path.exists():
            break
        status = json.loads(status_path.read_text())
        if status.get("state") != "complete":
            break
        for name, expected in status["sha256"].items():
            if not (cycle / name).exists() or sha(cycle / name) != expected:
                raise ValueError(f"Raw output hash mismatch: {cycle / name}")
        cycle_rows = validate_snapshots(cycle / "snapshots.csv")
        expected_rows = cfg["mc_blocks_per_cycle"] + cfg["npt_steps_per_cycle"] // cfg["snapshot_every"]
        if len(cycle_rows) != expected_rows or sum(r["stage"] == "mc" for r in cycle_rows) != cfg["mc_blocks_per_cycle"]:
            raise ValueError(f"Incomplete MC/NPT snapshot schedule: {cycle}")
        for row in cycle_rows:
            attempts += int(row["attempts_increment"])
            accepts += int(row["accepts_increment"])
            converted = {k: (v if k == "stage" else float(v)) for k, v in row.items()}
            converted.update(attempts_total=attempts, accepts_total=accepts,
                             attempted_sweeps=attempts / converted["N"])
            rows.append(converted)
    if not rows:
        return case, [], None, None
    steps = np.array([r["md_step"] for r in rows])
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Duplicated/nonincreasing trajectory steps: {folder}")
    if any(int(r["N"]) != 4 * int(np.prod(cfg["box_cells"])) for r in rows):
        raise ValueError(f"Atom count differs from configured FCC system: {folder}")
    cutoff = (1 - cfg["tail_fraction"]) * steps[-1]
    retained = [r for r in rows if r["md_step"] > cutoff]
    mc_retained = [r for r in retained if r["stage"] == "mc"]
    if len(mc_retained) < 8:
        return case, rows, None, None
    # Each whole cycle has equal-duration NVT and NPT records. The original
    # composition mean uses all snapshots; correlation diagnostics use MC
    # records, avoiding repeated unchanged identities during NPT.
    sc = correlated_stats([r["c"] for r in mc_retained])
    su = correlated_stats([r["U_per_atom"] for r in retained])
    all_c = np.array([r["c"] for r in retained])
    cycle_groups = {}
    for r in retained:
        cycle_number = int((r["md_step"] - 1) //
                           (cfg["mc_blocks_per_cycle"] * cfg["md_steps_per_mc_block"] + cfg["npt_steps_per_cycle"]))
        cycle_groups.setdefault(cycle_number, []).append(r["c"])
    # Include the long NPT hold in the mean-error estimate via cycle blocks.
    cm = np.array([np.mean(v) for v in cycle_groups.values()])
    cycle_sem = float(np.std(cm, ddof=1) / math.sqrt(len(cm))) if len(cm) >= 2 else sc["sem"]
    sem_c = max(sc["sem"], cycle_sem)
    df_c = min(sc["df"], max(1, len(cm) - 1)) if len(cm) >= 2 else sc["df"]
    half = len(mc_retained) // 2
    early = correlated_stats([r["c"] for r in mc_retained[:half]])
    late = correlated_stats([r["c"] for r in mc_retained[half:]])
    margin = composition_margin(cfg, case, int(rows[-1]["N"]))
    drift, drift_ci, stable = equivalence((early["mean"], early["sem"], early["df"]),
                                         (late["mean"], late["sem"], late["df"]),
                                         margin)
    da = sum(int(r["attempts_increment"]) for r in retained)
    dc = sum(int(r["accepts_increment"]) for r in retained)
    acceptance_stats = correlated_stats([r["accepts_increment"] / r["attempts_increment"] for r in mc_retained])
    result = {"case": case_id(case), "state": state_id(case["phase"], case["T"], case["mu"]),
              "phase": case["phase"], "T": case["T"], "mu": case["mu"], "start": case["start"],
              "initial_c_target": case["c0"], "seed": case["seed"], "fix": case["fix"], "N": int(rows[-1]["N"]),
              "actual_c_start": actual_initial_c,
              "md_steps": int(steps[-1]), "attempted_sweeps": rows[-1]["attempted_sweeps"],
              "retained_md_start": int(cutoff), "retained_mc_records": len(mc_retained),
              "retained_attempts": da, "retained_accepts": dc, "acceptance": dc / da if da else 0,
              "acceptance_sem_by_mc_call": acceptance_stats["sem"],
              "acceptance_ci95_halfwidth": acceptance_stats["ci95_halfwidth"],
              "mean_c": float(all_c.mean()), "sem_c": sem_c, "df_c": float(df_c),
              "ci95_c_halfwidth": float(student_t.ppf(0.975, df_c) * sem_c),
              "tau_c_mc_records": sc["tau"], "effective_mc_records_c": sc["n_effective"],
              "mean_U_per_atom": su["mean"], "sem_U_per_atom": su["sem"],
              "df_U_per_atom": su["df"],
              "mean_csp": float(np.mean([r["mean_csp"] for r in retained])),
              "ordered_fraction": float(np.mean([r["ordered_fraction"] for r in retained])),
              "tail_drift_c": drift, "tail_drift_ci90_halfwidth": drift_ci,
              "equivalence_margin_c": margin,
              "tail_drift_equivalent": bool(stable), "constant_c": bool(sc["constant"]),
              "complete": bool(steps[-1] >= cfg["minimum_cycles"] *
                               (cfg["mc_blocks_per_cycle"] * cfg["md_steps_per_mc_block"] + cfg["npt_steps_per_cycle"]))}
    block_width = max(1, int(math.ceil(sc["tau"])))
    return case, rows, result, {"all_c": all_c, "mc_c": np.array([r["c"] for r in mc_retained]),
                                "cycle_means": cm, "block_width": block_width,
                                "cycle_block_width": max(1, math.ceil(block_width / cfg["mc_blocks_per_cycle"]))}


def fit_mu(T, c, mu, degree=3):
    c, mu = np.asarray(c, dtype=float), np.asarray(mu, dtype=float)
    usable = (c > 0) & (c < 1) & np.isfinite(c) & np.isfinite(mu)
    c, mu = c[usable], mu[usable]
    if len(np.unique(c)) < degree + 2:
        raise ValueError("Insufficient distinct nonzero compositions for an overdetermined fit")
    if np.ptp(c) < 1e-6:
        raise ValueError("Composition range is too narrow")
    y = mu - T * np.log(c / (1 - c))
    # Work in a scaled polynomial basis to avoid ill-conditioning at dilute c.
    polynomial = np.polynomial.Polynomial.fit(c, y, degree).convert()
    coefficients = polynomial.coef
    predicted = chemical_potential(c, T, coefficients)
    singular_values = np.linalg.svd(np.polynomial.polynomial.polyvander(
        2 * (c - c.min()) / np.ptp(c) - 1, degree), compute_uv=False)
    if singular_values[-1] < 1e-10 * singular_values[0]:
        raise ValueError("Rank-deficient composition fit")
    return {"coefficients": coefficients.tolist(), "c_min": float(c.min()), "c_max": float(c.max()),
            "rmse_mu": float(np.sqrt(np.mean((predicted - mu) ** 2))), "n_points": len(c)}


def chemical_potential(c, T, coefficients):
    c = np.asarray(c)
    return T * np.log(c / (1 - c)) + np.polynomial.polynomial.polyval(c, coefficients)


def gibbs(c, T, coefficients, offset=0.0):
    c = np.asarray(c)
    result = offset + T * (xlogy(c, c) + xlogy(1 - c, 1 - c))
    for index, coefficient in enumerate(coefficients):
        result = result + coefficient * c ** (index + 1) / (index + 1)
    return result


def free_energy_offset(T, Tm, h_difference_coefficients):
    A, B = h_difference_coefficients
    return float(A * (Tm - T) / Tm + B * T * math.log(Tm / T))


def common_tangent(T, solid_fit, liquid_fit, offset):
    """Locate mixed-phase lower-hull edges, refine and check a supporting line.

    Supporting-line stability is checked on each phase's measured composition
    sector and its pure-A anchor. A dilute fit is not extrapolated to c=1.
    """
    cs = np.asarray(solid_fit["coefficients"])
    cl = np.asarray(liquid_fit["coefficients"])
    def phase_grid(fit):
        low, high = fit["c_min"], fit["c_max"]
        return np.unique(np.r_[0.0, np.linspace(low, high, 1025),
                               np.geomspace(max(low, 1e-10), high, 256)])
    solid_grid, liquid_grid = phase_grid(solid_fit), phase_grid(liquid_fit)
    points = [(float(c), float(g), 0) for c, g in zip(solid_grid, gibbs(solid_grid, T, cs))]
    points += [(float(c), float(g), 1) for c, g in zip(liquid_grid, gibbs(liquid_grid, T, cl, offset))]
    lowest = {}
    for c, g, label in points:
        if c not in lowest or g < lowest[c][0]:
            lowest[c] = (g, label)
    grid = np.array(sorted(lowest))
    energies = np.array([lowest[c][0] for c in grid])
    labels = np.array([lowest[c][1] for c in grid])
    hull = []
    for i in range(len(grid)):
        while len(hull) >= 2:
            j, k = hull[-2:]
            cross = ((grid[k] - grid[j]) * (energies[i] - energies[j]) -
                     (energies[k] - energies[j]) * (grid[i] - grid[j]))
            if cross > 0:
                break
            hull.pop()
        hull.append(i)
    candidates = []
    for i, j in zip(hull[:-1], hull[1:]):
        if labels[i] == labels[j] or grid[j] - grid[i] < 1e-6:
            continue
        initial = [grid[i], grid[j]] if labels[i] == 0 else [grid[j], grid[i]]
        initial = np.clip(initial, 1e-9, 1 - 1e-9)

        def residual(v):
            s, l = v
            mus, mul = chemical_potential(s, T, cs), chemical_potential(l, T, cl)
            return [float(mus - mul), float(gibbs(s, T, cs) - mus * s -
                                             gibbs(l, T, cl, offset) + mul * l)]

        solved = least_squares(residual, initial, bounds=(1e-10, 1 - 1e-10),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        s, l = solved.x
        if max(abs(x) for x in residual(solved.x)) > 1e-7 or abs(s - l) < 1e-7:
            continue
        mu = float(chemical_potential(s, T, cs))
        intercept = float(gibbs(s, T, cs) - mu * s)
        if min(np.min(gibbs(solid_grid, T, cs) - mu * solid_grid - intercept),
               np.min(gibbs(liquid_grid, T, cl, offset) - mu * liquid_grid - intercept)) < -2e-7:
            continue
        coverage = (solid_fit["c_min"] <= s <= solid_fit["c_max"] and
                    liquid_fit["c_min"] <= l <= liquid_fit["c_max"])
        if any(abs(s - previous["c_s"]) < 1e-7 and abs(l - previous["c_l"]) < 1e-7
               for previous in candidates):
            continue
        candidates.append({"c_s": float(s), "c_l": float(l), "mu_eq": mu,
                           "intercept": intercept, "endpoint_coverage": bool(coverage),
                           "stability_scope": "sampled phase composition sectors and pure-A anchors"})
    if not candidates:
        raise ValueError("No stable solid/liquid common tangent found for the fitted model")
    if len(candidates) != 1:
        raise ValueError("Multiple coexistence branches; requires explicit physical branch review")
    return candidates[0]


def block_resample(values, width, rng):
    values = np.asarray(values)
    width = min(len(values), max(1, width))
    starts = rng.integers(0, len(values), size=math.ceil(len(values) / width))
    indices = (starts[:, None] + np.arange(width)[None, :]) % len(values)
    return values[indices.ravel()[:len(values)]]


def analyze_reference(output, cfg, analysis):
    rows, curves = [], {}
    for phase, T, seed in itertools.product(PHASES, cfg["reference_temperatures"],
                                          cfg["base_seeds"][:cfg["dense_seed_count"]]):
        folder = output / "reference" / f"{phase}_T{T:.2f}_seed{seed}"
        status_path = folder / "status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("state") != "complete":
            continue
        status = json.loads(status_path.read_text())
        for name, expected in status["sha256"].items():
            if not (folder / name).exists() or sha(folder / name) != expected:
                raise ValueError(f"Reference output hash mismatch: {folder / name}")
        data = validate_snapshots(folder / "snapshots.csv")
        if any(int(r["n_B"]) != 0 for r in data):
            raise ValueError("Pure-A enthalpy reference contains B atoms")
        values = np.array([float(r["H_per_atom"]) for r in data])
        stats = correlated_stats(values)
        rows.append({"phase": phase, "T": T, "seed": seed, "mean_H_per_atom": stats["mean"],
                     "mean_H_thermo_per_atom": float(np.mean([float(r["H_thermo_per_atom"]) for r in data])),
                     "sem_H_per_atom": stats["sem"], "df_H": stats["df"],
                     "effective_records": stats["n_effective"],
                     "mean_csp": np.mean([float(r["mean_csp"]) for r in data]),
                     "ordered_fraction": np.mean([float(r["ordered_fraction"]) for r in data])})
        curves[phase, T, seed] = values
    write_csv(analysis / "pure_A_enthalpy_replicas.csv", rows)
    fits = {}
    for phase in PHASES:
        points = []
        for T in cfg["reference_temperatures"]:
            group = [r for r in rows if r["phase"] == phase and r["T"] == T]
            if len(group) != cfg["dense_seed_count"]:
                return rows, curves, None
            mean, sem, df = aggregate_replicas(group, "mean_H_per_atom", "sem_H_per_atom", "df_H")
            points.append({"phase": phase, "T": T, "H": mean, "sem": sem, "df": df})
        if len(points) < 3:
            return rows, curves, None
        polynomial = np.polynomial.Polynomial.fit([p["T"] for p in points], [p["H"] for p in points], 1).convert()
        fits[phase] = {"coefficients": polynomial.coef.tolist(), "points": points,
                       "rmse_H": float(np.sqrt(np.mean([(p["H"] - polynomial(p["T"])) ** 2 for p in points])))}
    atomic_json(analysis / "enthalpy_linear_fits.json", fits)
    return rows, curves, fits


def bootstrap_boundary(T, cfg, summaries, retained, reference_curves):
    rng = np.random.default_rng(seed_for(T, "phase-bootstrap"))
    results = []
    failures = {}
    base_seeds = cfg["base_seeds"][:cfg["dense_seed_count"]]
    for _ in range(cfg["bootstrap_replicates"]):
        groups = {}
        for s in summaries:
            if s["T"] == T and s["fix"] == "fast":
                groups.setdefault((s["phase"], s["mu"], s["start"]), []).append(s)
        phase_points = {p: [] for p in PHASES}
        # Resample seed clusters shared across state points; moving blocks
        # within each selected trajectory preserve temporal correlations.
        resampled_seeds = rng.choice(base_seeds, len(base_seeds), replace=True).tolist()
        state_samples = {}
        for (phase, mu, start), group in groups.items():
            available = {s["seed"]: s for s in group}
            chosen = [available[seed] for seed in resampled_seeds if seed in available]
            if len(available) == 1:
                chosen = list(available.values())
            elif not chosen:
                chosen = [group[int(rng.integers(len(group)))]]
            samples = []
            for s in chosen:
                trace = retained[s["case"]]
                # Cycle means include the NPT holds in the legacy time mean.
                samples.append(float(block_resample(trace["cycle_means"], trace["cycle_block_width"], rng).mean()))
            state_samples.setdefault((phase, mu), []).append(float(np.mean(samples)))
        for (phase, mu), values in state_samples.items():
            phase_points[phase].append((float(np.mean(values)), mu))
        try:
            fits = {phase: fit_mu(T, *zip(*points), cfg["polynomial_degree"])
                    for phase, points in phase_points.items()}
            href = {}
            for phase in PHASES:
                mean_H = []
                for temperature in cfg["reference_temperatures"]:
                    resamples = []
                    for seed in resampled_seeds:
                        values = reference_curves[phase, temperature, seed]
                        stats = correlated_stats(values)
                        resamples.append(block_resample(values, math.ceil(stats["tau"]), rng).mean())
                    mean_H.append(float(np.mean(resamples)))
                href[phase] = np.polynomial.Polynomial.fit(cfg["reference_temperatures"], mean_H, 1).convert().coef
            tm = cfg["melting_temperature"] + rng.normal(0, cfg["melting_temperature_sigma"])
            offset = free_energy_offset(T, tm, href["liquid"] - href["solid"])
            boundary = common_tangent(T, fits["solid"], fits["liquid"], offset)
            if boundary["endpoint_coverage"]:
                results.append([boundary["c_s"], boundary["c_l"]])
            else:
                failures["outside measured composition range"] = failures.get("outside measured composition range", 0) + 1
        except (ValueError, KeyError, FloatingPointError) as exc:
            key = str(exc)
            failures[key] = failures.get(key, 0) + 1
    if len(results) < max(20, int(0.8 * cfg["bootstrap_replicates"])):
        return {"valid_bootstraps": len(results), "bootstrap_failures": failures, "ci_reliable": False}
    intervals = np.percentile(results, [2.5, 97.5], axis=0)
    return {"valid_bootstraps": len(results), "bootstrap_failures": failures, "ci_reliable": True,
            "cs_ci_low": float(intervals[0, 0]), "cs_ci_high": float(intervals[1, 0]),
            "cl_ci_low": float(intervals[0, 1]), "cl_ci_high": float(intervals[1, 1])}


def analyze(cfg, output, trajectory_sources=None, reference_output=None):
    check_protocol(cfg, output)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    summaries, retained, trajectories = [], {}, {}
    environment_file = output / "environment.json"
    if not environment_file.exists() and trajectory_sources is not None and reference_output is not None:
        environment_file = Path(reference_output) / "environment.json"
    environment = json.loads(environment_file.read_text()) if environment_file.exists() else {}
    folders = (sorted((output / "trajectories").glob("*")) if trajectory_sources is None
               else [Path(trajectory_sources[name]) for name in sorted(trajectory_sources)])
    expected_cases = {case_id(case): case for case in make_cases(cfg)}
    if trajectory_sources is not None:
        atomic_json(analysis / "trajectory_sources.json",
                    {name: str(Path(folder).resolve()) for name, folder in trajectory_sources.items()})
    for folder in folders:
        if not (folder / "case.json").exists():
            continue
        if trajectory_sources is not None:
            actual = json.loads((folder / "case.json").read_text())
            expected = expected_cases.get(folder.name)
            if expected is None or any(actual[key] != expected[key]
                                       for key in ("phase", "T", "mu", "start", "c0", "seed", "fix")):
                raise ValueError(f"Reused/new case metadata do not match the analysis configuration: {folder}")
        case, rows, summary, tail = load_trajectory(folder, cfg)
        if rows:
            write_csv(analysis / "trajectories" / (folder.name + ".csv"), rows)
            trajectories[folder.name] = (case, rows)
        if summary:
            summary["ranks"] = environment.get("ranks", "unknown")
            if trajectory_sources is not None:
                summary["raw_source_folder"] = str(folder.resolve())
            summaries.append(summary)
            retained[summary["case"]] = tail
    write_csv(analysis / "trajectory_statistics.csv", summaries)
    reference_comparisons = []
    for state in cfg["reference_comparison_states"]:
        fast = [s for s in summaries if s["state"] == state and s["fix"] == "fast" and s["start"] == "middle"]
        atom = [s for s in summaries if s["state"] == state and s["fix"] == "atom" and s["start"] == "middle"]
        if not fast or not atom:
            continue
        af, aa = aggregate_replicas(fast), aggregate_replicas(atom)
        margin = fast[0]["equivalence_margin_c"]
        delta, half, passed = equivalence(af, aa, margin)
        uf = aggregate_replicas(fast, "mean_U_per_atom", "sem_U_per_atom", "df_U_per_atom")
        ua = aggregate_replicas(atom, "mean_U_per_atom", "sem_U_per_atom", "df_U_per_atom")
        reference_comparisons.append({"state": state, "fast_replicas": len(fast), "atom_replicas": len(atom),
                                      "delta_mean_c": delta, "ci90_halfwidth_c": half, "margin_c": margin,
                                      "composition_equivalent": bool(passed), "delta_mean_U_per_atom": uf[0] - ua[0],
                                      "sem_delta_U_per_atom": math.hypot(uf[1], ua[1]),
                                      "note": "Physical MD/MC comparison; not a requirement of stepwise parallel identity"})
    write_csv(analysis / "atom_swap_application_comparison.csv", reference_comparisons)
    state_stats, comparisons = [], []
    for state in cfg["states"]:
        sid = state_id(state["phase"], state["T"], state["mu"])
        group = [s for s in summaries if s["state"] == sid and s["fix"] == "fast"]
        starts = list(cfg["dense_starts"] if state.get("dense_validation", False) else cfg["ordinary_starts"])
        starts += sorted({s["start"] for s in group} - set(starts))
        required = cfg["dense_seed_count"] if state.get("dense_validation", False) else cfg["ordinary_seed_count"]
        complete = all(len([s for s in group if s["start"] == start and s["complete"]]) >= required for start in starts)
        aggregates = {start: aggregate_replicas([s for s in group if s["start"] == start])
                      for start in starts if any(s["start"] == start for s in group)}
        equivalent = complete
        N = int(group[0]["N"]) if group else 4 * int(np.prod(cfg["box_cells"]))
        margin = composition_margin(cfg, state, N)
        for left, right in itertools.combinations(starts, 2):
            if left not in aggregates or right not in aggregates:
                equivalent = False
                continue
            delta, half, passed = equivalence(aggregates[left], aggregates[right], margin)
            comparisons.append({"state": sid, "left": left, "right": right, "delta_mean_c": delta,
                                "ci90_halfwidth": half, "margin": margin,
                                "equivalence_demonstrated": passed})
            equivalent = equivalent and passed
        enough_samples = bool(group) and all(s["effective_mc_records_c"] >= cfg["minimum_effective_samples"] for s in group)
        no_drift = bool(group) and all(s["tail_drift_equivalent"] for s in group)
        result = {"state": sid, "phase": state["phase"], "T": state["T"], "mu": state["mu"],
                  "complete": bool(complete), "initial_state_equivalent": bool(equivalent),
                  "enough_effective_samples": enough_samples, "tail_drift_equivalent": no_drift,
                  "statistical_gate_passed": bool(complete and equivalent and enough_samples and no_drift),
                  "legacy_c": state.get("legacy_c"), "n_replicas": len(group), "equivalence_margin_c": margin}
        if aggregates:
            # Never combine separated initial-state plateaus for a phase fit.
            result["diagnostic_mean_c"] = float(np.mean([v[0] for v in aggregates.values()]))
            if state.get("legacy_c") is not None:
                result["delta_from_legacy_c"] = result["diagnostic_mean_c"] - state["legacy_c"]
        if result["statistical_gate_passed"]:
            result["fit_mean_c"] = float(np.mean([v[0] for v in aggregates.values()]))
        state_stats.append(result)
    write_csv(analysis / "state_convergence.csv", state_stats)
    write_csv(analysis / "initial_composition_equivalence.csv", comparisons)
    _, reference_curves, href = analyze_reference(Path(reference_output) if reference_output is not None else output,
                                                cfg, analysis)
    fits, boundaries, problems = {}, [], []
    for T in sorted({s["T"] for s in cfg["states"]}):
        temperature_states = [s for s in state_stats if s["T"] == T]
        usable_states = [s for s in temperature_states if s["statistical_gate_passed"]]
        if len(usable_states) != len(temperature_states):
            problems.append({"T": T, "reason": "Incomplete states or states failing conservative statistical diagnostics excluded from this gated fit; review flags do not prove nonequilibration.",
                             "excluded_states": [s["state"] for s in temperature_states if not s["statistical_gate_passed"]]})
        try:
            current = {}
            for phase in PHASES:
                points = [s for s in usable_states if s["phase"] == phase]
                current[phase] = fit_mu(T, [s["fit_mean_c"] for s in points], [s["mu"] for s in points],
                                        cfg["polynomial_degree"])
            fits[str(T)] = current
            if href is None:
                raise ValueError("Pure-A enthalpy references are incomplete")
            difference = np.array(href["liquid"]["coefficients"]) - np.array(href["solid"]["coefficients"])
            offset = free_energy_offset(T, cfg["melting_temperature"], difference)
            boundary = {"T": T, "delta_g_pure_A": offset,
                        **common_tangent(T, current["solid"], current["liquid"], offset)}
            if not boundary["endpoint_coverage"]:
                problems.append({"T": T, "reason": "Common-tangent endpoints outside measured composition ranges; add mu points.",
                                 "candidate_cs": boundary["c_s"], "candidate_cl": boundary["c_l"]})
                continue
            usable_ids = {s["state"] for s in usable_states}
            boundary.update(bootstrap_boundary(T, cfg, [s for s in summaries if s["state"] in usable_ids],
                                               retained, reference_curves))
            boundaries.append(boundary)
        except ValueError as exc:
            problems.append({"T": T, "reason": str(exc)})
    atomic_json(analysis / "chemical_potential_fits.json", fits)
    write_csv(analysis / "phase_boundaries.csv", boundaries)
    atomic_json(analysis / "phase_fit_diagnostics.json", problems)
    planned = make_cases(cfg)
    missing = [case_id(c) for c in planned if not any(s["case"] == case_id(c) and s["complete"] for s in summaries)]
    failed_states = [s["state"] for s in state_stats if not s["statistical_gate_passed"]]
    ready = (not cfg.get("verification_only", False) and not missing and not failed_states and not problems and
             all(b.get("ci_reliable", False) for b in boundaries) and
             len(boundaries) == len({s["T"] for s in cfg["states"]}) and
             all(cfg["author_confirmations"].values()))
    report = {"publication_ready_checks_passed": bool(ready), "planned_trajectories": len(planned),
              "analyzed_trajectories": len(summaries), "missing_trajectories": missing,
              "states_requiring_extension_or_review": failed_states, "phase_fit_problems": problems,
              "author_confirmations": cfg["author_confirmations"],
              "uncertainty_scope": "Conditional on supplied Tm and linear enthalpy model; does not include polynomial model error.",
              "verification_only": cfg.get("verification_only", False),
              "note": "Statistical gates and CSP proxies are diagnostics, not a proof of ergodicity or phase identity."}
    atomic_json(analysis / "revision_readiness.json", report)
    make_plots(cfg, analysis, trajectories, summaries, state_stats, fits, boundaries, href)
    make_si_fragment(cfg, analysis, summaries, boundaries, report)
    print(f"Analyzed {len(summaries)} trajectories. Conservative diagnostic review flags for {len(failed_states)} states; not a count of proven unconverged states.")
    print(f"Report: {analysis / 'revision_readiness.json'}")
    return report


def make_plots(cfg, analysis, trajectories, summaries, state_stats, fits, boundaries, href):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    figure_manifest = figures / "generated_figures.json"
    if figure_manifest.exists():
        for name in json.loads(figure_manifest.read_text()):
            path = figures / name
            if (Path(name).name == name and name.startswith("Fig_") and
                    path.suffix in (".pdf", ".png") and path.parent.resolve() == figures.resolve() and path.exists()):
                path.unlink()
    generated_figures = []
    colors = {"poor": "#0072B2", "middle": "#009E73", "rich": "#D55E00"}

    def save(fig, name):
        fig.savefig(figures / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(figures / (name + ".png"), bbox_inches="tight", dpi=300)
        generated_figures.extend([name + ".pdf", name + ".png"])
        plt.close(fig)

    # Dense but readable SI: no more than four chemical potentials per page.
    for T, phase in itertools.product(sorted({s["T"] for s in cfg["states"]}), PHASES):
        states = sorted((s for s in cfg["states"] if s["T"] == T and s["phase"] == phase), key=lambda s: s["mu"])
        for page in range(math.ceil(len(states) / 4)):
            selected = states[4 * page:4 * (page + 1)]
            relevant = [(c, r) for c, r in trajectories.values() if c["T"] == T and c["phase"] == phase and c["fix"] == "fast"]
            if not relevant:
                continue
            fig, axes = plt.subplots(3, len(selected), figsize=(3.2 * len(selected), 7), squeeze=False, constrained_layout=True)
            for column, state in enumerate(selected):
                for case, rows in relevant:
                    if case["mu"] != state["mu"]:
                        continue
                    x = np.array([r["md_step"] for r in rows]) / 1e5
                    for row, key in enumerate(("c", "U_per_atom", "ordered_fraction")):
                        axes[row, column].plot(x, [r[key] for r in rows], color=colors.get(case["start"], "black"), lw=0.7, alpha=0.7)
                axes[0, column].set_title(f"{phase}; T={T:.2f}; mu={state['mu']:.1f}", fontsize=9)
                axes[2, column].set_xlabel("MD steps / $10^5$")
            for row, label in enumerate(("B fraction", "$U/N$ ($\\epsilon_{AA}$)", "CSP below threshold fraction")):
                axes[row, 0].set_ylabel(label)
            handles = [plt.Line2D([], [], color=color, label=start) for start, color in colors.items()]
            fig.legend(handles=handles, loc="outside upper center" if tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 7) else "upper center",
                       ncol=3, bbox_to_anchor=(0.5, 1.04))
            save(fig, f"Fig_SI_convergence_{phase}_T{T:.2f}_page{page + 1}")
    if summaries:
        representative = []
        for phase in PHASES:
            options = [s for s in cfg["states"] if s["phase"] == phase and any(
                c["phase"] == phase and c["T"] == s["T"] and c["mu"] == s["mu"] for c, _ in trajectories.values())]
            if options:
                representative.append(min(options, key=lambda s: abs(s["T"] - 0.62) + 0.001 * abs(s["mu"] + 2.2)))
        fig, axes = plt.subplots(2, len(representative), figsize=(5.8 * max(1, len(representative)), 6), squeeze=False, constrained_layout=True)
        for column, state in enumerate(representative):
            for case, rows in trajectories.values():
                if any(case[k] != state[k] for k in ("phase", "T", "mu")) or case["fix"] != "fast":
                    continue
                x = np.array([r["md_step"] for r in rows]) / 1e5
                for row, key in enumerate(("c", "U_per_atom")):
                    axes[row, column].plot(x, [r[key] for r in rows], color=colors.get(case["start"], "black"), lw=0.8, alpha=0.75)
            axes[0, column].set_title(f"{state['phase']}; T={state['T']:.2f}; mu={state['mu']:.1f}")
            axes[1, column].set_xlabel("MD steps / $10^5$")
        axes[0, 0].set_ylabel("B fraction")
        axes[1, 0].set_ylabel("$U/N$ ($\\epsilon_{AA}$)")
        handles = [plt.Line2D([], [], color=color, label=start) for start, color in colors.items()]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
        save(fig, "Fig_Main_application_multistart")
    if href:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for phase, color in zip(PHASES, ("#0072B2", "#D55E00")):
            points = href[phase]["points"]
            ax.errorbar([p["T"] for p in points], [p["H"] for p in points],
                        yerr=[student_t.ppf(0.975, p["df"]) * p["sem"] for p in points],
                        fmt="o", color=color, capsize=3, label=phase)
            x = np.linspace(min(p["T"] for p in points), max(p["T"] for p in points), 100)
            ax.plot(x, np.polynomial.polynomial.polyval(x, href[phase]["coefficients"]), color=color)
        ax.set_xlabel("$T$ ($\\epsilon_{AA}/k_B$)")
        ax.set_ylabel("$H/N$ ($\\epsilon_{AA}$)")
        ax.legend()
        save(fig, "Fig_SI_pure_A_enthalpy")
    if fits:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, (T_string, pairfits) in zip(axes.flat, fits.items()):
            T = float(T_string)
            for phase, color in zip(PHASES, ("#0072B2", "#D55E00")):
                points = [s for s in state_stats if s["T"] == T and s["phase"] == phase and "fit_mean_c" in s]
                ax.plot([s["fit_mean_c"] for s in points], [s["mu"] for s in points], "o", color=color, ms=3, label=phase)
                fit = pairfits[phase]
                x = np.linspace(fit["c_min"], fit["c_max"], 300)
                ax.plot(x, chemical_potential(x, T, fit["coefficients"]), color=color)
            ax.set_title(f"T={T:.2f}")
            ax.set_xlabel("B fraction")
            ax.set_ylabel("$\\Delta\\mu$ ($\\epsilon_{AA}$)")
        axes.flat[0].legend()
        save(fig, "Fig_SI_mu_composition_fits")
    if boundaries:
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        for phase, key, color in (("solid", "c_s", "#0072B2"), ("liquid", "c_l", "#D55E00")):
            ordered = sorted(boundaries, key=lambda b: b["T"])
            ax.plot([b[key] for b in ordered], [b["T"] for b in ordered], "o-", color=color, label=phase)
            for b in ordered:
                prefix = "cs" if phase == "solid" else "cl"
                if b.get("ci_reliable", False):
                    ax.plot([b[prefix + "_ci_low"], b[prefix + "_ci_high"]], [b["T"]] * 2, color=color, lw=2)
        ax.plot(0, cfg["melting_temperature"], "ks", label="supplied pure-A melting point")
        legacy = ROOT / "legacy_data" / "phase_diagram.txt"
        if legacy.exists():
            old = np.loadtxt(legacy, ndmin=2)
            ax.plot(old[:, 1], old[:, 0], ":", color="#777777", label="historical phase boundaries")
            ax.plot(old[:, 2], old[:, 0], ":", color="#777777")
        ax.set_xlabel("B fraction")
        ax.set_ylabel("$T$ ($\\epsilon_{AA}/k_B$)")
        ax.legend(loc="best")
        save(fig, "Fig_Main_phase_diagram_revision")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, boundary in zip(axes.flat, sorted(boundaries, key=lambda b: b["T"])):
            T = boundary["T"]
            pairfits = fits[str(T)]
            for phase, color in zip(PHASES, ("#0072B2", "#D55E00")):
                fit = pairfits[phase]
                x = np.linspace(fit["c_min"], fit["c_max"], 400)
                offset = boundary["delta_g_pure_A"] if phase == "liquid" else 0
                ax.plot(x, gibbs(x, T, fit["coefficients"], offset), color=color, label=phase)
            x = np.array([boundary["c_s"], boundary["c_l"]])
            ax.plot(x, boundary["intercept"] + boundary["mu_eq"] * x, "k--", lw=1)
            ax.set_title(f"T={T:.2f}")
            ax.set_xlabel("B fraction")
            ax.set_ylabel("Relative $g$ ($\\epsilon_{AA}$)")
        axes.flat[0].legend()
        save(fig, "Fig_SI_common_tangents")
    atomic_json(figure_manifest, generated_figures)


def make_si_fragment(cfg, analysis, summaries, boundaries, report):
    text = [r"% Generated fragment; requires graphicx, booktabs, longtable.",
            r"\subsection{Initial-composition sensitivity of the phase-diagram application}",
            r"The binary model uses \texttt{lj/cut} with $r_c=3.0$, "
            r"$\epsilon_{AA}=\epsilon_{AB}=\epsilon_{BB}=1$, "
            r"$\sigma_{AA}=1.0$, $\sigma_{AB}=1.1$, $\sigma_{BB}=1.2$, and equal unit masses. "
            r"No energy shifting or tail corrections are enabled. Temperature is in $\epsilon_{AA}/k_{\mathrm B}$.",
            f"The supplied pure-A melting temperature is $T_m={cfg['melting_temperature']}$. "
            "Its two-phase coexistence calculation is not repeated by this workflow.",
            f"Each cycle contains {cfg['mc_blocks_per_cycle']} NVT blocks of {cfg['md_steps_per_mc_block']} MD steps, "
            f"with one all-atom attempted MC sweep at the first step of each block, followed by "
            f"{cfg['npt_steps_per_cycle']} NPT steps without MC. "
            "The actual attempted and accepted counts are recorded; a parallel final batch may exceed one nominal sweep slightly.",
            "The MC fix is removed before every NPT segment. A distinct deterministic child seed is assigned to each MC block. "
            "The final sampling window is an analysis choice, not an automatic equilibration criterion.",
            r"For thermodynamic integration at imposed pressure, the sampled enthalpy is "
            r"$H=E+P_{\rm ext}V$, including kinetic energy. The instantaneous-pressure thermo enthalpy is retained separately as a diagnostic.",
            f"Means use the final {100 * cfg['tail_fraction']:.0f}\\% of the trajectory. "
            "Autocorrelation and block analyses avoid treating repeated unchanged composition records during NPT as independent MC samples. "
            "Initial-composition equivalence uses a 90\\% Welch interval. The prescribed state-dependent margin is "
            r"$\delta_c=\min[\delta_{\max},\max(n_{\min}/N,f_c c_{\rm old})]$, with "
            f"$\\delta_{{\\max}}={cfg['composition_equivalence_margin']}$, "
            f"$n_{{\\min}}={cfg['composition_equivalence_atom_floor']}$ and "
            f"$f_c={cfg['composition_equivalence_relative_margin']}$; "
            "these tolerances are diagnostics that require author review relative to phase-boundary resolution. "
            "Structural CSP records and coordinate dumps require physical phase review.",
            "Cubic chemical-potential fits use a numerically scaled polynomial basis. Common tangents must be supporting lines "
            "of both fitted phase curves and have endpoints inside sampled composition ranges. "
            "Block/replica resampling propagates sampling uncertainty to phase boundaries; "
            "these intervals do not include polynomial model error or unspecified melting-temperature uncertainty.",
            r"% REVIEW STATUS: " + ("all automatic/author checks passed" if report["publication_ready_checks_passed"] else "NOT submission-ready; inspect revision_readiness.json"),
            r"{\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{llrlrrrrr}", r"\caption{Retained-window fast/swap application statistics; replica labels map to the base seeds in the configuration. Atom numbers, MPI ranks and acceptance uncertainties are given in the accompanying CSV.}\label{tab:application_acceptance}\\",
            r"\toprule Phase & $T$ & $\Delta\mu$ & Replica & $c_0$ & $N_{\rm att}$ & $N_{\rm acc}$ & $a$ & $\bar c$\\",
            r"\midrule\endfirsthead", r"\toprule Phase & $T$ & $\Delta\mu$ & Replica & $c_0$ & $N_{\rm att}$ & $N_{\rm acc}$ & $a$ & $\bar c$\\",
            r"\midrule\endhead"]
    for s in summaries:
        if s["fix"] != "fast":
            continue
        label = s["start"] + "/r" + str(cfg["base_seeds"].index(s["seed"]) + 1)
        text.append(f"{s['phase']} & {s['T']:.2f} & {s['mu']:.3g} & {label} & {s['initial_c_target']:.3f} & "
                    f"{s['retained_attempts']} & {s['retained_accepts']} & {s['acceptance']:.3f} & "
                    f"{s['mean_c']:.6f} \\\\")
    text += [r"\bottomrule\end{longtable}", "}", "% Seed-level identities, uncertainty and actual sweep budgets are in trajectory_statistics.csv."]
    (analysis / "application_SI_fragment.tex").write_text("\n\n".join(text) + "\n", encoding="ascii")


def plan(cfg, output, workers=1):
    cases = make_cases(cfg)
    rows = [{"id": case_id(c), **{k: c[k] for k in ("phase", "T", "mu", "start", "c0", "seed", "fix")}}
            for c in cases]
    write_csv(output / "planned_trajectories.csv", rows)
    counts = {"historical_state_points": len(cfg["states"]),
              "dense_validation_states": sum(s.get("dense_validation", False) for s in cfg["states"]),
              "phase_trajectories": len(cases), "cycles_per_trajectory": cfg["cycles"],
              "nominal_mc_sweeps_per_trajectory": cfg["cycles"] * cfg["mc_blocks_per_cycle"],
              "production_md_steps_per_trajectory": cfg["cycles"] *
                (cfg["mc_blocks_per_cycle"] * cfg["md_steps_per_mc_block"] + cfg["npt_steps_per_cycle"]),
              "reference_jobs": len(PHASES) * len(cfg["reference_temperatures"]) * cfg["dense_seed_count"],
              "maximum_concurrent_lammps_jobs": workers}
    atomic_json(output / "plan.json", counts)
    print(json.dumps(counts, indent=2))


def check_historical(cfg, output):
    """Reanalyze genuine old data without treating them as revision repeats."""
    folder = output / "historical_check"
    folder.mkdir(exist_ok=True)
    h_coefficients = {}
    for phase in PHASES:
        data = np.loadtxt(ROOT / "legacy_data" / (phase + ".H"), ndmin=2)
        h_coefficients[phase] = np.polynomial.Polynomial.fit(data[:, 0], data[:, 1], 1).convert().coef
    old_boundaries = np.loadtxt(ROOT / "legacy_data" / "phase_diagram.txt", ndmin=2)
    fits, rows = {}, []
    for T in sorted({s["T"] for s in cfg["states"]}):
        try:
            current = {}
            for phase in PHASES:
                states = [s for s in cfg["states"] if s["T"] == T and s["phase"] == phase and s.get("legacy_c") is not None]
                current[phase] = fit_mu(T, [s["legacy_c"] for s in states], [s["mu"] for s in states])
            fits[str(T)] = current
            offset = free_energy_offset(T, cfg["melting_temperature"], h_coefficients["liquid"] - h_coefficients["solid"])
            result = common_tangent(T, current["solid"], current["liquid"], offset)
            old = old_boundaries[np.argmin(abs(old_boundaries[:, 0] - T))]
            rows.append({"T": T, **result, "archived_c_s": float(old[1]), "archived_c_l": float(old[2]),
                         "rmse_mu_solid": current["solid"]["rmse_mu"], "rmse_mu_liquid": current["liquid"]["rmse_mu"]})
        except ValueError as error:
            rows.append({"T": T, "diagnostic_error": str(error)})
    write_csv(folder / "legacy_phase_reconstruction.csv", rows)
    atomic_json(folder / "legacy_fits.json", fits)
    atomic_json(folder / "STATUS.json", {"historical_diagnostic_only": True,
                "note": "Genuine old data; no new simulations, independent repeats, acceptance counters or confidence intervals implied.",
                "Tm_used": cfg["melting_temperature"], "archived_phase_file_pure_A_endpoint": float(old_boundaries[0, 0]),
                "old_md_steps_per_trajectory": sorted({s["legacy_md_steps"] for s in cfg["states"] if "legacy_md_steps" in s})})
    print(f"Historical data diagnostic: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "plan", "run", "analyze", "legacy-check"))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--lmp")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--mpi-launcher", default="mpirun")
    parser.add_argument("--np-flag", default="-np")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--filter", help="Substring of trajectory id; prerequisite reference jobs still run")
    parser.add_argument("--stage", choices=("all", "prepare", "reference", "sample"), default="all")
    parser.add_argument("--timeout", type=int, default=86400, help="Per-segment timeout seconds")
    parser.add_argument("--include-reference", action="store_true", help="Add selected atom/swap MD/MC comparators")
    parser.add_argument("--auto-extend", action="store_true", help="Extend statistically flagged states, up to --max-cycles")
    parser.add_argument("--max-cycles", type=int, default=40)
    args = parser.parse_args()
    if args.command == "init":
        if args.original is None:
            parser.error("init requires --original")
        initialize_bundle(args.original)
        return
    output = args.output.resolve()
    config_path = args.config or ((output / "effective_config.json")
                                 if args.command == "analyze" and (output / "effective_config.json").exists()
                                 else ROOT / "config.json")
    cfg = load_config(config_path)
    if args.cycles is not None:
        if args.cycles <= 0:
            parser.error("--cycles must be positive")
        cfg["cycles"] = args.cycles
    if args.include_reference:
        cfg["include_atom_swap_reference"] = True
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        plan(cfg, output)
        if args.command == "run":
            if args.auto_extend and (args.filter or args.stage not in ("all", "sample")):
                parser.error("--auto-extend requires an unfiltered all/sample run")
            if args.auto_extend and args.max_cycles < cfg["cycles"]:
                parser.error("--max-cycles cannot be smaller than the initial target --cycles")
            run_workflow(args, cfg, output)
            if args.stage in ("all", "sample"):
                report = analyze(cfg, output)
                while args.auto_extend and report["states_requiring_extension_or_review"] and cfg["cycles"] < args.max_cycles:
                    args.selected_states = set(report["states_requiring_extension_or_review"])
                    cfg["cycles"] = min(cfg["cycles"] + 20, args.max_cycles)
                    args.stage = "sample"
                    print(f"Extending {len(args.selected_states)} flagged states to {cfg['cycles']} cycles; no new simultaneous jobs.", flush=True)
                    run_workflow(args, cfg, output)
                    report = analyze(cfg, output)
        elif args.command == "analyze":
            analyze(cfg, output)
        elif args.command == "legacy-check":
            check_historical(cfg, output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
