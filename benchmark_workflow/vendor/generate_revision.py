"""Revision tests: explicit energies, global attempt counts, and fixed MPI geometry."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex

import models

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.name: c for c in models.CASES}
CASES["buck"] = dataclasses.replace(CASES["buck_coul_cut"], name="buck", atom_style="atomic", charges=None,
    description="standalone Buckingham", pair_style="buck 2.4")
CASES["yukawa"] = dataclasses.replace(CASES["lj_cut"], name="yukawa", description="standalone Yukawa",
    pair_style="yukawa 1.2 2.4", pair_coeffs=("pair_coeff 1 1 1.00 2.4", "pair_coeff 1 2 1.10 2.4", "pair_coeff 2 2 0.85 2.4"))


def validate_config(cfg):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    if not 1 <= cfg["max_ranks"] <= 24:
        raise ValueError("Use 1--24 MPI ranks")
    if len(set(cfg["seeds"])) < 3 or any(not 0 < s < 900000000 for s in cfg["seeds"]):
        raise ValueError("At least three distinct positive seeds below 900000000 are required")
    if cfg["timing"]["minimum_loop_seconds"] < 5 or cfg["timing"]["repeats"] < 3:
        raise ValueError("Production timing needs >=5 seconds and >=3 repeats")
    if cfg["exact"]["sweeps"] < 160 or cfg["parallel"]["sweeps"] < 160:
        raise ValueError("Use at least 160 sweeps; the default is deliberately longer")
    if set(cfg["models"]) - CASES.keys():
        raise ValueError("Unknown model in config")
    for stage in ("exact", "parallel", "hybrid", "closure"):
        box = cfg[stage]["box"]
        if len(box) != 3 or any(not isinstance(x, int) or x < 1 for x in box):
            raise ValueError("Boxes must have three positive integer FCC cell counts")
    if any(not 0 <= f < 1 for f in cfg["analysis"]["discard_fractions"]):
        raise ValueError("Discard fractions must be in [0,1)")


def geometry(box, ranks, radius=2.4, skin=0.3, require_parallel=True):
    a = (4 / models.LATTICE_RHO) ** (1 / 3)
    lengths = [a * n for n in box]
    if min(lengths) <= 2 * (radius + skin):
        raise ValueError("Box allows ambiguous self-image coverage; enlarge it")
    half_domain_z = lengths[2] / ranks / 2
    if ranks > 1 and require_parallel and half_domain_z <= radius + skin:
        raise ValueError(f"Parallel fallback risk: Lz/(2p)={half_domain_z:.6g} <= {radius + skin}")
    return {"box_lengths": lengths, "processors": [1, 1, ranks], "max_pair_cutoff": radius,
            "neighbor_skin": skin, "half_domain_z": half_domain_z,
            "geometric_margin": half_domain_z - radius - skin if ranks > 1 else None}


def setup(case, box, ranks, modifier=(), newton=None, initial_data=None):
    rows = ["clear", "units lj", "dimension 3", "boundary p p p", f"atom_style {case.atom_style}",
            f"processors 1 1 {ranks}"]
    if newton:
        rows.append(f"newton {newton}")
    if initial_data:
        rows += [f"read_data {initial_data}"]
    else:
        rows += [f"lattice fcc {models.LATTICE_RHO}", f"region box block 0 {box[0]} 0 {box[1]} 0 {box[2]} units lattice",
                 f"create_box {case.ntypes} box", "create_atoms 1 box", *models.init_regions(case, *box)]
    rows += [f"mass {i} {mass:.17g}" for i, mass in enumerate(case.masses, 1)]
    if case.charges:
        rows += [f"set type {i} charge {q:.17g}" for i, q in enumerate(case.charges, 1)]
    rows += [f"pair_style {case.pair_style}", *case.pair_coeffs, *modifier,
             "neighbor 0.3 bin", "neigh_modify delay 0 every 1 check yes"]
    return rows


def render_mc(spec, sweeps=None):
    case = CASES[spec["case"]]
    n = spec["natoms"]
    sweeps = spec["requested_sweeps"] if sweeps is None else sweeps
    rows = setup(case, spec["box"], spec["ranks"], tuple(spec.get("pair_modifiers", [])), spec.get("newton"), spec.get("initial_data_path"))
    if spec.get("guard") == "charge":
        rows += ["set type 2 charge 0.4"]
    if spec.get("guard") == "kspace":
        rows += ["pair_style lj/cut/coul/long 2.4", "pair_coeff * * 1.0 1.0", "kspace_style pppm 1e-4"]
    if spec.get("guard") == "molecular":
        # atom_style full still has no bonds, but the fix must reject molecular topology capability.
        rows = [r.replace("atom_style atomic", "atom_style full") for r in rows]
    if spec.get("guard") == "manybody":
        rows = [r.replace("units lj", "units metal") for r in rows]
        rows += ["pair_style sw", "pair_coeff * * fixtures/Si.sw Si Si"]
    hybrid = spec["stage"] == "hybrid"
    if hybrid:
        rows += [f"timestep {spec['timestep']:.17g}", f"velocity all create {spec['temperature']:.17g} {spec['seed']} mom yes rot no dist gaussian",
                 f"fix integrate all nvt temp {spec['temperature']:.17g} {spec['temperature']:.17g} 0.2"]
    else:
        rows += ["timestep 0.0", "fix integrate all nve"]
    rows += [f"variable ncycle equal {n}", f"variable seed equal {spec['seed']}",
             f"variable tmc equal {spec.get('temperature', 1.0):.17g}", "compute ctype all count/type atom"]
    fix = models.fix_line(case, spec["kind"])
    if hybrid:
        fix = fix.replace("all atom/swap 1 ", f"all atom/swap {spec['mc_every']} ").replace("all fast/swap 1 ", f"all fast/swap {spec['mc_every']} ")
    rows += [fix, "variable epa equal pe/atoms"]
    countcols = " ".join(f"c_ctype[{i}]" for i in range(1, case.ntypes + 1))
    rows += [f"thermo_style custom step atoms pe v_epa {countcols} f_mc[1] f_mc[2] temp press vol",
             "thermo_modify norm no format float %.17g"]
    every = spec.get("mc_every", 1)
    if spec["stage"] == "timing":
        warmup = spec["warmup_sweeps"]
        rows += [f"thermo {max(1, warmup)}", f"run {warmup}", "print \"REVISION_PRODUCTION_BEGIN\"",
                 f"thermo {sweeps}", f"run {sweeps}", "print \"REVISION_PRODUCTION_END\""]
    else:
        dump_every = spec.get("type_dump_every", 0)
        if dump_every:
            rows += [f"dump labels all custom {dump_every * every} results/raw/{spec['id']}.types.dump id type",
                     "dump_modify labels sort id"]
        rows += [f"thermo {every}", "print \"REVISION_PRODUCTION_BEGIN\"", f"run {sweeps * every}",
                 "print \"REVISION_PRODUCTION_END\"",
                 f"write_dump all custom results/raw/{spec['id']}.final.dump id type x y z modify sort id format float %.17g"]
        if spec["stage"] == "prepare":
            rows += [f"write_data results/raw/{spec['id']}.data nocoeff"]
    return "\n".join(rows) + "\n"


def render_closure(spec):
    case = CASES[spec["case"]]
    rows = setup(case, spec["box"], 1)
    rows += ["timestep 0.0", "fix integrate all nve", "thermo_style custom step atoms pe",
             "thermo_modify norm no format float %.17g", "thermo 1", "run 0",
             f"write_dump all custom results/raw/{spec['id']}.initial.dump id type modify sort id",
             "compute ids all property/local patom1 patom2", "compute ep all pair/local eng"]
    rng = random.Random(spec["seed"])
    # Each trial starts from type 1 on a selected site, changes to type 2/3, and rolls back.
    # This diagnostic is NOT a Markov chain or an acceptance-rate measurement.
    for trial in range(spec["trials"]):
        atom_id = rng.randint(1, spec["natoms"])
        trial_type = rng.choice(case.swap_types[1:])
        for label, atom_type in (("before", 1), ("after", trial_type)):
            rows += [f"set atom {atom_id} type {atom_type}",
                     f"dump pair_dump all local 1 results/raw/{spec['id']}.{trial:04d}.{label}.pairs index c_ids[1] c_ids[2] c_ep",
                     "dump_modify pair_dump format float %.17g", "run 0", "undump pair_dump", "variable eg equal pe",
                     f'print "CLOSURE {trial} {atom_id} 1 {trial_type} {label} $(v_eg:%.17g)"',
                     "variable eg delete"]
        rows += [f"set atom {atom_id} type 1"]
    return "\n".join(rows) + "\n"


def application_dependencies(paths):
    hashes = {}
    scanned = set()
    pending = list(paths)
    while pending:
        path = (ROOT / pending.pop()).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Missing or external application dependency: {path}")
        relative = str(path.relative_to(ROOT))
        if relative in scanned:
            continue
        scanned.add(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix in (".data", ".restart"):
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            words = shlex.split(line, comments=True)
            if words and words[0] in ("include", "read_data", "read_restart") and len(words) > 1:
                if "$" in words[1]:
                    raise ValueError("Use literal relative application asset paths so their contents can be hashed and archived")
                asset = (ROOT / words[1]).resolve()
                if ROOT.resolve() not in asset.parents or not asset.is_file():
                    raise ValueError(f"Copy the application asset inside the bundle: {words[1]}")
                hashes[str(asset.relative_to(ROOT))] = hashlib.sha256(asset.read_bytes()).hexdigest()
                if words[0] == "include":
                    # Let the recursive pass scan the included file's own assets.
                    pending.append(str(asset.relative_to(ROOT)))
    return hashes


def application_specs(cfg):
    out = []
    for app in cfg["applications"]:
        required = {"id", "phase", "model", "temperature", "mu_B_minus_A", "natoms", "setup_file", "dynamics_file", "md_equil_steps", "md_steps_between_mc", "mc_attempts_per_call", "production_mc_calls", "initial_compositions", "seeds", "units", "pressure"}
        missing = required - app.keys()
        if missing:
            raise ValueError(f"Application {app.get('id')}: missing {sorted(missing)}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", app["id"]):
            raise ValueError("Application ids must be safe filename labels")
        if app["units"] != "lj" or app["phase"] not in ("solid", "liquid") or app["natoms"] != 6912:
            raise ValueError("This adapter is for the paper's binary LJ phase runs (6912 atoms); adapt explicitly for other applications")
        if len(set(app["initial_compositions"])) < 2 or max(app["initial_compositions"]) - min(app["initial_compositions"]) < 0.3:
            raise ValueError("Application needs at least two initial compositions separated by >=0.3")
        if len(set(app["seeds"])) < 3:
            raise ValueError("Application needs >=3 independent seeds")
        if any(not 0 <= c <= 1 for c in app["initial_compositions"]):
            raise ValueError("Invalid composition")
        for key in ("temperature", "md_steps_between_mc", "mc_attempts_per_call", "production_mc_calls"):
            if not isinstance(app[key], (int, float)) or app[key] <= 0:
                raise ValueError(f"Application {key} must be positive")
        for key in ("md_steps_between_mc", "mc_attempts_per_call", "production_mc_calls", "thermo_every_md_steps", "structure_dump_every_mc_calls"):
            if key in app and (not isinstance(app[key], int) or app[key] <= 0):
                raise ValueError(f"Application {key} must be a positive integer")
        if not isinstance(app["pressure"], (int, float)) or not isinstance(app["mu_B_minus_A"], (int, float)):
            raise ValueError("Fill the application's actual pressure and chemical-potential difference")
        if not isinstance(app["md_equil_steps"], int) or app["md_equil_steps"] < 0:
            raise ValueError("Fill the actual non-negative MD equilibration length")
        p = app.get("ranks", 1)
        if not isinstance(p, int) or not 1 <= p <= cfg["max_ranks"] or p in (2, 3):
            raise ValueError("For the application use rank 1, or >=4 ranks with verified permanent serial fallback; 2/3-rank variable-box parallel proposals are not enabled by this adapter")
        for key in ("setup_file", "dynamics_file"):
            path = (ROOT / app[key]).resolve()
            if ROOT.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"Application file missing or outside bundle: {app[key]}")
            text = path.read_text(encoding="utf-8")
            if "AUTHOR_FILL" in text or re.search(r"^\s*(clear|processors|reset_timestep|run|fix\s+\S+\s+\S+\s+(fast/swap|atom/swap))\b", text, re.M):
                raise ValueError(f"Fill real application setup/dynamics and remove run/MC commands: {app[key]}")
        dependencies = application_dependencies([app["setup_file"], app["dynamics_file"]])
        for c in app["initial_compositions"]:
            for seed in app["seeds"]:
                # Serial NPT avoids claiming safety for the source's one-time MPI geometry check.
                for kind in ("atom", "fast"):
                    cid = f"{round(c * 1000000):06d}"
                    sid = f"application__{app['id']}__c{cid}__s{seed}__{kind}"
                    out.append(dict(app, id=sid, state_id=app["id"], stage="applications", case=app["model"],
                                    kind=kind, ranks=p, seed=seed, initial_composition=c,
                                    requested_sweeps=app["production_mc_calls"] * app["mc_attempts_per_call"] / app["natoms"], ntypes=2, expect_parallel=False,
                                    expected_fallback=p > 1 and kind == "fast",
                                    application_scheduling="serial rank 1" if p == 1 else "MPI MD with required permanent serial MC fallback; not a parallel MC benchmark",
                                    application_dependency_sha256=dependencies,
                                    setup_sha256=hashlib.sha256((ROOT / app["setup_file"]).read_bytes()).hexdigest(),
                                    dynamics_sha256=hashlib.sha256((ROOT / app["dynamics_file"]).read_bytes()).hexdigest()))
    return out


def render_application(spec):
    # Included author files supply actual pair coefficients, initial phase, timestep,
    # and dynamics; no phase model is guessed by this workflow.
    rows = ["clear", f"processors 1 1 {spec['ranks']}", f"variable revision_T equal {spec['temperature']:.17g}",
            f"variable revision_P equal {spec['pressure']:.17g}", f"variable revision_seed equal {spec['seed']}",
            f"include {spec['setup_file']}", "set group all type 1"]
    c = spec["initial_composition"]
    if c == 1:
        rows += ["set group all type 2"]
    elif c > 0:
        rows += [f"set group all type/fraction 2 {c:.17g} {spec['seed']}"]
    rows += [f"velocity all create {spec['temperature']:.17g} {spec['seed']} mom yes rot no dist gaussian",
             f"include {spec['dynamics_file']}", "thermo_style custom step atoms pe temp press vol",
             "thermo_modify norm no format float %.17g", "thermo 1000", f"run {spec['md_equil_steps']}",
             "reset_timestep 0", "compute ctype all count/type atom", "compute centro all centro/atom 12",
             "compute centro_mean all reduce ave c_centro", "variable epa equal pe/atoms"]
    every = spec["md_steps_between_mc"]
    if spec["kind"] == "fast":
        rows += [f"fix mc all fast/swap {every} {spec['mc_attempts_per_call']} {spec['temperature']:.17g} {spec['seed']} types 2 1 2 mu 0 {spec['mu_B_minus_A']:.17g}"]
    else:
        rows += [f"fix mc all atom/swap {every} {spec['mc_attempts_per_call']} {spec['seed']} {spec['temperature']:.17g} ke no semi-grand yes types 1 2 mu 0 {spec['mu_B_minus_A']:.17g}"]
    rows += ["thermo_style custom step atoms pe v_epa c_ctype[1] c_ctype[2] f_mc[1] f_mc[2] temp press vol c_centro_mean",
             "thermo_modify norm no format float %.17g", f"thermo {spec.get('thermo_every_md_steps', min(1000, every))}",
             f"dump structure all custom {every * spec.get('structure_dump_every_mc_calls', 10)} results/raw/{spec['id']}.structure.dump id type x y z c_centro",
             "dump_modify structure sort id format float %.17g", "print \"REVISION_PRODUCTION_BEGIN\"",
             f"run {every * spec['production_mc_calls']}", "print \"REVISION_PRODUCTION_END\"",
             f"write_data results/raw/{spec['id']}.data nocoeff"]
    return "\n".join(rows) + "\n"


def build_manifest(cfg, suite="core"):
    validate_config(cfg)
    maxp = cfg["max_ranks"]
    selected = set(["exact", "parallel", "hybrid", "closure", "options", "guards", "timing", "applications"] if suite in ("all", "core") else [suite])
    specs = []
    def add(stage, name, box, p, seed, sweeps, **kwargs):
        c = CASES[name]
        kind = kwargs.pop("kind", "fast")
        sid = kwargs.pop("id", f"{stage}__{name}__p{p}__s{seed}__{kind}")
        geom = geometry(box, p, require_parallel=kwargs.get("expect_parallel", p > 1))
        expect_parallel = kwargs.pop("expect_parallel", p > 1 and kind == "fast")
        specs.append(dict(id=sid, stage=stage, case=name, kind=kind, box=list(box), natoms=4 * math.prod(box),
                          ntypes=c.ntypes, seed=seed, ranks=p, requested_sweeps=sweeps,
                          expect_parallel=expect_parallel, energy_normalization="norm no; PotEng=U; v_epa=U/N",
                          **geom, **kwargs))
    if "exact" in selected:
        for name in cfg["models"]:
            for seed in cfg["seeds"]:
                for kind in ("atom", "fast"):
                    add("exact", name, cfg["exact"]["box"], 1, seed, cfg["exact"]["sweeps"], kind=kind, type_dump_every=cfg["exact"]["type_dump_every"])
    if "parallel" in selected and maxp > 1:
        for name in cfg["models"]:
            ps = sorted({maxp} | {p for p in cfg["parallel"]["extra_ranks"] if p <= maxp and name in cfg["parallel"]["extra_models"]})
            for p in ps:
                for seed in cfg["seeds"]:
                    for kind in ("atom", "fast"):
                        add("parallel", name, cfg["parallel"]["box"], p, seed, cfg["parallel"]["sweeps"], kind=kind, type_dump_every=cfg["parallel"]["type_dump_every"])
    if "hybrid" in selected:
        h = cfg["hybrid"]
        for name in h["models"]:
            for p in h["ranks"]:
                if p > maxp:
                    continue
                for seed in cfg["seeds"]:
                    for kind in ("atom", "fast"):
                        add("hybrid", name, h["box"], p, seed, h["sweeps"], kind=kind, temperature=h["temperature"],
                            timestep=h["timestep"], mc_every=h["md_steps_between_mc"], type_dump_every=25)
    if "closure" in selected:
        for name in cfg["models"]:
            add("closure", name, cfg["closure"]["box"], 1, cfg["seeds"][0], 0, kind="diagnostic", trials=cfg["closure"]["trials"])
    if "options" in selected:
        for name in ("lj_cut", "morse", "unequal_lj_cut"):
            for option, modifiers, newton in (("shift_yes", ["pair_modify shift yes tail no"], None), ("newton_off", [], "off")):
                for kind in ("atom", "fast"):
                    add("options", name, cfg["exact"]["box"], 1, cfg["seeds"][0], 160, kind=kind,
                        id=f"options__{name}__{option}__{kind}", pair_modifiers=modifiers, newton=newton, type_dump_every=25)
    if "guards" in selected:
        for guard, name, expected in (("tail", "lj_cut", "tail corrections"), ("charge", "buck_coul_cut", "identical charge"),
                                      ("kspace", "buck_coul_cut", "KSpace long-range"), ("molecular", "lj_cut", "supports only atomic systems"),
                                      ("manybody", "lj_cut", "does not support many-body")):
            add("guards", name, [6, 6, 6], 1, cfg["seeds"][0], 1, id=f"guard__{guard}", guard=guard,
                expected_error=expected, pair_modifiers=["pair_modify tail yes"] if guard == "tail" else [])
        if maxp >= 2:
            add("guards", "lj_cut", [4, 4, 4], 2, cfg["seeds"][0], 3,
                id="guard__fallback_positive_control", expected_fallback=True, expect_parallel=False)
    if "timing" in selected or "prepare" in selected:
        t = cfg["timing"]
        groups = []
        for p in t["strong_ranks"]:
            if p <= maxp:
                groups.append(("strong", "lj_cut", t["strong_box"], p))
        for n in t["serial_cells"]:
            groups.append(("size", "lj_cut", [n, n, n], 1))
        if maxp > 1:
            for n in t["parallel_transverse_cells"]:
                groups.append(("size", "lj_cut", [n, n, 96], maxp))
        for name in cfg["models"]:
            groups.append(("pair_style", name, t["pair_style_box_serial"], 1))
            if maxp > 1:
                groups.append(("pair_style", name, t["pair_style_box_parallel"], maxp))
        for name in t["nonlj_scaling_models"]:
            for p in t["nonlj_scaling_ranks"]:
                if 1 < p <= maxp:
                    groups.append(("nonlj_strong", name, t["pair_style_box_parallel"], p))
        # Merge identical tasks across figures; group_membership retains their roles.
        unique = {}
        for label, name, box, p in groups:
            unique.setdefault((name, tuple(box), p), []).append(label)
        preparations = {}
        for name, box, _ in unique:
            key = (name, box)
            if key not in preparations:
                pid = f"prepare__{name}__{'x'.join(map(str, box))}"
                preparations[key] = pid
                add("prepare", name, box, 1, cfg["seeds"][0], t["preparation_sweeps"], id=pid, kind="fast")
        for (name, box, p), labels in unique.items():
            if "timing" not in selected:
                continue
            for rep in range(t["repeats"]):
                seed = cfg["seeds"][rep % len(cfg["seeds"])] + 10000 * (rep // len(cfg["seeds"]))
                for kind in ("atom", "fast"):
                    add("timing", name, box, p, seed, t["pilot_sweeps"], kind=kind,
                        id=f"timing__{name}__{'x'.join(map(str, box))}__p{p}__r{rep + 1}__{kind}",
                        repeat=rep + 1, group_membership=labels, warmup_sweeps=t["warmup_sweeps"],
                        preparation_id=preparations[(name, box)], initial_data_path=f"results/raw/{preparations[(name, box)]}.data",
                        minimum_loop_seconds=t["minimum_loop_seconds"])
    if "applications" in selected:
        specs.extend(application_specs(cfg))
        if suite in ("all", "applications") and not cfg["applications"]:
            raise ValueError("The actual phase-diagram state points and phase setup files are not configured. Fill applications in config.json; use --suite core for the complete algorithm/benchmark suite.")
    ids = [s["id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task ids")
    return specs


def write_inputs(specs):
    for d in ("inputs", "logs", "results/raw", "results/metadata", "results/analysis"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for spec in specs:
        inp = render_closure(spec) if spec["stage"] == "closure" else render_application(spec) if spec["stage"] == "applications" else render_mc(spec)
        path = ROOT / "inputs" / ("in." + spec["id"])
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(inp)
        spec["input"] = str(path.relative_to(ROOT))
        spec["input_sha256"] = hashlib.sha256(inp.encode()).hexdigest()
    (ROOT / "results/manifest.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    (ROOT / "results/metadata/model_parameters.json").write_text(
        json.dumps({name: dataclasses.asdict(case) for name, case in CASES.items()}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--suite", default="core")
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    specs = build_manifest(cfg, args.suite)
    write_inputs(specs)
    print(f"Generated {len(specs)} sequential tasks; real application states: {len(cfg['applications'])}")


if __name__ == "__main__":
    main()
