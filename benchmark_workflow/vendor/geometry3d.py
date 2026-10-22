"""Focused 3D LJ supplement using the existing fix syntax and MC protocol."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import models

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.name: c for c in models.CASES if c.name == "lj_cut"}


def geometry(box, ranks, processors=None):
    grid = list(processors or ([1, 1, 1] if ranks == 1 else [2, 3, 4]))
    if len(box) != 3 or len(grid) != 3 or any(type(n) is not int or n < 1 for n in [*box, *grid]):
        raise ValueError("Box and processor grid need three positive integers")
    if math.prod(grid) != ranks or any(n % p for n, p in zip(box, grid)):
        raise ValueError("Processor grid must match rank count and divide the FCC cell counts")
    a = (4 / models.LATTICE_RHO) ** (1 / 3)
    lengths = [a * n for n in box]
    half = [length / p / 2 for length, p in zip(lengths, grid)]
    active = [h for h, p in zip(half, grid) if p > 1]
    minimum = min(active) if active else None
    if min(lengths) <= 5.4 or (minimum is not None and minimum <= 2.7):
        raise ValueError("Insufficient periodic-image or parallel independence margin")
    local = [n // p for n, p in zip(box, grid)]
    return {
        "box_lengths": lengths, "processors": grid,
        "local_fcc_cells": local, "local_aspect_ratio": max(local) / min(local),
        "max_pair_cutoff": 2.4, "neighbor_skin": 0.3,
        "half_domain_lengths": half, "half_domain_z": half[2],
        "minimum_active_half_domain": minimum,
        "geometric_margin": minimum - 2.7 if minimum is not None else None,
    }


def validate_config(cfg):
    if cfg["schema_version"] != 1 or cfg["max_ranks"] != 24 or cfg["processors"] != [2, 3, 4]:
        raise ValueError("This focused supplement requires 24 ranks and a 2x3x4 grid")
    if cfg["models"] != ["lj_cut"] or cfg["applications"]:
        raise ValueError("Only the two agreed LJ supplement tests are included")
    seeds = cfg["seeds"]
    if len(set(seeds)) < 3 or any(type(s) is not int or not 0 < s < 899900000 for s in seeds):
        raise ValueError("At least three distinct valid seeds are required")
    if cfg["lattice_origin"] != [0.25, 0.25, 0.25]:
        raise ValueError("Keep the quarter-cell origin used to avoid subdomain-plane sites")
    t = cfg["timing"]
    if t["repeats"] < 5 or t["minimum_loop_seconds"] < 30 or t["preparation_sweeps"] < 160:
        raise ValueError("Use >=5 repeats, >=30-second loops, and >=160 preparation sweeps")
    if t["warmup_sweeps"] < 1 or t["pilot_sweeps"] < 1 or t["maximum_sweeps"] < 1:
        raise ValueError("Invalid timing lengths")
    if cfg["parallel"]["sweeps"] < 600 or cfg["parallel"]["type_dump_every"] < 1:
        raise ValueError("Keep >=600 sweeps with recorded type snapshots")
    boxes = cfg["size_boxes"]
    if not boxes or len({tuple(b) for b in boxes}) != len(boxes):
        raise ValueError("Size boxes must be nonempty and distinct")
    for box in [*boxes, cfg["parallel"]["box"]]:
        g = geometry(box, 24, cfg["processors"])
        if g["local_aspect_ratio"] > 1.25:
            raise ValueError("The selected local domains are not sufficiently balanced")
    if any(not 0 <= f < 1 for f in cfg["analysis"]["discard_fractions"]):
        raise ValueError("Invalid discard fraction")


def build_manifest(cfg, suite="all"):
    validate_config(cfg)
    if suite not in ("core", "all", "timing", "parallel", "prepare"):
        raise ValueError("Use --suite all, timing, parallel, or prepare")
    specs = []

    def add(stage, box, ranks, seed, sweeps, kind, identifier, **extra):
        specs.append(dict(
            id=identifier, stage=stage, case="lj_cut", kind=kind, box=list(box),
            natoms=4 * math.prod(box), ntypes=2, seed=seed, ranks=ranks,
            requested_sweeps=sweeps, expect_parallel=ranks > 1 and kind == "fast",
            energy_normalization="norm no; PotEng=U; v_epa=U/N",
            lattice_origin=cfg["lattice_origin"],
            **geometry(box, ranks, cfg["processors"] if ranks > 1 else [1, 1, 1]), **extra))

    if suite in ("core", "all", "parallel"):
        p = cfg["parallel"]
        for seed in cfg["seeds"]:
            for kind in ("atom", "fast"):
                add("parallel", p["box"], 24, seed, p["sweeps"], kind,
                    f"parallel3d__lj_cut__p24__s{seed}__{kind}", type_dump_every=p["type_dump_every"])
    if suite in ("core", "all", "timing", "prepare"):
        t = cfg["timing"]
        for box in cfg["size_boxes"]:
            label = "x".join(map(str, box))
            pid = f"prepare3d__lj_cut__{label}"
            add("prepare", box, 1, cfg["seeds"][0], t["preparation_sweeps"], "fast", pid)
            if suite == "prepare":
                continue
            for repeat in range(t["repeats"]):
                seed = cfg["seeds"][repeat % len(cfg["seeds"])] + 10000 * (repeat // len(cfg["seeds"]))
                for kind in ("atom", "fast"):
                    add("timing", box, 24, seed, t["pilot_sweeps"], kind,
                        f"timing3d__lj_cut__{label}__p24__r{repeat + 1}__{kind}",
                        repeat=repeat + 1, group_membership=["size_3d"],
                        warmup_sweeps=t["warmup_sweeps"], preparation_id=pid,
                        initial_data_path=f"results/raw/{pid}.data",
                        minimum_loop_seconds=t["minimum_loop_seconds"])
    return specs


def render_mc(spec, sweeps=None):
    case = CASES[spec["case"]]
    sweeps = spec["requested_sweeps"] if sweeps is None else sweeps
    box, ranks = spec["box"], spec["ranks"]
    grid = spec.get("processors", [1, 1, 1] if ranks == 1 else [2, 3, 4])
    geometry(box, ranks, grid)
    origin = spec.get("lattice_origin", [0.25, 0.25, 0.25])
    rows = ["clear", "units lj", "dimension 3", "boundary p p p", "atom_style atomic",
            "processors " + " ".join(map(str, grid)), "comm_style brick", "newton on"]
    if spec.get("initial_data_path"):
        rows += [f"read_data {spec['initial_data_path']}"]
    else:
        rows += [f"lattice fcc {models.LATTICE_RHO} origin " + " ".join(map(str, origin)),
                 f"region box block 0 {box[0]} 0 {box[1]} 0 {box[2]} units lattice",
                 "create_box 2 box", "create_atoms 1 box", *models.init_regions(case, *box)]
    rows += ["mass 1 1.0", "mass 2 1.0", f"pair_style {case.pair_style}", *case.pair_coeffs,
             "pair_modify shift no tail no", "neighbor 0.3 bin", "neigh_modify delay 0 every 1 check yes",
             "timestep 0.0", "fix integrate all nve", f"variable ncycle equal {spec['natoms']}",
             f"variable seed equal {spec['seed']}", "variable tmc equal 1.0",
             "compute ctype all count/type atom", models.fix_line(case, spec["kind"]),
             "variable epa equal pe/atoms",
             "thermo_style custom step atoms pe v_epa c_ctype[1] c_ctype[2] f_mc[1] f_mc[2] temp press vol",
             "thermo_modify norm no format float %.17g"]
    if spec["stage"] == "timing":
        rows += [f"thermo {spec['warmup_sweeps']}", f"run {spec['warmup_sweeps']}",
                 'print "REVISION_PRODUCTION_BEGIN"', f"thermo {sweeps}", f"run {sweeps}",
                 'print "REVISION_PRODUCTION_END"']
    else:
        if spec.get("type_dump_every"):
            rows += [f"dump labels all custom {spec['type_dump_every']} results/raw/{spec['id']}.types.dump id type",
                     "dump_modify labels sort id"]
        rows += ["thermo 1", 'print "REVISION_PRODUCTION_BEGIN"', f"run {sweeps}",
                 'print "REVISION_PRODUCTION_END"',
                 f"write_dump all custom results/raw/{spec['id']}.final.dump id type x y z modify sort id format float %.17g"]
        if spec["stage"] == "prepare":
            rows += [f"write_data results/raw/{spec['id']}.data nocoeff"]
    return "\n".join(rows) + "\n"


def write_inputs(specs):
    for name in ("inputs", "logs", "results/raw", "results/metadata", "results/analysis"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for spec in specs:
        text = render_mc(spec)
        path = ROOT / "inputs" / ("in." + spec["id"])
        path.write_bytes(text.encode("ascii"))
        spec.update(input=path.relative_to(ROOT).as_posix(), input_sha256=hashlib.sha256(text.encode()).hexdigest())
    (ROOT / "results/manifest.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    (ROOT / "results/metadata/model_parameters.json").write_text(
        json.dumps({k: dataclasses.asdict(v) for k, v in CASES.items()}, indent=2), encoding="utf-8")
