from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

DEFAULT_SEED = 24681357
LATTICE_RHO = 0.8442


@dataclass(frozen=True)
class PairCase:
    name: str
    description: str
    ntypes: int
    swap_types: tuple[int, ...]
    atom_style: str
    pair_style: str
    pair_coeffs: tuple[str, ...]
    mu: tuple[float, ...]
    masses: tuple[float, ...]
    charges: tuple[float, ...] | None = None


def lines(*items: str) -> str:
    return "\n".join(item.rstrip() for item in items if item is not None) + "\n"


def pair_coeff_lj(ntypes: int, cutoff: float = 2.4, unequal: bool = False) -> tuple[str, ...]:
    rows: list[str] = []
    for i in range(1, ntypes + 1):
        for j in range(i, ntypes + 1):
            eps = 0.78 + 0.07 * i + 0.04 * j
            sig = 0.90 + 0.025 * i + 0.015 * j
            cut = cutoff
            if unequal:
                cut = 1.65 + 0.17 * i + 0.11 * j
            rows.append(f"pair_coeff {i} {j} {eps:.6f} {sig:.6f} {cut:.6f}")
    return tuple(rows)


CASES: tuple[PairCase, ...] = (
    PairCase(
        name="lj_cut",
        description="binary LJ/cut, equal cutoff",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="lj/cut 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 1.00 2.4",
            "pair_coeff 1 2 1.10 0.98 2.4",
            "pair_coeff 2 2 0.85 0.95 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="unequal_lj_cut",
        description="binary LJ/cut, unequal type-pair cutoffs",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="lj/cut 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 1.00 1.8",
            "pair_coeff 1 2 1.12 0.98 2.1",
            "pair_coeff 2 2 0.85 0.95 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="morse",
        description="binary Morse pair style",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="morse 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 2.0 1.00 2.4",
            "pair_coeff 1 2 1.10 2.0 0.98 2.4",
            "pair_coeff 2 2 0.85 2.0 0.95 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="buck_coul_cut",
        description="binary buck/coul/cut with identical swappable charges",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="charge",
        pair_style="buck/coul/cut 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 1.00 0.10 2.4",
            "pair_coeff 1 2 1.10 1.00 0.10 2.4",
            "pair_coeff 2 2 0.85 1.00 0.10 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
        charges=(0.30, 0.30),
    ),
    PairCase(
        name="coul_dsf",
        description="binary coul/dsf with identical swappable charges",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="charge",
        pair_style="coul/dsf 0.05 2.4",
        pair_coeffs=("pair_coeff * *",),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
        charges=(0.30, 0.30),
    ),
    PairCase(
        name="table_lj",
        description="binary tabulated LJ pair style",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="table linear 400",
        pair_coeffs=(
            "pair_coeff 1 1 tables/pair_lj.table LJAA",
            "pair_coeff 1 2 tables/pair_lj.table LJAB",
            "pair_coeff 2 2 tables/pair_lj.table LJBB",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="hybrid_overlay_lj_yukawa",
        description="binary hybrid/overlay LJ plus Yukawa",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="hybrid/overlay lj/cut 2.4 yukawa 1.2 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 lj/cut 1.00 1.00 2.4",
            "pair_coeff 1 2 lj/cut 1.10 0.98 2.4",
            "pair_coeff 2 2 lj/cut 0.85 0.95 2.4",
            "pair_coeff 1 1 yukawa 1.00 2.4",
            "pair_coeff 1 2 yukawa 1.10 2.4",
            "pair_coeff 2 2 yukawa 0.85 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="zbl",
        description="binary ZBL pair style",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="atomic",
        pair_style="zbl 1.8 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 10.0 10.0",
            "pair_coeff 1 2 10.0 12.0",
            "pair_coeff 2 2 12.0 12.0",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
    ),
    PairCase(
        name="ternary_lj_cut",
        description="ternary LJ/cut with three swappable types",
        ntypes=3,
        swap_types=(1, 2, 3),
        atom_style="atomic",
        pair_style="lj/cut 2.4",
        pair_coeffs=pair_coeff_lj(3, 2.4, unequal=False),
        mu=(0.0, 0.18, -0.12),
        masses=(1.0, 1.0, 1.0),
    ),
)


PARALLEL_STAT_CASE_NAMES = frozenset(
    {
        "lj_cut",
        "unequal_lj_cut",
        "ternary_lj_cut",
        "morse",
        "buck_coul_cut",
        "coul_dsf",
        "hybrid_overlay_lj_yukawa",
        "zbl",
    }
)

NONLJ_RICH_CASE_NAMES = frozenset(
    {
        "morse",
        "buck_coul_cut",
        "coul_dsf",
        "hybrid_overlay_lj_yukawa",
        "zbl",
    }
)


def cases_by_name(names: frozenset[str]) -> list[PairCase]:
    return [case for case in CASES if case.name in names]


def write_lj_table() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    sections = [
        ("LJAA", 1.00, 1.00),
        ("LJAB", 1.10, 0.98),
        ("LJBB", 0.85, 0.95),
    ]
    n = 400
    rmin = 0.80
    rmax = 2.40
    out: list[str] = []
    for label, eps, sig in sections:
        out += [label, f"N {n} R {rmin:.8f} {rmax:.8f}", ""]
        for idx in range(1, n + 1):
            r = rmin + (rmax - rmin) * (idx - 1) / (n - 1)
            sr = sig / r
            sr6 = sr**6
            sr12 = sr6**2
            energy = 4.0 * eps * (sr12 - sr6)
            force = 24.0 * eps * (2.0 * sr12 - sr6) / r
            out.append(f"{idx:6d} {r:.12g} {energy:.16g} {force:.16g}")
        out.append("")
    (TABLES / "pair_lj.table").write_text("\n".join(out), encoding="ascii")


def lattice_atoms(nx: int, ny: int, nz: int) -> int:
    return 4 * nx * ny * nz


def init_regions(case: PairCase, nx: int, ny: int, nz: int) -> list[str]:
    if case.ntypes == 2:
        return [
            f"region upper block INF INF INF INF {0.5 * nz:g} {nz:g} units lattice",
            "set region upper type 2",
        ]
    if case.ntypes == 3:
        return [
            f"region slab2 block INF INF INF INF {nz / 3:g} {2 * nz / 3:g} units lattice",
            f"region slab3 block INF INF INF INF {2 * nz / 3:g} INF units lattice",
            "set region slab2 type 2",
            "set region slab3 type 3",
        ]
    raise ValueError(f"Unsupported ntypes in generator: {case.ntypes}")


def fix_line(
    case: PairCase,
    kind: str,
    *,
    self_check: bool = False,
    check_period: int = 1,
    check_tolerance: float = 1.0e-8,
) -> str:
    types = " ".join(str(t) for t in case.swap_types)
    mus = " ".join(f"{mu:.8g}" for mu in case.mu)
    if kind == "atom":
        return (
            "fix mc all atom/swap 1 ${ncycle} ${seed} ${tmc} ke no "
            f"semi-grand yes types {types} mu {mus}"
        )
    if kind == "fast":
        cmd = (
            "fix mc all fast/swap 1 ${ncycle} ${tmc} ${seed} "
            f"types {len(case.swap_types)} {types} mu {mus}"
        )
        if self_check:
            cmd += f" check yes {check_period} {check_tolerance:.8g}"
        return cmd
    raise ValueError(kind)


def make_input(
    *,
    spec_id: str,
    case: PairCase,
    kind: str,
    nx: int,
    ny: int,
    nz: int,
    nsweeps: int,
    ranks: int,
    ncycle: int | None = None,
    seed: int = DEFAULT_SEED,
    thermo_every: int = 50,
    self_check: bool = False,
    check_period: int = 1,
    check_tolerance: float = 1.0e-8,
    pair_modifiers: tuple[str, ...] = (),
    extra_after_pair: tuple[str, ...] = (),
) -> str:
    natoms = lattice_atoms(nx, ny, nz)
    if ncycle is None:
        ncycle = natoms
    masses = [f"mass {i} {case.masses[i - 1]:.8g}" for i in range(1, case.ntypes + 1)]
    charges: list[str] = []
    if case.charges is not None:
        charges = [f"set type {i} charge {case.charges[i - 1]:.8g}" for i in range(1, case.ntypes + 1)]
    count_cols = " ".join(f"c_ctype[{i}]" for i in range(1, case.ntypes + 1))
    processors = f"processors 1 1 {ranks}" if ranks > 1 else None
    return lines(
        "clear",
        "units lj",
        "dimension 3",
        "boundary p p p",
        f"atom_style {case.atom_style}",
        processors,
        "",
        f"variable ncycle equal {ncycle}",
        f"variable nsweeps equal {nsweeps}",
        f"variable seed equal {seed}",
        "variable tmc equal 1.0",
        "",
        f"lattice fcc {LATTICE_RHO}",
        f"region box block 0 {nx} 0 {ny} 0 {nz} units lattice",
        f"create_box {case.ntypes} box",
        "create_atoms 1 box",
        *init_regions(case, nx, ny, nz),
        *masses,
        *charges,
        "",
        f"pair_style {case.pair_style}",
        *case.pair_coeffs,
        *pair_modifiers,
        *extra_after_pair,
        "",
        "neighbor 0.3 bin",
        "neigh_modify delay 0 every 1 check yes",
        "timestep 0.0",
        "fix int all nve",
        "",
        "compute ctype all count/type atom",
        fix_line(
            case,
            kind,
            self_check=self_check,
            check_period=check_period,
            check_tolerance=check_tolerance,
        ),
        f"thermo {thermo_every}",
        "thermo_modify format float %.17g",
        f"thermo_style custom step pe {count_cols} f_mc[1] f_mc[2]",
        "run ${nsweeps}",
        "",
        f"write_data results/{spec_id}.data nocoeff",
    )


def add_pair_style_specs(manifest: list[dict], max_ranks: int, include_parallel: bool) -> None:
    # Small serial case: enough atoms for stable timing but cheap enough for many pair styles.
    for case in CASES:
        for kind in ("atom", "fast"):
            spec_id = f"serial_pair__{case.name}__{kind}"
            write_spec_input(manifest, spec_id, "serial_pair_styles", case, kind, 6, 6, 6, 20, 1)
    if include_parallel and max_ranks >= 2:
        # Long z dimension and processors 1 1 P ensure the active subdomain is larger
        # than cutoff + skin at P=24, avoiding the serial fallback path.
        for case in CASES:
            for kind in ("atom", "fast"):
                spec_id = f"parallel{max_ranks}_pair__{case.name}__{kind}"
                write_spec_input(manifest, spec_id, "parallel_pair_styles", case, kind, 4, 4, 96, 6, max_ranks)


def add_lj_scaling_specs(manifest: list[dict], max_ranks: int, include_parallel: bool) -> None:
    lj = next(case for case in CASES if case.name == "lj_cut")
    for ncell, sweeps in ((5, 30), (7, 15), (9, 6)):
        for kind in ("atom", "fast"):
            spec_id = f"serial_scaling__lj_cut_n{ncell}__{kind}"
            write_spec_input(manifest, spec_id, "serial_lj_scaling", lj, kind, ncell, ncell, ncell, sweeps, 1)
    if include_parallel and max_ranks >= 2:
        ranks = [r for r in (2, 4, 8, 16, 24) if r <= max_ranks]
        if max_ranks not in ranks:
            ranks.append(max_ranks)
        for ranks_i in sorted(set(ranks)):
            for kind in ("atom", "fast"):
                spec_id = f"parallel_scaling__lj_cut_p{ranks_i}__{kind}"
                write_spec_input(manifest, spec_id, "parallel_lj_scaling", lj, kind, 4, 4, 96, 5, ranks_i)


def add_strong_scaling_specs(manifest: list[dict], max_ranks: int) -> None:
    lj = next(case for case in CASES if case.name == "lj_cut")
    ranks = [r for r in (1, 2, 4, 8, 12, 16, 24) if r <= max_ranks]
    if max_ranks not in ranks:
        ranks.append(max_ranks)
    for ranks_i in sorted(set(ranks)):
        for kind in ("atom", "fast"):
            spec_id = f"strong_scaling__lj_cut_p{ranks_i}__{kind}"
            write_spec_input(
                manifest,
                spec_id,
                "strong_scaling",
                lj,
                kind,
                8,
                8,
                96,
                4,
                ranks_i,
                notes="large fixed-size LJ/cut strong-scaling test; enlarged to reduce low-rank fast/swap MPI-overhead artifacts",
            )


def add_unequal_cutoff_scaling_specs(manifest: list[dict], max_ranks: int) -> None:
    unequal = next(case for case in CASES if case.name == "unequal_lj_cut")
    # This suite isolates the conservative maximum-cutoff path.  The smallest
    # point reproduces the representative serial test; larger points show
    # whether the apparent low single-rank speedup is a small-system artifact.
    for ncell, sweeps in ((6, 20), (8, 8), (10, 4), (12, 2)):
        for kind in ("atom", "fast"):
            spec_id = f"unequal_serial__lj_cut_n{ncell}__{kind}"
            write_spec_input(
                manifest,
                spec_id,
                "unequal_cutoff_scaling",
                unequal,
                kind,
                ncell,
                ncell,
                ncell,
                sweeps,
                1,
                notes="serial scaling of the exact maximum-cutoff path for unequal type-pair cutoffs",
            )

    if max_ranks < 2:
        return
    pranks = max_ranks
    for nx, ny, nz, sweeps in ((4, 4, 96, 6), (6, 6, 96, 4), (8, 8, 96, 3)):
        for kind in ("atom", "fast"):
            spec_id = f"unequal_parallel{pranks}__lj_cut_{nx}x{ny}x{nz}__{kind}"
            write_spec_input(
                manifest,
                spec_id,
                "unequal_cutoff_scaling",
                unequal,
                kind,
                nx,
                ny,
                nz,
                sweeps,
                pranks,
                notes="24-rank scaling of the exact maximum-cutoff path for unequal type-pair cutoffs",
            )


def add_size_scaling_specs(manifest: list[dict], max_ranks: int) -> None:
    lj = next(case for case in CASES if case.name == "lj_cut")
    # Rank-1 cubic boxes test the expected size dependence without domain
    # decomposition.  The sweep counts are deliberately higher than in the
    # quick-look suite so that millisecond-level fast/swap timings are less
    # affected by launch and scheduler noise.
    for ncell, sweeps in ((5, 80), (6, 60), (7, 40), (8, 30), (9, 20), (10, 14), (11, 10), (12, 8)):
        for kind in ("atom", "fast"):
            spec_id = f"size_serial__lj_cut_n{ncell}__{kind}"
            write_spec_input(manifest, spec_id, "size_scaling", lj, kind, ncell, ncell, ncell, sweeps, 1)

    if max_ranks < 2:
        return
    pranks = max_ranks
    # Long z boxes with processors 1 1 P keep true parallel fast/swap active up
    # to 24 ranks while changing the total atom count through the transverse area.
    for nx, ny, nz, sweeps in (
        (3, 3, 96, 20),
        (4, 4, 96, 15),
        (5, 5, 96, 12),
        (6, 6, 96, 10),
        (7, 7, 96, 8),
        (8, 8, 96, 6),
        (9, 9, 96, 5),
        (10, 10, 96, 4),
    ):
        for kind in ("atom", "fast"):
            spec_id = f"size_parallel{pranks}__lj_cut_{nx}x{ny}x{nz}__{kind}"
            write_spec_input(manifest, spec_id, "size_scaling", lj, kind, nx, ny, nz, sweeps, pranks)


def add_nonlj_parallel_scaling_specs(manifest: list[dict], max_ranks: int) -> None:
    if max_ranks < 2:
        return
    names = ("morse", "table_lj", "buck_coul_cut", "coul_dsf", "hybrid_overlay_lj_yukawa", "zbl")
    cases = [case for case in CASES if case.name in names]
    ranks = [r for r in (2, 8, 24) if r <= max_ranks]
    if max_ranks >= 2 and max_ranks not in ranks:
        ranks.append(max_ranks)
    for case in cases:
        for ranks_i in sorted(set(ranks)):
            for kind in ("atom", "fast"):
                spec_id = f"nonlj_parallel_scaling__{case.name}_p{ranks_i}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "nonlj_parallel_scaling",
                    case,
                    kind,
                    4,
                    4,
                    96,
                    5,
                    ranks_i,
                    notes="fixed-size parallel scaling for representative non-LJ/cut pair styles",
                )


def add_timeseries_specs(
    manifest: list[dict],
    max_ranks: int,
    *,
    parallel_case_names: frozenset[str] = PARALLEL_STAT_CASE_NAMES,
    include_exact: bool = True,
) -> None:
    # Dense per-sweep trajectory output for later plotting.  These runs are not
    # used for timing because thermo output every sweep would bias performance.
    if include_exact:
        for case in CASES:
            for kind in ("atom", "fast"):
                spec_id = f"timeseries_exact__{case.name}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "timeseries_exact",
                    case,
                    kind,
                    6,
                    6,
                    6,
                    160,
                    1,
                    thermo_every=1,
                    self_check=(kind == "fast"),
                    check_period=25,
                    check_tolerance=1.0e-8,
                    analysis_role="timeseries_exact",
                    notes="dense single-rank per-sweep trajectory for energy/composition parity plots",
                )

    if max_ranks < 2:
        return
    pranks = max_ranks
    seeds = (24681357, 24681457, 24681557)
    cases = cases_by_name(parallel_case_names)
    for case in cases:
        for seed in seeds:
            for kind in ("atom", "fast"):
                spec_id = f"timeseries_parallel{pranks}__{case.name}__seed{seed}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "timeseries_parallel_stats",
                    case,
                    kind,
                    4,
                    4,
                    96,
                    160,
                    pranks,
                    seed=seed,
                    thermo_every=1,
                    analysis_role="timeseries_parallel_stats",
                    notes="dense 24-rank per-sweep trajectory for stationary statistics; trajectory identity is not expected",
                )


def add_timing_repeat_specs(manifest: list[dict], max_ranks: int) -> None:
    # Repeated timing cases quantify run-to-run noise without dense thermo I/O.
    lj = next(case for case in CASES if case.name == "lj_cut")
    seeds = (24681357, 24682357, 24683357)
    serial_cases = ((6, 6, 6, 50), (9, 9, 9, 24), (12, 12, 12, 10))
    for seed in seeds:
        for nx, ny, nz, sweeps in serial_cases:
            for kind in ("atom", "fast"):
                spec_id = f"timing_repeat__lj_cut_{nx}x{ny}x{nz}_seed{seed}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "timing_repeats",
                    lj,
                    kind,
                    nx,
                    ny,
                    nz,
                    sweeps,
                    1,
                    seed=seed,
                    thermo_every=sweeps,
                    analysis_role="timing_repeat",
                    notes="single-rank repeated timing point for estimating speedup uncertainty",
                )

    if max_ranks < 2:
        return
    pranks = max_ranks
    parallel_cases = ((4, 4, 96, 12), (8, 8, 96, 6), (10, 10, 96, 4))
    for seed in seeds:
        for nx, ny, nz, sweeps in parallel_cases:
            for kind in ("atom", "fast"):
                spec_id = f"timing_repeat{pranks}__lj_cut_{nx}x{ny}x{nz}_seed{seed}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "timing_repeats",
                    lj,
                    kind,
                    nx,
                    ny,
                    nz,
                    sweeps,
                    pranks,
                    seed=seed,
                    thermo_every=sweeps,
                    analysis_role="timing_repeat",
                    notes="24-rank repeated timing point for estimating speedup uncertainty",
                )


def add_exact_validation_specs(manifest: list[dict]) -> None:
    for case in CASES:
        for kind in ("atom", "fast"):
            spec_id = f"exact__{case.name}__{kind}"
            write_spec_input(
                manifest,
                spec_id,
                "validation_exact",
                case,
                kind,
                6,
                6,
                6,
                60,
                1,
                thermo_every=5,
                self_check=(kind == "fast"),
                check_period=100,
                check_tolerance=1.0e-8,
                analysis_role="exact_consistency",
                notes="single-rank trajectory-level equality; fast/swap self-check enabled",
            )


def add_parallel_stat_validation_specs(
    manifest: list[dict],
    max_ranks: int,
    *,
    case_names: frozenset[str] = PARALLEL_STAT_CASE_NAMES,
) -> None:
    if max_ranks < 2:
        return
    pranks = max_ranks
    seeds = (24681357, 24681457, 24681557)
    cases = cases_by_name(case_names)
    for case in cases:
        for seed in seeds:
            for kind in ("atom", "fast"):
                spec_id = f"pstats{pranks}__{case.name}__seed{seed}__{kind}"
                write_spec_input(
                    manifest,
                    spec_id,
                    "validation_parallel_stats",
                    case,
                    kind,
                    4,
                    4,
                    96,
                    80,
                    pranks,
                    seed=seed,
                    thermo_every=5,
                    analysis_role="statistical_consistency",
                    notes="parallel statistical consistency; trajectory identity is not expected",
                )


def add_nonlj_rich_validation_specs(manifest: list[dict], max_ranks: int) -> None:
    add_parallel_stat_validation_specs(manifest, max_ranks, case_names=NONLJ_RICH_CASE_NAMES)
    add_timeseries_specs(
        manifest,
        max_ranks,
        parallel_case_names=NONLJ_RICH_CASE_NAMES,
        include_exact=False,
    )


def add_boundary_guard_specs(manifest: list[dict]) -> None:
    lj = next(case for case in CASES if case.name == "lj_cut")
    kspace_case = PairCase(
        name="lj_cut_coul_long",
        description="binary LJ/coul/long with KSpace; expected fast/swap rejection",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="charge",
        pair_style="lj/cut/coul/long 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 1.00 2.4",
            "pair_coeff 1 2 1.10 0.98 2.4",
            "pair_coeff 2 2 0.85 0.95 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
        charges=(0.0, 0.0),
    )
    charge_case = PairCase(
        name="charge_mismatch_buck_coul_cut",
        description="charged atom types with non-identical charges; expected fast/swap rejection",
        ntypes=2,
        swap_types=(1, 2),
        atom_style="charge",
        pair_style="buck/coul/cut 2.4",
        pair_coeffs=(
            "pair_coeff 1 1 1.00 1.00 0.10 2.4",
            "pair_coeff 1 2 1.10 1.00 0.10 2.4",
            "pair_coeff 2 2 0.85 1.00 0.10 2.4",
        ),
        mu=(0.0, 0.20),
        masses=(1.0, 1.0),
        charges=(0.30, -0.30),
    )
    boundary_specs = (
        (
            "tail_lj_cut",
            lj,
            ("pair_modify tail yes",),
            (),
            "tail corrections",
            "pair_modify tail yes is intentionally outside the localized formulation",
        ),
        (
            "kspace_lj_cut_coul_long",
            kspace_case,
            (),
            ("kspace_style pppm 1.0e-4",),
            "KSpace",
            "long-range reciprocal-space energy is intentionally outside the localized formulation",
        ),
        (
            "charge_mismatch",
            charge_case,
            (),
            (),
            "identical charge",
            "fast/swap changes atom types only and should reject non-identical swappable charges",
        ),
    )
    for name, case, pair_modifiers, extra_after_pair, pattern, note in boundary_specs:
        for kind in ("atom", "fast"):
            spec_id = f"boundary__{name}__{kind}"
            write_spec_input(
                manifest,
                spec_id,
                "boundary_guards",
                case,
                kind,
                5,
                5,
                5,
                3,
                1,
                thermo_every=1,
                pair_modifiers=pair_modifiers,
                extra_after_pair=extra_after_pair,
                analysis_role="expected_rejection",
                expected="fast_reject",
                expected_error_pattern=pattern,
                notes=note,
            )


def write_spec_input(
    manifest: list[dict],
    spec_id: str,
    suite: str,
    case: PairCase,
    kind: str,
    nx: int,
    ny: int,
    nz: int,
    nsweeps: int,
    ranks: int,
    *,
    seed: int = DEFAULT_SEED,
    thermo_every: int = 50,
    self_check: bool = False,
    check_period: int = 1,
    check_tolerance: float = 1.0e-8,
    pair_modifiers: tuple[str, ...] = (),
    extra_after_pair: tuple[str, ...] = (),
    analysis_role: str = "performance",
    expected: str = "pass",
    expected_error_pattern: str | None = None,
    notes: str = "",
) -> None:
    text = make_input(
        spec_id=spec_id,
        case=case,
        kind=kind,
        nx=nx,
        ny=ny,
        nz=nz,
        nsweeps=nsweeps,
        ranks=ranks,
        seed=seed,
        thermo_every=thermo_every,
        self_check=self_check,
        check_period=check_period,
        check_tolerance=check_tolerance,
        pair_modifiers=pair_modifiers,
        extra_after_pair=extra_after_pair,
    )
    path = INPUTS / f"in.{spec_id}"
    path.write_text(text, encoding="ascii")
    lattice_spacing = (4.0 / LATTICE_RHO) ** (1.0 / 3.0)
    manifest.append(
        {
            "id": spec_id,
            "suite": suite,
            "case": case.name,
            "description": case.description,
            "kind": kind,
            "ranks": ranks,
            "ntypes": case.ntypes,
            "swap_types": list(case.swap_types),
            "natoms": lattice_atoms(nx, ny, nz),
            "box_lattice": [nx, ny, nz],
            "box_length_lj": [nx * lattice_spacing, ny * lattice_spacing, nz * lattice_spacing],
            "nsweeps": nsweeps,
            "ncycle": lattice_atoms(nx, ny, nz),
            "seed": seed,
            "thermo_every": thermo_every,
            "analysis_role": analysis_role,
            "expected": expected,
            "expected_error_pattern": expected_error_pattern,
            "notes": notes,
            "input": path.relative_to(ROOT).as_posix(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="all",
        choices=[
            "all",
            "representative",
            "scaling",
            "strong_scaling",
            "size_scaling",
            "unequal_cutoff_scaling",
            "nonlj_parallel_scaling",
            "validation",
            "validation_exact",
            "validation_parallel_stats",
            "boundary",
            "timeseries",
            "timing_repeats",
            "rich_validation",
            "nonlj_rich_validation",
            "paper_full",
        ],
    )
    parser.add_argument("--max-ranks", type=int, default=24)
    args = parser.parse_args()

    INPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_lj_table()

    manifest: list[dict] = []
    if args.suite in ("all", "representative", "paper_full"):
        add_pair_style_specs(manifest, args.max_ranks, include_parallel=True)
    if args.suite in ("all", "scaling", "paper_full"):
        add_lj_scaling_specs(manifest, args.max_ranks, include_parallel=True)
    if args.suite in ("all", "scaling", "strong_scaling", "paper_full"):
        add_strong_scaling_specs(manifest, args.max_ranks)
    if args.suite in ("all", "scaling", "size_scaling", "paper_full"):
        add_size_scaling_specs(manifest, args.max_ranks)
    if args.suite in ("all", "scaling", "unequal_cutoff_scaling", "paper_full"):
        add_unequal_cutoff_scaling_specs(manifest, args.max_ranks)
    if args.suite in ("all", "scaling", "nonlj_parallel_scaling", "paper_full"):
        add_nonlj_parallel_scaling_specs(manifest, args.max_ranks)
    if args.suite in ("all", "validation", "validation_exact", "paper_full"):
        add_exact_validation_specs(manifest)
    if args.suite in ("all", "validation", "validation_parallel_stats", "paper_full"):
        add_parallel_stat_validation_specs(manifest, args.max_ranks)
    if args.suite in ("all", "validation", "boundary", "paper_full"):
        add_boundary_guard_specs(manifest)
    if args.suite in ("all", "timeseries", "rich_validation", "paper_full"):
        add_timeseries_specs(manifest, args.max_ranks)
    if args.suite in ("all", "timing_repeats", "rich_validation", "paper_full"):
        add_timing_repeat_specs(manifest, args.max_ranks)
    if args.suite in ("nonlj_rich_validation",):
        add_nonlj_rich_validation_specs(manifest, args.max_ranks)

    manifest_path = RESULTS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} LAMMPS inputs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
