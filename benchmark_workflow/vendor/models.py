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
