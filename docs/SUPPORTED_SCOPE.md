# Supported scope and boundary cases

`fix fast/swap` evaluates the trial energy difference for a type-swap move from
the local pair environment of the selected atom. The method is exact when the
changed potential-energy terms are only the pair terms involving that atom.

## Supported energy form

The supported total potential energy has the form:

```text
U = sum_{i<j} phi_{type_i,type_j}(r_ij)
```

The pair contribution may be analytic, tabulated, shifted, screened, or
cutoff-dependent, provided that:

- the active LAMMPS pair style supplies a compatible `pair->single()` energy;
- all pair partners that may contribute before or after the type change are in
  the full neighbor list;
- there are no additional nonlocal or many-body energy terms contributing to
  the trial energy difference.

## Validated representative cases

- `lj/cut`
- `pair_style table` tabulated LJ
- `morse`
- `buck/coul/cut`
- `coul/dsf`
- `hybrid/overlay` with LJ and Yukawa pair terms
- `zbl`
- unequal LJ/cut type-pair cutoffs
- binary and ternary type sets

## Boundary cases handled by checks

The implementation rejects:

- many-body pair styles;
- KSpace long-range energy terms;
- pair tail corrections;
- charged type swaps where swappable atom types do not have identical charges;
- models whose energy changes cannot be represented by the local pair sum.

## Unequal cutoffs

Unequal type-pair cutoffs are supported by requesting a neighbor-list cutoff
large enough to include all pair partners that may contribute before or after a
type change. The pair style still evaluates the actual pair energy using the
correct type-pair cutoff. Accepted type changes under unequal cutoffs trigger a
conservative communication/neighbor consistency path, so performance can be
lower than equal-cutoff cases.

## Parallel execution

Under domain decomposition, independent local type-swap moves may be proposed
concurrently when their affected pair environments do not overlap. If the
geometric independence condition is not satisfied, the implementation falls
back to the serial selection path to preserve correctness.

