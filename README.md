# fix fast/swap

`fix fast/swap` is a LAMMPS Monte Carlo fix for semi-grand-canonical
type-swap sampling in atomic systems with pairwise-additive interaction models.
It preserves the usual SGC-MC Metropolis transition rule while replacing
repeated global potential-energy reevaluation with an exact localized
pair-energy difference for a trial atom-type change.

This repository is prepared as the code companion for the CPC manuscript:

**Fast Semi-Grand-Canonical Type-Swap Monte Carlo in LAMMPS via Localized
Energy-Difference Evaluation for Pairwise-Additive Interaction Models**

## Repository layout

```text
src/MC/
  fix_fast_swap.cpp
  fix_fast_swap.h

examples/
  in.exact__lj_cut__fast
  in.exact__lj_cut__atom
  in.exact__ternary_lj_cut__fast
  in.exact__morse__fast
  in.exact__table_lj__fast
  in.exact__unequal_lj_cut__fast
  in.parallel24_pair__lj_cut__fast
  tables/pair_lj.table

benchmark_workflow/
  run_all.sh
  submit_slurm.sh
  diagnose_lmp_binary.sh
  scripts/
  tables/

docs/
  INSTALL.md
  SUPPORTED_SCOPE.md
```

## Basic LAMMPS syntax

```lammps
fix ID group-ID fast/swap Nevery Ncycle T seed types N type1 type2 ... typeN mu mu1 mu2 ... muN
```

Optional single-rank energy-closure diagnostics:

```lammps
fix ID group-ID fast/swap Nevery Ncycle T seed types N ... mu ... check yes Period Tolerance
```

Example:

```lammps
fix mc all fast/swap 1 ${ncycle} ${tmc} ${seed} types 2 1 2 mu 0.0 0.2
```

Here `Ncycle` is the number of attempted type-swap moves made whenever the fix
is invoked. An MC sweep in the manuscript corresponds to `Ncycle = N_atoms`.

## Quick start

Copy the source files into the LAMMPS `src/MC` package directory:

```bash
cp src/MC/fix_fast_swap.cpp /path/to/lammps/src/MC/
cp src/MC/fix_fast_swap.h   /path/to/lammps/src/MC/
```

Build LAMMPS with the MC package and any pair-style packages required by your
inputs. A typical CMake build is:

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=yes \
  -DPKG_MC=yes
cmake --build . -j 4
```

Then run a minimal example:

```bash
cd examples
mkdir -p results
mpirun -np 1 /path/to/lmp_mpi -in in.exact__lj_cut__fast
```

More detailed installation notes are in `docs/INSTALL.md`.

## Validation and benchmark workflow

The `benchmark_workflow/` directory contains the workflow used for the paper
validation and performance benchmarks. From that directory:

```bash
bash run_all.sh /path/to/lmp_mpi
```

Useful subsets:

```bash
SUITE=validation bash run_all.sh /path/to/lmp_mpi
SUITE=size_scaling bash run_all.sh /path/to/lmp_mpi
SUITE=strong_scaling bash run_all.sh /path/to/lmp_mpi
SUITE=nonlj_parallel_scaling bash run_all.sh /path/to/lmp_mpi
```

The full raw data and publication figure scripts are archived in the companion
Zenodo dataset.

## Supported scope

The current implementation supports atomic systems whose potential energy can be
written as a sum of pair terms that are accessible through the LAMMPS
`pair->single()` interface. See `docs/SUPPORTED_SCOPE.md`.

Validated examples include:

- LJ/cut
- tabulated LJ
- Morse
- Buckingham+Coul/cut
- Coul/DSF
- LJ+Yukawa hybrid/overlay
- ZBL
- unequal type-pair cutoffs
- binary and ternary type-swap cases

Unsupported cases include many-body pair styles, KSpace long-range energy
terms, pair tail corrections, molecular topology terms, and charged type swaps
with incompatible per-atom charges.

## License

This code is intended to be distributed under GPL-2.0-or-later, consistent with
LAMMPS-derived source files. See `LICENSE`.

