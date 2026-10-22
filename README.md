# fix fast/swap

Localized semi-grand-canonical type substitutions for LAMMPS, using the
existing pair-style `single()` interface and a full neighbor list.
This is the source/data companion to **Fast Semi-Grand-Canonical Type-Swap Monte Carlo in LAMMPS via Localized Energy-Difference Evaluation for Pairwise-Additive Interaction Models** (CPC manuscript under revision).

## Snapshot

- Prepared release candidate: `2026.10.21-r2`; not an assertion that a release has been published.
- Algorithm source: rank-local RNG initialization revision **20261020**.
- Tested LAMMPS snapshot: **17 Apr 2024 (Development)**; GNU C++ 15.3.0 and MPICH 5.0.2.
- Repository: https://github.com/cqx333/fix-fast-swap
- Archival DOI for this version: https://doi.org/10.5281/zenodo.23551995.
- Previous archival record: https://doi.org/10.5281/zenodo.20740858.

The new-version DOI was reserved in the author's Zenodo draft. The link becomes
publicly available when that record is published; DOI reservation alone is not publication.

Only `src/MC/fix_fast_swap.cpp` and `.h` are installed into LAMMPS.
See `docs/SOURCE_SNAPSHOT.md` for exact hashes and runtime verification.
Other LAMMPS releases require compatibility checks; this is not a complete LAMMPS distribution.

## Quick start

```bash
bash install_to_lammps.sh /path/to/lammps
# Configure/build as described in docs/INSTALL.md, then:
cd examples
mpirun -np 1 /path/to/lmp_mpi -in in.lj_fast
mpirun -np 1 /path/to/lmp_mpi -in in.lj_reference
```

Syntax:
```text
fix ID group-ID fast/swap Nevery Ntrial T seed types K t1 ... tK mu mu1 ... muK
```
`Ntrial` is a count of global attempted substitutions, not a swap fraction.
One sweep is `N` global attempts for `N` atoms. The optional `check yes period tolerance`
diagnostic uses a full-system `single()` enumeration, not an independent force-path reference.

## Reproduction

- `benchmark_workflow/`: complete inputs, prepared reference states, retained reference
  logs and offline tests for the 206-task affected-parallel recomputation. It deliberately
  does not rerun unaffected `atom/swap` or single-rank records. Its original README
  specifies node-specific timing checks and source/library identity requirements.
- `02_analysis_and_figure_sources/`: curated paired observations, provenance and plotting
  programs. From that directory run `python update_assets.py` after installing requirements.
  This regenerates the updated validation figures/statistical tables. It does not
  regenerate retained closure/exact assets. The application plotting sources are
  also included; see `application/REPRODUCE_FIGURE_8.md` and the separate data archive.
- Zenodo companion archives: full completed raw outputs and the phase-diagram
  workflow, parameters, observations and analysis.

Offline checks, without launching MPI simulations:
```bash
python -m unittest discover -s benchmark_workflow/tests
python benchmark_workflow/scripts/workflow.py --dry-run --output results_dryrun
```
The dry run renders input files but launches no simulation. Fresh performance timings
need a consistent machine/runtime and should not be pooled with unlike geometries.

## Scope and limitations

See `docs/SUPPORTED_SCOPE.md`. Tested models include LJ/cut, table-LJ, Morse,
Buckingham, Buckingham+Coul/cut, Coul/DSF, Yukawa, LJ+Yukawa overlay, ZBL,
ternary types and unequal LJ cutoffs. Coverage is limited to documented settings;
`single()` availability alone does not certify arbitrary modifiers or custom models.
EAM/many-body, KSpace, topology and tail corrections are outside this implementation.
It is not a replacement for grand-canonical particle exchange or VC-SGC sampling.

## Licence and citation

Extension and derived software: GPL-2.0-or-later; see LICENSE and COPYING.
Original notices in vendored sources are retained. Curated observations are CC BY 4.0;
see LICENSE_DATA.txt. No editorial correspondence, access credentials or build binaries
are included. Cite CITATION.cff and the actual archival version used. The preparation
date/version here does not mint a new DOI or imply journal acceptance.
