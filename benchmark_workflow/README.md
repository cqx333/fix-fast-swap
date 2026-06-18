# fast/swap validation and performance workflow

This bundle generates and runs a sequential validation and benchmark workflow
for the current `fix fast/swap` implementation.  It is intended for cluster runs
where at most 24 MPI ranks are available.

## One-command use

From this folder on the cluster:

```bash
bash run_all.sh /path/to/lmp
```

Use the modified LAMMPS executable that contains `fix fast/swap` and the pair
styles listed below.

For SLURM systems, either edit `submit_slurm.sh` or submit with environment
variables:

```bash
sbatch --export=ALL,LMP_BIN=/path/to/lmp,MPI_LAUNCHER=srun,NP_FLAG=-n submit_slurm.sh
```

For OpenMPI/MPICH-style launchers, the default is:

```bash
MPI_LAUNCHER=mpirun NP_FLAG=-np MAX_RANKS=24 bash run_all.sh /path/to/lmp
```

The workflow runs jobs sequentially.  It does not submit all simulations at the
same time.

All runs, including `ranks=1`, are launched through the configured MPI launcher.
This avoids environment differences between direct serial execution and MPI
execution on private workstation-style clusters.

## Suites

The default suite is `all`.

```bash
SUITE=representative bash run_all.sh /path/to/lmp
SUITE=scaling bash run_all.sh /path/to/lmp
SUITE=strong_scaling bash run_all.sh /path/to/lmp
SUITE=size_scaling bash run_all.sh /path/to/lmp
SUITE=unequal_cutoff_scaling bash run_all.sh /path/to/lmp
SUITE=nonlj_parallel_scaling bash run_all.sh /path/to/lmp
SUITE=validation bash run_all.sh /path/to/lmp
SUITE=validation_exact bash run_all.sh /path/to/lmp
SUITE=validation_parallel_stats bash run_all.sh /path/to/lmp
SUITE=boundary bash run_all.sh /path/to/lmp
SUITE=timeseries bash run_all.sh /path/to/lmp
SUITE=timing_repeats bash run_all.sh /path/to/lmp
SUITE=rich_validation bash run_all.sh /path/to/lmp
SUITE=nonlj_rich_validation bash run_all.sh /path/to/lmp
SUITE=paper_full bash run_all.sh /path/to/lmp
SUITE=all bash run_all.sh /path/to/lmp
```

To add only the non-LJ dense validation runs to an existing completed workflow:

```bash
APPEND_STATUS=1 SUITE=nonlj_rich_validation bash run_all.sh /path/to/lmp
```

`representative` runs:

- single-rank ORI/OPT timing for representative pair styles;
- 24-rank ORI/OPT timing for the same pair styles.

`scaling` runs:

- LJ single-rank size scaling;
- LJ strong-scaling data at 1, 2, 4, 8, 12, 16, and 24 ranks;
- LJ atom-count size scaling in both single-rank and 24-rank layouts;
- non-LJ/cut parallel scaling for representative pair styles.

`strong_scaling` runs only the fixed-size LJ/cut strong-scaling suite at 1, 2,
4, 8, 12, 16, and 24 ranks, bounded by `MAX_RANKS`.  The fixed box is enlarged
to `8 x 8 x 96` FCC cells (`24576` atoms) so that the fast/swap timing is less
dominated by low-rank MPI overhead.  The rank-1 point is also launched as
`mpirun -np 1 ...`.

`size_scaling` runs LJ/cut tests at several atom counts.  The single-rank boxes
are cubic (`500`, `864`, `1372`, `2048`, `2916`, `4000`, `5324`, and `6912`
atoms).  The 24-rank boxes keep the long `z` dimension needed for true parallel
`fast/swap` proposals while varying the transverse area (`3456`, `6144`, `9600`,
`13824`, `18816`, `24576`, `31104`, and `38400` atoms).  The sweep counts in
this suite are higher than in the quick-look timing tests so that fast/swap
times are less affected by millisecond-level timing noise.

`nonlj_parallel_scaling` runs fixed-size 6144-atom parallel scaling tests at 2,
8, and 24 ranks for `morse`, `table_lj`, `buck_coul_cut`, `coul_dsf`,
`hybrid_overlay_lj_yukawa`, and `zbl`.  These cases test analytic, tabulated,
charged, composite, and expensive screened pair interactions beyond LJ/cut.

`unequal_cutoff_scaling` isolates the conservative maximum-cutoff path used for
unequal type-pair cutoffs.  It runs serial boxes of `864`, `2048`, `4000`, and
`6912` atoms and 24-rank long-box tests of `6144`, `13824`, and `24576` atoms.
This suite is intended to determine whether the low single-rank speedup observed
in the small representative unequal-cutoff case is a small-system effect or a
true overhead boundary of the current implementation.

`validation` combines three correctness-oriented suites:

- `validation_exact`: single-rank trajectory-level equality for all
  representative pair styles.  The `fast/swap` inputs enable `check yes`, which
  periodically compares localized and full energy differences.
- `validation_parallel_stats`: 24-rank statistical consistency for binary
  LJ/cut, unequal-cutoff LJ/cut, ternary LJ/cut, Morse, Buck/Coul/cut,
  Coul/DSF, hybrid/overlay LJ+Yukawa, and ZBL over three independent seeds.
- `boundary`: expected-rejection tests for pair tail corrections, KSpace
  long-range energy, and non-identical charges on swappable charged atoms.

`timeseries` is a high-information diagnostic suite rather than a timing suite.
It writes thermo output every MC sweep for all representative pair styles in
single-rank exact-comparison runs, and for LJ/cut, unequal-cutoff LJ/cut,
ternary LJ/cut, Morse, Buck/Coul/cut, Coul/DSF, hybrid/overlay LJ+Yukawa, and
ZBL in 24-rank statistical runs over three seeds.  The analyzer
exports these logs to per-run CSV files under `results/timeseries/`, including
energy, energy per atom, type counts, compositions, MC attempts, MC accepts, and
acceptance ratios at every recorded snapshot.

`timing_repeats` repeats selected LJ/cut serial and 24-rank timing points over
three independent seeds without dense thermo output.  It is intended to
separate true scaling trends from run-to-run timing noise.

`rich_validation` generates and runs both `timeseries` and `timing_repeats`.
`nonlj_rich_validation` generates only the added non-LJ 24-rank statistical and
per-sweep trajectory runs.  Use it with `APPEND_STATUS=1` when adding the new
non-LJ validation data to an existing `results/run_status.json` without
repeating the full workflow.

`all` and `paper_full` are both one-command final-data suites.  They include the
performance, correctness, boundary, dense-snapshot, and timing-noise outputs
needed for article and supplementary-material figure generation.

## Representative pair styles

The generated representative cases are:

- `lj_cut`
- `unequal_lj_cut`
- `morse`
- `buck_coul_cut`
- `coul_dsf`
- `table_lj`
- `hybrid_overlay_lj_yukawa`
- `zbl`
- `ternary_lj_cut`

The charged cases assign identical charges to all swappable atom types, matching
the current code boundary: `fix fast/swap` changes atom type only.

## Parallel-box design

Parallel representative tests use a long box of `4 x 4 x 96` FCC unit cells and
force the processor grid to `processors 1 1 P`.  At `P=24`, the local domain
length in the decomposed direction is still large enough that the active
subregion size exceeds the cutoff-plus-skin communication radius for the
generated pair styles.  This is designed to avoid the serial fallback path and
produce meaningful parallel timing for `fast/swap`.

The analyzer scans the logs for fallback messages.  Any fast parallel run with a
fallback warning is marked as `FAIL_FALLBACK` and should not be used as a
parallel-performance data point.

## Outputs

Generated inputs:

```text
inputs/in.*
tables/pair_lj.table
results/manifest.json
```

Run outputs:

```text
logs/log.*.lammps
results/run_status.json
results/run_status_flat.csv
results/performance_summary.csv
results/performance_summary.md
results/strong_scaling_summary.csv
results/strong_scaling_summary.md
results/size_scaling_summary.csv
results/unequal_cutoff_scaling_summary.csv
results/unequal_cutoff_scaling_summary.md
results/nonlj_parallel_scaling_summary.csv
results/nonlj_parallel_scaling_summary.md
results/exact_consistency_summary.csv
results/exact_consistency_summary.md
results/parallel_statistical_consistency_summary.csv
results/parallel_statistical_consistency_summary.md
results/parallel_statistical_consistency_aggregate.csv
results/boundary_guard_summary.csv
results/timeseries_manifest.csv
results/timeseries/*.csv
results/timing_repeats_raw.csv
results/timing_repeats_summary.csv
```

`performance_summary.csv` contains the time per MC sweep and the ORI/OPT speedup
for each matched case.
`strong_scaling_summary.csv` additionally reports speedup and parallel
efficiency relative to the MPI-launched rank-1 run of the same fix.
The consistency summaries report trajectory-level equality, stationary
composition/energy agreement for parallel runs, and expected rejection of models
outside the current formulation.
The time-series CSV files provide the raw snapshot-level energy and composition
records needed for high-density validation figures and supplementary plots.

## Running selected cases

After inputs are generated, selected runs can be executed by id, case name, or
suite name:

```bash
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --ids lj_cut table_lj
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --ids parallel24_pair__lj_cut__fast
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite strong_scaling
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite unequal_cutoff_scaling
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite nonlj_parallel_scaling
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite validation_exact
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite rich_validation
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite nonlj_rich_validation --append-status
python3 scripts/run_workflow.py --lmp /path/to/lmp --np-flag=-np --suite paper_full
```

Then re-run:

```bash
python3 scripts/analyze_results.py
```
