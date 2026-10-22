# Validated fix fast/swap source snapshot

This directory contains the implementation used for the parallel results in
the accompanying revised manuscript and Supplementary Information (SI).
The target LAMMPS snapshot reports **17 Apr 2024 (Development)**.
It is a LAMMPS extension, not a standalone program or a complete LAMMPS source tree.

## Files and identity

- `src/MC/fix_fast_swap.cpp`: implementation, including the 20 October 2026
  rank-local random-seed initialization.
- `src/MC/fix_fast_swap.h`: corresponding header.
- `COPYING`: complete GNU GPL v2 licence text.
- `LICENSE`: GPL-2.0-or-later notice.
- `SHA256SUMS.txt`: fingerprints of the files in this source package.

The implementation SHA-256 is
`66c3edd4e9c06665dc9f5132b4a0a64a47b931914330291c20db57deb1d1331b`.
The tested Linux shared-library SHA-256 is
`5fa13058c965a2f0ffec776343acce4bc021a5f6988ad2937a82502b8cf2156d`.
Rebuilding with a different toolchain normally changes the library fingerprint.
The thin `lmp_mpi` executable fingerprint alone does not identify its loaded library.

## Integration

Copy the two implementation files to `src/MC/` in the compatible LAMMPS tree.
Configure/build LAMMPS with `PKG_MC=ON` and `BUILD_MPI=ON`, plus the packages
required by the chosen pair styles and guard tests. Rebuild the shared library
and executable together. Ensure that the launcher and runtime library resolve
to that build, not an older installation. These files were validated with
GNU C++ 15.3.0 and MPICH 5.0.2; other LAMMPS releases require a compatibility check.

On initialization the fix prints:

```text
Fix fast/swap RNG revision 20261020: rank-local seeds; serial stream unchanged
```

## Input syntax

```text
fix ID group-ID fast/swap Nevery Ntrial T seed types K t1 ... tK mu mu1 ... muK
```

`Nevery` and `Ntrial` are positive integers. `Ntrial` is the requested number
of global attempted substitutions per invocation, not a swap fraction.
`K >= 2`, and chemical potentials follow the listed type order.
An example with two types is:

```text
fix mc all fast/swap 1 864 1.0 24681357 types 2 1 2 mu 0.0 0.2
```

The optional internal diagnostic is `check yes period tolerance` (or `check no`).
It uses a full-system `single()` sum, so it is an enumeration check, not an
independent force-path energy reference. Independent closure tests accompany
the validation data.

## Random streams and retained evidence

For input seed `s`, MPI rank `r` uses
`1 + ((s - 1 + r) % 2147483646)` for its local choices, with
`1 <= s <= 2147483646`. Rank zero retains the original input seed.
The separate synchronized global-selection stream, pair-energy evaluator,
and Metropolis acceptance rule are unchanged. Distinct seed initialization
does not by itself prove independence of every finite pseudorandom sequence.

All affected parallel localized trajectories and timings in the accompanying
non-application validation were repeated with this implementation. Global
reference and single-rank records were retained with explicit provenance.
The phase-diagram workflow, observations and analysis are supplied in a
dedicated application archive.

## Validated scope

Atomic systems with pairwise-additive type-dependent energy, consistent
`pair->single()` energies, and complete before/after neighbor coverage.
The tested models, coefficients, modifiers, masses, and charge assumptions
are listed in the SI. Many-body, KSpace, molecular topology and pair-tail
energy terms are rejected. Equal fixed charges are required for the tested
charge styles; unequal-mass MD/MC and self-image interactions are not validated.
Do not infer support for every parameter, modifier, accelerated backend, or
custom pair style from the existence of a `single()` method alone.

## Licence

GNU General Public License, version 2 or (at your option) any later version.
See `LICENSE`, `COPYING`, and the notices in the source files.
