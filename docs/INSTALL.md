# Installation

## 1. Copy the fix into LAMMPS

From this repository root:

```bash
cp src/MC/fix_fast_swap.cpp /path/to/lammps/src/MC/
cp src/MC/fix_fast_swap.h   /path/to/lammps/src/MC/
```

The file names must remain:

```text
fix_fast_swap.cpp
fix_fast_swap.h
```

LAMMPS discovers the fix through the `FixStyle(fast/swap,FixFastSwap)` macro in
the header file.

## 2. Configure LAMMPS

At minimum, enable MPI and the MC package:

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=yes \
  -DPKG_MC=yes
```

Enable any additional packages needed by the pair styles you want to test. For
the benchmark suite in this repository, the exact package set depends on your
LAMMPS build, but representative pair styles include LJ/cut, table, Morse,
Buckingham+Coul/cut, Coul/DSF, Yukawa, and ZBL.

## 3. Build

```bash
cmake --build . -j 4
```

If the LAMMPS executable links to a shared `liblammps_mpi.so`, make sure the
runtime loader sees the newly built library before older LAMMPS libraries:

```bash
export LD_LIBRARY_PATH=/path/to/lammps/build:$LD_LIBRARY_PATH
```

## 4. Check that the fix is available

Run:

```bash
/path/to/lmp_mpi -h | grep fast/swap
```

or run one of the examples:

```bash
cd examples
mkdir -p results
mpirun -np 1 /path/to/lmp_mpi -in in.exact__lj_cut__fast
```

## 5. Run the benchmark workflow

```bash
cd benchmark_workflow
bash run_all.sh /path/to/lmp_mpi
```

For systems without `mpirun` on the default `PATH`, load the MPI environment
first or provide the launcher explicitly:

```bash
MPI_LAUNCHER=/path/to/mpirun NP_FLAG=-np bash run_all.sh /path/to/lmp_mpi
```

