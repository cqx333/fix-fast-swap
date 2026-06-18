# Examples

Run examples from this directory so that relative paths such as
`tables/pair_lj.table` and `results/*.data` resolve correctly:

```bash
cd examples
mkdir -p results
mpirun -np 1 /path/to/lmp_mpi -in in.exact__lj_cut__fast
```

For comparison with the original LAMMPS implementation:

```bash
mpirun -np 1 /path/to/lmp_mpi -in in.exact__lj_cut__atom
mpirun -np 1 /path/to/lmp_mpi -in in.exact__lj_cut__fast
```

Representative cases:

- `in.exact__lj_cut__fast`: binary LJ/cut exact-consistency example.
- `in.exact__ternary_lj_cut__fast`: three swappable atom types.
- `in.exact__morse__fast`: analytic non-LJ pair style.
- `in.exact__table_lj__fast`: tabulated pair potential; requires
  `tables/pair_lj.table`.
- `in.exact__unequal_lj_cut__fast`: unequal type-pair cutoff boundary case.
- `in.parallel24_pair__lj_cut__fast`: larger MPI-oriented example.
- `in.boundary__*.fast`: expected-rejection examples for unsupported boundary
  cases.

