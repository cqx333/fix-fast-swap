# Source snapshot

This upload-ready GitHub package uses the optimized source files from:

```text
fast_swap_parallel_optimized_20260518/fix_fast_swap.cpp
fast_swap_parallel_optimized_20260518/fix_fast_swap.h
```

SHA256 hashes of the copied source files:

```text
333ffe8393ee2fefd5d5c7650be93f8b3d57bc3606478d28286b24ad45d7b7a0  src/MC/fix_fast_swap.cpp
162b781f529f0ce80047e09f031816755cdc7393c6dcb42ffc3cd5475f1dade6  src/MC/fix_fast_swap.h
```

These files include:

- localized pair-energy evaluation through `pair->single()`;
- full neighbor-list request and unequal-cutoff support through enlarged local
  neighbor coverage;
- binary and multi-type chemical-potential lists;
- optional single-rank `check yes Period Tolerance` energy-closure diagnostics;
- geometric parallel proposal selection with serial fallback;
- initialization guards for unsupported many-body, KSpace, tail-correction, and
  incompatible charged-type cases.

