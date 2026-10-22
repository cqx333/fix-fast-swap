# Prepared revision 2026.10.21-r2

- Rank-local seeds differ by MPI rank while rank zero retains the original stream.
- Pair evaluator, Metropolis criterion and synchronized scheduling stream are unchanged.
- All affected non-application parallel trajectories and timings were recomputed.
- Added independent closure, mobile MD/MC and repeated three-dimensional size evidence.
- Documentation distinguishes local 2024 SGCMC behavior from newer atomic/energy interfaces.
- The introduction acknowledges SGCMC's parallel sampling-window algorithm; only
  its generic full-energy fallback carries the single-process restriction.
- Figure 8 connects the unchanged coexistence points to the prescribed pure-A
  endpoint using visual guides, with no extra binary simulation data.
- Release files retain source/runtime fingerprints and distinguish old/new cohorts.

This file describes a prepared local release candidate, not an online publication event.
The phase-diagram workflow, parameters, observations and analysis are archived separately.
