# Phase-Application Reproducibility

This directory contains simulation observations and numerical analysis for the phase-diagram application. No LAMMPS run is required to regenerate fits, bootstrap intervals and figures.

## Data

- `data/snapshots/*.npz`: 422 production histories and 42 pure-A enthalpy series; fields are in `data/snapshot_schema.json`.
- `data/inputs/`: exact temperature configurations, supplied summaries, case metadata and provenance. Original absolute paths are provenance, not dependencies for cached analysis.
- `data/trajectory_statistics.csv`, `state_composition_and_acceptance.csv`, `initial_state_comparison.csv`: correlated statistics, actual counters and preparation comparisons.
- `data/phase_boundaries.csv`, `bootstrap_samples.csv`, `fit_sensitivity.csv`: endpoints, 500 bootstrap realizations and sensitivities.
- `data/structure_identification.csv`: classification of 886 structures. Full atomic dumps/restarts are not duplicated in this compact archive; fingerprints identify the original files.
- `data/historical_*` and `replacement_*`: phase-crossing audit and replacement provenance. Liquidized solid starts are not counted as solid replicas.

## Dependencies

Python, NumPy, SciPy, Matplotlib. OVITO 3.10.6 is needed only to repeat structural classification against original atomic dumps.

## Reproduce

From this directory:

```bash
python prepare_application.py --cached-only
python make_application_assets.py
```

The first command reconstructs means, fits and the 500-replicate bootstrap from the supplied histories and metadata. The second generates figures/tables and two-sided/Holm-adjusted diagnostics. Random seed: 20261018.

To redraw only Figure 8 without recalculating fits or rewriting any data CSV:

```bash
python make_application_assets.py --main-only
```

Figure 8(d) connects the six computed coexistence temperatures to the prescribed
pure-A endpoint at composition zero and temperature 0.65. These extra line segments
are visual guides, not new binary observations or extrapolated confidence intervals.
The endpoint and the corresponding solidus inset use the same fixed reference.
`figure8_endpoint_guides.json` records the guide coordinates separately from the data.

In the complete tree, figures/tables go to `../../01_manuscript_tex_and_source`. In a standalone extracted archive without that directory, they go to `generated_tex/` in this directory. PNG previews are 600dpi; PDFs are vector outputs.

The default mode of `prepare_application.py` reads original results and checks pure-A reference hashes. `audit_structures.py` needs original dumps; it cannot reconstruct coordinates from averaged observables.

## Rules and Limitations

The primary fit retains complete, structurally correct phase branches uniformly across temperatures. Warnings are reported rather than automatically deleting the high-composition liquid range. Preparation families have equal weight. NPT holds contribute to the time mean, not to an inflated count of independent MC observations. Cycle blocks and seed resampling propagate composition and enthalpy uncertainty.

The melting reference is fixed at 0.65. Intervals exclude melting-reference, finite-size, arbitrary model-form and application timestep-extrapolation uncertainty. Five unadjusted intervals exclude zero and are retained with Holm diagnostics and boundary sensitivities. None of these checks proves global ergodicity.

Workflow scripts, potential and metadata document the simulation schedule, not additional simulations performed locally.
