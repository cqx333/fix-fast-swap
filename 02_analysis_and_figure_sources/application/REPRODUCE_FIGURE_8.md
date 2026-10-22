# Phase-application plotting

Extract the matching Zenodo `application_reproducibility.zip` into this directory
to supply `data/` and the remaining workflow files, then run:

`python make_application_assets.py --main-only`

This preserves the recorded coexistence points, fitted models and bootstrap intervals.
Figure 8(d) includes visual guide segments from the highest sampled temperature
to the prescribed pure-A melting reference (c=0, T=0.65). The endpoint is not a
new binary simulation. Main-only plotting does not rewrite the archived CSV data.
Simulation inputs and run metadata document the application protocol;
this command regenerates the figure from the supplied observations and fits.
