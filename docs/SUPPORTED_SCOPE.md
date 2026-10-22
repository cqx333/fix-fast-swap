# Validated boundary

The changed energy must be a sum over affected atom pairs, plus terms invariant
under the substitution. Local `single()` energies must agree with the actual global
energy definition, and the full list must cover every possible before/after partner.
The method changes type, not charge or other per-atom attributes.

Validated models/settings are given explicitly in the companion SI. These include
11 configurations: binary LJ/cut, unequal LJ/cut, ternary LJ/cut, tabulated LJ,
Morse, Buckingham, Buckingham+Coul/cut, Coul/DSF, Yukawa, LJ+Yukawa overlay and ZBL.
Charged swappable atoms have identical fixed charges. Equal masses are tested.
Unequal cutoffs are supported but conservative accepted-move rebuilding can
remove the serial speed benefit. Pair tail corrections are rejected.

Concurrent moves are demonstrated in fixed orthorhombic boxes with unchanged
decomposition and the geometric independence check satisfied. Otherwise the
implementation selects serially. Shrinking cells or repartitioning require a
renewed feasibility check and separate regression tests. The application confines
MC to fixed-volume blocks. Do not infer a guarantee for arbitrary fix energies,
accelerator backends, unequal-mass velocity handling, or periodic self-image terms.

EAM and other many-body models, reciprocal-space KSpace terms, molecular topology,
and incompatible charge/type settings are outside this implementation. Initialization
guards are not a proof that every accepted custom pair style is additive. Numerical
validation is finite evidence, not a universal parameter certification.
