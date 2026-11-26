# FoldJet AF3 input features

This package contains the CPU feature and native structure parsing subset of
official AlphaFold3 v3.0.1. See SOURCE.json for the exact revision and LICENSE
for its terms. JAX model inference is not included; FoldJet uses its separate
PyTorch AF3 port. Feature preparation uses NumPy and standard dataclasses; JAX, Haiku and Chex
are not dependencies. The local distribution version is 3.0.2; the upstream
source revision remains the one recorded in SOURCE.json.

Build through the FoldJet locked environment. Generate CCD indices with
`foldjet ccd prepare --components <components.cif> --rdkit <molecules.pkl> --out <database>`
and pass the same `--ccd-db <database>` to every FoldJet predictor. AF3 features
receive the explicit CCD index plus scoped glycan classification sets.
The external asset directory avoids losing generated files during venv rebuilds.
