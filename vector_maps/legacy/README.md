# legacy

Pre-refactor, hardcoded per-material fitting scripts, kept for reference.
Superseded by the modular pipeline (`routines` / `refinement_routines` /
`plot_routines` / `dicts_handling`) driven by `vmap_run.py` + the
`examples/fit_*.toml` configs.

They import the modular modules from the package root (`from routines import *`),
so to run one from here add the parent directory to the path, e.g.
`PYTHONPATH=.. python legacy/fit_lattice_Si.py`.
