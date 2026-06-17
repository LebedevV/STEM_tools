# vector_maps

Locate atomic-column positions on HAADF STEM images and refine them against an
idealized 2D Bravais lattice (arbitrary cell + motif) with per-degree-of-freedom
control, producing residual vector maps and statistics.

## Workflow

1. **Detect columns.** `detect_columns` (the `detect_columns.ipynb` notebook or
   `detect_columns.py`, both wrappers around atomap) finds column positions,
   intensities, and ellipticities for a frame and writes a 14-column
   `<frame>_xyI.csv`.
2. **Fit the lattice.** Describe the cell, motif, calibration, and a sequence of
   refinement passes in a TOML config, then run `vmap_run.py`. The theoretical
   lattice is built, paired with the observed columns, and the free parameters are
   refined to minimize the mean column-to-column distance. Motif atoms can be tied
   together with equations (e.g. a dumbbell as a polar `extra_pars` vector).
3. **Read the output.** Each saved pass writes residual vector maps, histograms,
   and CSVs of the refined parameters, ratios, and the minimized distance.

See `DESIGN.md` for the full config schema and pass semantics.

## Examples

`examples/` has ready-to-edit configs; point each `[io]` at your own frame and its
`<fname>_xyI.csv` (detected columns), then run from this directory:

```bash
# TaTe2 -- one Ta + two Te sublattices; set [io] to your frame + _xyI.csv
python vmap_run.py --config examples/fit_tate2.toml --no-gui

# Si<110> dumbbell -- staged lattice -> dumbbell -> free; point [io] at your own frame
python vmap_run.py --config examples/fit_si.toml --no-gui

# Batch sweep over an abtem tilt series (one fit per frame, then maps vs tilt).
# Needs the simulation out_full tree (not shipped; generate it with abtem_run):
python vmap_sweep.py --config examples/batch_pm3m.toml
```

`fit_pm3m.toml` is the placeholder per-frame template the batch sweep fills in per row.

## Tests

```bash
python -m pytest -q        # or run a single file standalone: python test_routines.py
```

## Known issues / TODO

- ambiguous observed/theoretical column matches keep the closest candidate; a global assignment would be better
- auto-assessment of the i,j lattice-index range (number of unit cells in use)
- simultaneous BF/DF imaging support
- a dedicated minimizer term for negative distances

## Acknowledgements

- Lewys Jones for ideas and supervision
- Project SFI/21/US/3785 for financial support

Part of the code has been created with AI assistance (OpenAI GPT-5) and manually reviewed.
