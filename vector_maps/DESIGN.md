# vector_maps — config-driven refinement runner (design)

Target shape for consolidating the current `fit_lattice_*` driver scripts into a
single runner driven by one TOML config (+ CLI overrides). One config drives any
refinement method; a separate batch layer maps results across a parameter sweep.
This mirrors the simulation side: a validated TOML feeding a single runner, with
sweep/aggregate concerns kept as an outer layer.

The *fitting* is unchanged — `refinement_run` + `dicts_handling` are reused as-is.
What is new is the orchestration around them: the config layer, `--set`
overrides, the seed sidecar, the `expand` / detect-step driver, and the batch
maps.

## Single-frame config (`fit.toml`)

```toml
[io]
folder = "./"                    # holds <fname>.tif + detected points
fname  = "sample_010_haadf"      # stem, no .tif

[calibration]
source = "sidecar"               # "sidecar" = <fname>_frame.txt | "value"
value  = 0.0188                  # nm/pixel, used when source = "value"
# precedence: --calib (CLI) -> sidecar / value (per source) -> error if neither

[lattice]
abg      = [0.32, 1.18, 145.5]   # a, b (nm), gamma (deg)
base     = [0.0, 0.0, -45.0]     # shx, shy, phi
fit_abg  = [true, true, true]    # a per-pass `fit = { abg = [...] }` overrides these,
fit_base = [true, true, true]    #              `fit = { base = [...] }`

# one [[motif]] block per column (top-level array-of-tables). label = dict key;
# el -> motif 'atom'; I/use = intensity weight + include flag (optional, default
# 1 / true); eq = ["= ...", "= ..."] couples a coord and disables its fit.
[[motif]]
label = "A_1"
el    = "Ta"
coord = [0.0, 0.0]
fit   = [false, false]

[[motif]]
label = "B_1"
el    = "Te"
coord = [0.0, 0.25]
fit   = [true, true]

[[motif]]
label = "B_2"
el    = "Te"
coord = [0.0, 0.75]
fit   = [true, true]

[extra_pars]                     # only for eq-coupled motifs; maps to (value, spec) tuples
# db_dist = { value = 0.1, fit = true }

[run]
gui       = true                 # open sliders on gui-marked passes; --no-gui overrides
seed      = true                 # persist + reuse the hand-tuned start
seed_file = "{fname}.start.toml" # separate sidecar (main config stays pristine)

passes = [
  { name = "prefit", sub_area = [1.0,3.0,1.0,3.0], vec_scale = 0.01, max_dist = 0.15, gui = true },
  { name = "fixed",  sub_area = [0.5,4.5,0.5,4.5], vec_scale = 0.25, max_dist = 0.15, save = true, fit = { abg = [false,false,false] } },
  { name = "free",   sub_area = [0.5,4.5,0.5,4.5], vec_scale = 0.25, max_dist = 0.15, save = true, fit = { B_1 = [true,true], B_2 = [true,true] } },
]
```

## Sections

- **`[io]`** — data directory + stem.
- **`[calibration]`** — `source = "sidecar"` reads nm/px from `<fname>_frame.txt`
  via `read_frame_calib`; `"value"` uses the inline constant; CLI `--calib`
  overrides both.
- **`[lattice]`** — starting `abg`/`base` and their fit flags (defaults; a seed
  sidecar overrides them at runtime).
- **`motif`** — array of columns. `label` is the dict key; `el` maps to the motif
  `atom` field; `I`/`use` are the intensity weight + include flag; `fit = [x, y]`
  per coord; `eq = ["= ...", "= ..."]` couples a coord (disables its fit) and
  reads from `[extra_pars]`.
- **`[run]` + `passes`** — the schedule. Detection is a *step* within `passes`
  (below), not a separate section; a detect step as the first pass does the
  initial detection.

## Schedule (`passes`)

`passes` is a list of *steps* run top to bottom; refined params and the active
point set carry forward. A step is either a **fit** or a **detect**.

**Fit step** fields:

- **`sub_area`** `[x0, x1, y0, y1]` — ROI.
- **`max_dist`** — pairing cutoff (required by `refinement_run`). Any other
  `refinement_run` kwarg (`kernel`, `relative_to`, `shift_ab`,
  `recall_zero`, `export_sublattice_xy`) is also accepted on a step and passed
  through.
- **`fit`** — fit mask, keyed by `abg` / `base` / motif label / extra_par name, each
  a bool list (`true` = refine, `false` = hold; an extra_par takes a one-element
  `[true]`/`[false]`); written into `fit_abg` / `fit_base` / the motif `fit` / the
  extra_par fit flag. The mask **mutates the running state (cumulative)**: a pass
  changes only the params it lists, others keep whatever the previous pass left them.
  `eq`-coupled coords ignore it — stage such a motif through its extra_par's flag
  (an `eq`-coupled extra_par can't be toggled and is rejected).
- **`vec_scale`**, **`save`** (default `false`), **`gui`** (default `false`).
- **`refine`** (default `true`) — `false` = plot-only: skip the optimiser,
  compute from current params, emit a / b / diff stats. `--no-fit` forces it
  everywhere.
- **`add`** — introduce new motif atoms here; persist for later steps.
- **`expand`** — growing-ROI loop (below).

**Detect step** — re-detect + optional merge, for the interleaved flow:

```toml
{ name = "redetect", detect = { ptonn = [0.6, 0.4], merge = true, imsize = [5.0, 5.0] } }
```

Wraps `detect_columns`: one detect pass per `ptonn` entry (the `percent_to_nn`
fit window), each chained on the previous pass's residual; `merge = true` combines
them into the `sub_AB`
point set the next fit consumes, expressing **fit → re-detect → merge → re-fit**.
`imsize` (nm) is required.

### `add` / `expand` examples

`[[passes]]` is the block spelling of the same `passes` list (use one form per
file) — multi-part steps read better as blocks:

```toml
[[passes]]                       # fit the lattice on A_1 alone, then add the Te columns
name = "lattice"
sub_area = [0.5,4.5,0.5,4.5]
vec_scale = 0.1
max_dist = 0.1

[[passes]]
name = "add_Te"
sub_area = [0.5,4.5,0.5,4.5]
vec_scale = 0.25
max_dist = 0.1
add = [ { label = "B_1", el = "Te", coord = [0.0,0.25], fit = [true,true] } ]

[[passes]]                       # growing ROI; two fits per box (abg held, then free)
name = "roi"
vec_scale = 0.01
max_dist = 0.1
expand = { from = [2,4,2,4], to = [2,14,2,14], step = 2 }
  [[passes.body]]
  fit = { abg = [false,false,false] }
  [[passes.body]]
  fit = { abg = [true,true,true] }
```

In `expand`, the dims that differ between `from`/`to` step by `step` until
reaching `to`; the step runs once per box, carrying params forward.

### Seed

With `seed = true`, the hand-tuned start persists to `seed_file` (default
`{fname}.start.toml`) — a separate sidecar holding only `abg`/`base` (and any
declaratively-added atoms). Precedence: `[lattice]` defaults → seed sidecar →
`--set`. The main config is never rewritten.

## Override

```
vmap-run --config fit.toml --set io.fname=frame_07 --set lattice.abg="[0.32,1.18,145]"
```

Flags: `--no-gui`, `--no-fit`, `--calib`.

## Batch layer (`batch.toml`)

A separate command (`vmap-sweep`) wraps the single-frame runner across a
parameter-sweep manifest — the outer layer, analogous to how the simulation side
wraps its per-item worker with the generator/aggregator.

```toml
[manifest]
path = "manifest.csv"            # the sweep manifest

[filter]                         # process only rows matching ALL of these
detector   = "haadf"
phonons    = 8
scan_s     = 50.0
thickness  = 50.0
borders    = 5.0
blur_sigma = 0.25
fph_sigma  = 0.1

[fit]
config = "fit.toml"              # per-frame config; io.folder/io.fname set per row

[run]
retries = 3                      # per-frame fit retries

# one PNG per [[maps]] entry, over (tilt_a, tilt_b)
[[maps]]
field = "residual_in_pm"
title = "Residual (pm)"

[[maps]]
field = "motif_dist"
title = "Sublattice shift"
scale = 1000
significant = "std"

[[maps]]
field = "std"
title = "Residual std"
scale = 1000
```

- **`[filter]`** selects one parameter slice; the sweep reports the match count
  plus an unmatched-values sanity dump.
- **`[fit]`** reuses the single-frame `fit.toml` per row (overriding only `io`),
  so the two layers stay DRY.
- **`[[maps]]`** — `field`/`significant` name columns the per-frame extractor
  produces (currently `a_fit`/`b_fit`/`g_fit`, `residual_in_pm`, `std`,
  `atoms_used`, `motif_dist`; per-sublattice stats like ellipticity need the
  export-sublattice CSVs — future). `significant` draws a point hollow when
  `|field| < its std`; `scale` is a unit multiplier (e.g. ×1000 → pm).
- Outputs land beside the manifest: `coverage.png`, `lookup_augmented.csv`, one
  PNG per map entry.

## Relation to existing code

- `lat_params` / motif / `extra_pars` map onto the config sections — restructured,
  not copied verbatim (motif becomes a labelled array carrying `I`/`use`,
  `extra_pars` becomes `{value, spec}` tables).
- The schedule is the per-driver pass sequence made declarative; each fit step is
  one `refinement_run` call with its params, unchanged.
- Detect steps wrap `detect_columns`; calibration uses `read_frame_calib`. Both
  land with the driver-harmonisation work (not yet in `main`).

## Phase 2 — iterative intensity-stratified detect/fit

For structures where the weak (low-I) columns can't be detected reliably up
front, the fit is bootstrapped from the strong columns, re-detecting and
refining in a loop.

The loop (single frame, interactive):

1. detect the bright columns (set by `separation` / `threshold`)
2. fit the lattice on them
3. re-detect the bright columns, **seeded by the fit**
4. add the weak columns as a fixed schema at guessed coords (e.g. `(0.25, 0.75)`,
   `fit = [false, false]`, no `eq` needed) — **no refit**; this only projects
   them through the current lattice to seed step 5
5. detect the weak columns: seeded by the schema, on the residual after the
   strong model (masked)
6. fit the full structure (weak now free)
7. repeat 3–6 until satisfied — **the stop is a manual choice each round**, not
   an automatic convergence criterion

Concretely, one round (stages 2–6):

- stage 2 — `run_fit_pipeline` with `export_sublattice_xy` → per-sublattice
  `…_free_motif_A/_full.csv`, `…_B/_full.csv` (the seeds).
- stage 3 — `detect_columns(ptonn=0.6, start_csv=csv_A)` — re-detect bright,
  seeded by the fit's A positions.
- stage 5 — `detect_columns(ptonn=0.4, source_fname=…_A_rerun…diff2.tif,
  start_csv=csv_B)` — detect weak on the strong-subtracted residual, seeded.
- merge — `concat(A, B)` → `…_sub_AB_xyI.csv`.
- stage 6 — `run_fit_pipeline(dataset_fname="…_sub_AB")` — fit on the merge.

New primitives beyond phase 1:

- **A — seeded detection.** A detect step takes a seed (`start_csv`) = the prior
  fit's per-sublattice output (or the projected guessed schema). The phase-1
  detect step runs fresh; this adds the seed, and the feeding fit pass must set
  `export_sublattice_xy`.
- **B — manual `repeat` loop.** A schedule block wrapping the detect→fit body,
  repeated until the user stops — manual, per round, no automatic criterion.
  Interactive only.
- **C — intensity-scoped / masked detection.** `source_fname` (detect on the
  residual after the prior sublattice's model) does the scoping; `separation` /
  `threshold` pick the columns. The phase-1 detect step already chains residuals
  and merges; this scopes it by intensity.

Stage 4 (add the schema) reuses phase-1 primitives: a plot-only pass
(`refine = false`) with `add` at guessed coords — no new construct.

Because B's stop is manual, the loop is the **single-frame, interactive
recipe-building** phase. The batch sweep does not run it: once the recipe (pass
sequence + seed + schema) is settled on one frame, the sweep **replays it
headless** across the manifest.

## Deferred

- Interactive residual-guided atom adding (click-to-place on the
  image − model residual), persisted to the seed.
