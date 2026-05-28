# Worker architecture for abtem-run

This document captures the design decisions for the worker + aggregator that
consume the per-seed `.todo` files emitted by `generator_run.py`. It records
*what was chosen and why*, not how to use them — the latter goes in `README.md`.

Status: **implemented** — preserved as the design-rationale record. Inline
`# See docs/worker.md decision #N` references in `worker.py` / `aggregate.py`
point back here. (Some code references below name `cli.simulation_run`; after
the worker landed, that in-process path was renamed to `pipeline.py`.)

## Background

The pipeline naturally decomposes into three checkpoints (see V. Lebedev's
intent for the architecture):

1. **Atomic coordinates** `(x0, y0, z0)` — deterministic, from the TOML.
2. **Per-phonon displacements** `{xi, yi, zi}` — random, must be strictly
   reproducible given a seed.
3. **Scattering output** — handled by abTEM's multislice. Monte-Carlo-style
   randomness here is acceptable.

`generator_run.py` already emits a queue of per-seed work:

```
gen_<UTC_timestamp>/
├── run_manifest.json
└── <phase>_<hkl>_<tilt>/
    ├── <stem>.toml
    ├── seeds/
    │   ├── seed_000000.todo
    │   └── seed_000001.todo …
    ├── outputs/         (empty until workers run)
    └── aggregate/       (empty until aggregator runs)
```

The **worker** consumes one `.todo`, runs the simulation for that single
phonon snapshot, and drops its output into `outputs/`. The **aggregator**
means the per-seed outputs into `aggregate/`. This document covers the
worker side.

## Decisions

### 1. Worker interface: PUSH

```bash
abtem-run-worker <job_dir> <todo_path>
```

Each invocation processes exactly one `.todo`. The worker is stateless;
the caller (a bash loop, GNU parallel, slurm, or the convenience wrapper)
supplies the `.todo` paths.

**Alternative considered: PULL.** A `--job-dir=<dir>` worker that scans
for `.todo` files, claims one via atomic rename, and loops until none
remain. Rejected because the locking code adds complexity that
`parallel -j N` from the push side already solves with a more proven
mechanism.

**Implication.** The worker function `run_one_seed(job_dir, todo_path)`
is the unit of work. The CLI entry just unpacks `argv` and calls it.
The convenience wrapper (below) imports and calls the same function
directly — no subprocess overhead.

### 2. `cli.main()` becomes a convenience wrapper

The current `abtem-run` runs the full sweep in-process. Once the worker
and aggregator exist, that path is repurposed:

```
abtem-run         → generate → for each todo: run_one_seed(…) → aggregate
abtem-run-worker  → push-mode single-seed entry (for orchestration)
abtem-run-aggregate → aggregator entry (for orchestration)
```

The convenience wrapper keeps the simple case simple — one command, run
your config, get a result. Power users who want to scale out (slurm,
parallel, multi-machine) call the lower-level commands directly.

### 3. Single-snapshot output: dropped from code, reconstructable via config

This is the most consequential decision. Current `cli.simulation_run`
produces **two** result sets per `(phase, hkl, tilt)`:

1. **Static lattice** — built from `add_potential(surf)` with no phonon
   displacements. Detectors: HAADF + ABF + BF.
2. **Frozen-phonon stack** — built from
   `add_frozen_phonons_potential(ctx, surf)`. Detectors: HAADF + ABF
   (BF intentionally excluded).

These are scientifically distinct quantities:

- Static lattice = what the sample "would" look like at 0 K, no thermal
  motion. Useful as a column-position reference.
- Frozen-phonon average = closer to a real microscope at non-zero
  temperature.

**However**, mechanically the static-lattice path is just the σ=0 limit
of FrozenPhonons:

```python
FrozenPhonons(num_configs=1, sigmas=0, seed=anything)
    .to_atoms_ensemble().trajectory[0]
# is identical to the input atoms (zero displacement)
```

The dual code path is a redundancy that V. flagged in his own README
TODO ("first frame is simulated separately from frozen phonons, and
this simulation is just repeated later on. maybe add a flag?").

#### Three options were considered

**Option A — keep the dual concepts.** Worker has two branches: one for
"single snapshot" and one for "phonon snapshot." Generator emits a
`single.todo` alongside `seeds/seed_NNNNNN.todo`. Aggregator merges two
separate output sets.

Pros: byte-for-byte preservation of current behavior, including the
BF/no-BF detector asymmetry.
Cons: two code paths everywhere; the legacy distinction propagates
forward.

**Option B — unify: "single" is just σ=0.** Worker has one code path.
Users who want the static-lattice baseline write a config with
`fph_sigma = false` (σ=0 sentinel) and `frozen_phonons = 1`. Result is
one snapshot with zero displacement, byte-equivalent to today's
`add_potential(surf)` output.

Pros: smallest worker; aligned with V.'s "maybe add a flag" intent;
single coherent mental model.
Cons: the BF/no-BF detector asymmetry is no longer hardcoded — must be
reconstructed via config.

**Option C — drop the static-lattice output entirely.** Like B, but no
mechanism to recover it.

Pros: simplest of all.
Cons: removes a feature. Static-lattice diffraction / CBED would go
away unless the user explicitly configures σ=0.

#### Chosen: Option B

The worker treats all seeds uniformly. To reproduce today's dual
output, the user writes two configs (or sweeps on `fph_sigma`).

#### How the BF-asymmetry is reconstructed

Add a per-config detector list to `[microscope]`:

```toml
[microscope]
# Which detectors to compute. Defaults to all three. Worker only runs
# probe.scan() with the listed detectors.
detectors = ["haadf", "abf", "bf"]   # static-lattice run
# or
detectors = ["haadf", "abf"]          # phonon-averaged run
```

The "fph skips BF" behavior becomes one line in the TOML, per sweep.
Two configs reproduce today's output structure exactly:

```toml
# static.toml
[simulations]
fph_sigma = false
frozen_phonons = 1
[microscope]
detectors = ["haadf", "abf", "bf"]
```

```toml
# fph.toml
[simulations]
fph_sigma = 0.1
frozen_phonons = 8
[microscope]
detectors = ["haadf", "abf"]
```

Two invocations of `abtem-run` give identical filenames + content to
the current single-call dual output.

#### Why this is an improvement, not a regression

1. **The asymmetry becomes auditable.** Today it's hidden in
   `simulation_run`; now it's stated in the TOML next to the detector
   geometry.
2. **The asymmetry becomes overridable.** Want BF in a phonon-averaged
   run? Change one line in the config. No code edit needed.
3. **The asymmetry becomes per-config.** Different jobs can use
   different detector sets without forking the codebase.

#### Caveat — verify

`abtem.probe.scan(detectors=[...])` is assumed to accept any subset of
{HAADF, ABF, BF}. The API takes a list of `AnnularDetector` objects, so
any count should work — but a 1-line smoke test before promising the
feature is wise.

### 4. Diffraction and CBED are per-seed, gated independently

Today the diffraction / CBED block is gated by a single
`microscope.do_diffraction` flag and produces **two** outputs each (one
for the static lattice, one for the FrozenPhonons stack):

```python
if ctx.do_diffraction:
    plot_diffraction(ctx, entry['potential'],     ..., '_single_diff.png')   # static
    plot_diffraction(ctx, entry['fph_potential'], ..., '_fph_diff.png')      # FPH
    plot_cbed(ctx,        entry['potential'],     ..., '_center_cbed.png')   # static
    plot_cbed(ctx,        entry['fph_potential'], ..., '_center_fph_cbed.png') # FPH
```

Three placement options were considered for the worker model:

- **Option I — per-seed diffraction + CBED, aggregator means.** Each
  worker run does up to three multislices for its seed: the scan, an
  optional `PlaneWave.multislice` for diffraction, and an optional
  `Probe.multislice` at a single position for CBED. Outputs land in
  `outputs/seed_NNNNNN_{diff,cbed,…}.tif`. Aggregator means them into
  `aggregate/`.
- **Option II — diffraction once, at the aggregator (static-only).**
  Workers do only the scan. Diffraction and CBED run once per
  `(phase, hkl, tilt)`, on the bare lamella with no displacement. No
  phonon-averaged diffraction output — regression from current
  behavior.
- **Option III — separate `abtem-run-diffraction` command.**
  Diffraction lives in its own worker entry. The convenience wrapper
  has to know to call it.

#### Chosen: Option I

Per-seed diffraction and CBED, aggregator means them. Uniform with the
scan-output pattern: each seed produces its share, aggregator merges.
The "static-lattice diffraction" diagnostic is reproducible by config
choice (`fph_sigma = false`, `frozen_phonons = 1`) — symmetric with the
scan output.

The compute story is essentially unchanged from today. Today's
`PlaneWave.multislice(fph_potential).compute()` call already runs N
multislices internally — abtem iterates over the FrozenPhonons ensemble
and produces N exit waves, then `mean(axis=0)` averages them. Per-seed
just makes that explicit and parallelizable. No new cost; better
distribution.

#### `do_diffraction` and `do_cbed` are split, both default to False

The current single `do_diffraction` flag gates both plane-wave
diffraction and CBED. The worker model splits them:

```toml
[microscope]
do_diffraction = false   # plane-wave -> exit-wave -> far-field pattern
do_cbed        = false   # probe at center -> exit-wave -> far-field pattern
```

Both default to `false` so users opt in. Today's default is
`do_diffraction = true`, which combined with N-seed FPH means N× extra
PlaneWave multislices per run. That's a non-trivial silent cost. The
worker era is the right time to make it explicit.

Splitting the two also lets users skip the per-position CBED
multislice (which is cheap, but distinct) while keeping the plane-wave
diffraction, or vice versa.

#### Output naming

```
outputs/
├── seed_000000_haadf.zarr      (always — if scan is configured)
├── seed_000000_abf.zarr
├── seed_000000_bf.zarr
├── seed_000000_diff.tif        (if do_diffraction)
├── seed_000000_cbed.tif        (if do_cbed)
├── seed_000001_haadf.zarr
└── …

aggregate/
├── haadf.zarr                  (mean over seeds)
├── abf.zarr
├── bf.zarr
├── diff.tif                    (if do_diffraction was set)
└── cbed.tif                    (if do_cbed was set)
```

The legacy `single_diff.png` / `fph_diff.png` / `center_cbed.png` /
`center_fph_cbed.png` naming disappears in the worker outputs. (The
legacy in-process `pipeline.py` path still emits the PNG previews.)

#### Open verification

Per-seed averaging via `mean over saved TIFFs` is mathematically the
same as `mean(axis=0)` over the abtem ensemble — but abtem might
normalize inside the ensemble mean (e.g., for intensity-conserving
diffraction patterns). One-line smoke test before fully committing:
run the same FPH config twice — once through the legacy
`plot_diffraction` path, once via N per-seed runs averaged by hand —
and confirm bit-equality (or at least floating-point closeness) of the
final pattern.

### 5. `plot_dataset` is split by cost: geometry at generator time, projection at aggregator time

Today's `plot_dataset` is called from the middle of `simulation_run`
and produces several files per `(phase, hkl, tilt)`, which fall into
two groups by what they show and what they cost:

- **Planning sanity check** — "did the lamella build correctly? is
  the scan box in the right place?" Answered by the atom-geometry
  view. You want this *before* you spend GPU time.
- **Result visualization** — "what does the projection look like?"
  Answered by the potential views. You want this alongside or after
  the scan results, mostly for diagnostics or figures.

#### Chosen: split by cost. Geometry → generator, projection → aggregator.

| What | Where | When | Why |
|---|---|---|---|
| Atom geometry (`combined.png`, `surf.xyz`) | generator | at planning time, before any worker runs | cheap, no GPU; lets you abort early if the lamella is wrong |
| Static-lattice potential projection (`potential_projection.{png,tif}`, `potential_projection_scanned.tif`) | aggregator | once per job, after workers finish | one `Potential.project().compute()` per job; reasonable scope for the aggregator since it's already the "produce final-form outputs" step |
| FPH-projection variants | — | — | **dropped unconditionally**: under Decision #3 (single dropped, reconstruct via σ=0), the static-vs-FPH distinction collapses for previews same as for the scan output |

Output layout becomes:

```
gen_<UTC>/<phase>_<hkl>_<tilt>/
├── <stem>.toml
├── surf.xyz                                 (emitted by generator)
├── combined.png                             (emitted by generator)
├── seeds/seed_NNNNNN.todo …
├── outputs/seed_NNNNNN_{haadf,abf,bf,diff,cbed}.{zarr,tif}
└── aggregate/
    ├── {haadf,abf,bf}.zarr                  (mean over seeds)
    ├── {haadf,abf,bf}.tif                   (with gaussian blurs)
    ├── potential_projection.png             (side-by-side projection + probe shape)
    ├── potential_projection.tif             (raw)
    ├── potential_projection_scanned.tif     (cropped to scan area)
    └── diff.tif, cbed.tif                   (if do_diffraction / do_cbed; per Decision #4)
```

#### Implications

1. **The generator becomes lamella-aware.** It imports
   `simulation.make_lamella`, builds the atoms per `(phase, hkl,
   tilt)`, and emits `combined.png` + `surf.xyz` alongside the `.toml`
   and `seeds/`. The generator still has **no GPU dependency**:
   `make_lamella` is dask-y but CPU-OK; `abtem.show_atoms` is a
   matplotlib helper. The "plan on a cheap CPU box, run workers on a
   GPU box" deployment shape is preserved.

2. **The aggregator gets one compute step.** It runs once per job
   after workers finish, adding a small once-per-job
   `Potential.project().compute()` for the projection preview.

3. **FPH-projection previews disappear.** Under Decision #3, "static
   vs FPH" is a config choice (σ=0 vs σ>0), not a code path.

4. **No new `abtem-run-preview` command.** The previews live next to
   where they're naturally useful — geometry at planning, potential at
   result time.

### 6. `test_enabled` flag — keep per-seed scratch + dump displaced atoms

New schema field: `simulations.test_enabled: bool = False` (default
`false`).

The flag has two effects, both gated on it being `true`:

| Effect | Component responsible | What it produces |
|---|---|---|
| Skip cleanup of `outputs/` | aggregator | per-seed `outputs/seed_NNNNNN_*.{zarr,tif}` stay on disk after `aggregate/` is produced |
| Dump per-seed displaced atoms | worker | `outputs/seed_NNNNNN_displaced.xyz` — the post-displacement atom positions for that snapshot |

The displaced `.xyz` is V.'s "checkpoint 2 explicit on disk" — today
the displaced atoms exist only in memory inside the FrozenPhonons
ensemble. With the flag on, every snapshot's atom positions become an
inspectable artifact.

#### What the three checkpoints look like on disk

Under `test_enabled = true`, every checkpoint is materialized:

| Checkpoint | What | Where |
|---|---|---|
| 1: static atomic coordinates | the bare lamella, no displacement | `surf.xyz` (generator-stage) |
| 2: per-phonon displacements | one displaced atom set per seed | `outputs/seed_NNNNNN_displaced.xyz` (worker, gated) |
| 3: scattering output | per-seed scan / diffraction / CBED | `outputs/seed_NNNNNN_*.{zarr,tif}` (worker, always) |
| (aggregated 3) | mean over snapshots | `aggregate/*.{zarr,tif,png}` (aggregator, always) |

Under `test_enabled = false`, only checkpoint 1 and the aggregated
output persist; everything per-seed is scratch.

#### Why per-seed Potential is NOT dumped

Was considered. Rejected because the `.array` is multi-megabyte at our
0.05 Å sampling (N seeds = hundreds of MB of scratch), and it's fully
reconstructable from `seed_NNNNNN_displaced.xyz` + the job config.

#### `test_enabled` is NOT for production runs

This is an explicit diagnostic switch. Production runs leave it
`false` and let the aggregator clean up.

## See also

- `README.md` — three-checkpoint architecture overview.
- `src/abtem_run/generator_run.py` — the producer side.
- `src/abtem_run/worker.py` / `src/abtem_run/aggregate.py` — the
  consumer + merge sides.
