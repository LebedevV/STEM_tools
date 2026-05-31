# abtem-run

STEM/TEM image simulation pipeline built on top of [`abtem`](https://github.com/abTEM/abTEM)
and [`ase`](https://gitlab.com/ase/ase).

What the code does:

- imports crystallographic data from a given CIF file
- expands the lattice into a "superblock" of atoms
- rotates the superblock so a chosen `uvw` (or the normal to `hkl`) points along Z
- crops the rotated block to the requested lamella dimensions
- runs the configured abTEM simulations — diffraction patterns, CBED, and
  HAADF / ABF / BF scans — with optional frozen-phonon averaging
- aggregates per-phonon outputs into mean images and projected-potential previews

The main benefit is the ability to handle non-orthogonal space groups.
**Trigonal symmetry has not been thoroughly assessed** — feedback welcome.

Outputs are written under `paths.folder_sim + paths.extr` (resolved relative
to the config file): TIFFs, PNG previews, Zarr arrays, and per-run TOML
config dumps capturing the exact parameters that produced each output.

## Status

`v0.1.0` alpha. Pip-installable; runs end-to-end on the original author's setup.
The runtime monkey-patches that adapt abtem internals are **conditional** —
they detect whether they're needed for the installed abtem build and no-op
otherwise (see `abtem_run._PATCHES_APPLIED` for which ones applied). The
abtem version is pinned to `~=1.0.9` because the monkey-patches target
specific internal substrings of that minor; bump explicitly and re-test
before relaxing.

## Install (development)

```bash
git clone https://github.com/LebedevV/STEM_tools.git
cd STEM_tools/abTEM_simulations
python -m venv venv
./venv/bin/pip install -e ".[dev]"          # CPU only + pytest + ruff + mypy
./venv/bin/pip install -e ".[dev,gpu-cu13]" # GPU (CUDA 13.x — see below)
```

GPU extras are CUDA-major-version-specific because `cupy` ships separate
wheels for each CUDA major. Pick the extra matching your runtime — check
`nvidia-smi` for the CUDA version string:

| Extra | What it installs | When to use |
|---|---|---|
| `gpu-cu11` | `cupy-cuda11x`, `dask-cuda`, `rmm-cu11` | CUDA 11.x |
| `gpu-cu12` | `cupy-cuda12x`, `dask-cuda`, `rmm-cu12` | CUDA 12.x |
| `gpu-cu13` | `cupy-cuda13x`, `dask-cuda` (no rmm wheel yet) | CUDA 13.x |

`rmm` is only used via the optional `dask-cuda` allocator path
(`gpu_related.dask_cuda = true`); the pipeline runs fine without it,
so `gpu-cu13` is functional even without a CUDA-13 rmm wheel.

## Run

Two entry points: the convenience wrapper (one process, all phases) or
the worker-era CLI (separate generator / worker / aggregator for slurm /
GNU parallel / multi-GPU).

### Convenience wrapper

```bash
cd abTEM_simulations          # CWD must contain config.toml (or pass --config)
./venv/bin/abtem-run          # generator -> workers -> aggregator, one process
# or equivalently:
./venv/bin/python -m abtem_run
```

`abtem-run` reads `config.toml` from the current working directory.
`--generate-only` stops after the generator stage so you can inspect the
planning artifacts (`surf.xyz`, `combined.png`) without burning GPU time.
`--resume <run_dir>` picks up any remaining `.todo` seeds in an existing
generator output directory and aggregates each job — idempotent on
already-complete sweeps.

### Worker-era CLIs (parallel / cluster)

```bash
abtem-run-generate <config.toml>          # plans the job tree + seed queue
abtem-run-worker <job_dir> <todo_path>    # consume one seed, write outputs
abtem-run-aggregate <job_dir>             # merge per-seed outputs once all are .done
abtem-run-extend <job_dir> --add N        # add N more phonon snapshots to an existing job
abtem-run-extend <job_dir> --seeds 23,24,25   # ... or specify the seed integers explicitly
abtem-run-to-ensemble <job_dir>           # bundle per-seed zarrs into one abtem-native ensemble
```

This split lets you fan out `abtem-run-worker` across nodes / GPUs and
run `abtem-run-aggregate` only once the job's `.todo` queue is empty.

### Extending a finished job with more phonon snapshots

After `abtem-run-aggregate` runs, the per-seed `outputs/seed_*_<channel>.zarr`
files move into `outputs_archive/` (instead of being deleted). When you
later realize you want more statistics, `abtem-run-extend` emits new
`.todo` files with non-overlapping seed integers (`--add N` picks them
starting just past the current max; `--seeds A,B,C` accepts them
explicitly). A follow-up `abtem-run-worker` + `abtem-run-aggregate` pass
produces the cumulative mean over the union of the archive and the new
batch — no need to redo the original snapshots.

Each extend call appends a record to `extensions.json` in the job dir
(timestamp, added seeds, source flag). `abtem-run-extend` refuses to
operate on a job that has no prior batch (nothing to extend).

### Re-aggregating without re-running workers

If the per-seed zarrs are already on disk (in `outputs/` or
`outputs_archive/`), `abtem-run --aggregate <job_dir>` re-runs the
aggregator alone — same effect as `abtem-run-aggregate <job_dir>`. Use
this to apply new `blur_sigmas` / `blur_boundary` settings, or to
re-emit the projection preview after editing the per-job TOML, without
re-running the multislice.

`abtem-run --aggregate-series <job_dir> [--n-phonons N]` emits a
cumulative-mean frame series: for k in `1..N` (defaulting to all
available seeds), it writes `<job_dir>/aggregate/n_<k:03d>/` containing
the per-channel mean of the first k seeds plus the configured blur
variants. Useful for visualising the 1/√N convergence on real data.
The static-block artifacts (projection preview, optional static
baseline scan) are emitted ONCE at `<job_dir>/aggregate/`, not per-k —
they don't depend on phonon count.

## Configuration

All runtime parameters live in a TOML config file, validated by Pydantic
models in `src/abtem_run/config.py`. Paths in `[paths]` are resolved
relative to the config file's directory (not the working directory), so
configs can travel with their referenced CIFs and output directories.

`abTEM_simulations/config.toml` is the annotated reference; the rest of
this section calls out the less-obvious knobs.

### Parameter sweeps

Several scalar fields accept lists; the pipeline iterates the Cartesian
product of all such fields. Sweepable: `frozen_phonons`, `fph_sigma`,
`thickness`, `global_tilt_a`, `global_tilt_b`, `probability_of_vac`,
`HT_value`. Each combination becomes a separate job directory.

### `[microscope]` probe optics

- **`defocus`** — accepts a number (Å) or the literal string `'scherzer'`
  to ask abtem to compute Scherzer defocus from `aberrations.C30` and the
  beam energy. **Footgun:** `'scherzer'` is a silent no-op when `C30 = 0`
  (the formula evaluates to zero), which on thin crystals produces an
  inverted-center "BF looks like dark field" pattern in the BF channel.
  The pipeline emits a runtime warning when this combination is set. Use
  either an explicit numeric defocus, or set `aberrations.C30` to a
  non-zero value (typical uncorrected 200 kV: `1.0e7` Å = 1 mm;
  aberration-corrected: `1.0e4` Å = 1 μm).
- **`aberrations`** — dict of abtem phase-aberration coefficients, all in
  Å (radians for the `phi*` angular terms). Common keys: `C30` (spherical
  aberration / Cs), `C50`, `C12` (twofold astigmatism) + `phi12`, `C32`,
  etc. `defocus` and `C10` are rejected here — use the dedicated
  `defocus` field above. Keys are validated against abtem's actual
  supported set (`abtem.transfer.polar_aliases`), so the validator stays
  in sync with upstream automatically.
- **`detectors`** — subset of `{haadf, abf, bf}` that the worker should
  compute. Defaults to all three.

### `[simulations]`

- **`blur_boundary`** — boundary mode for the post-aggregation
  gaussian-blur preview TIFFs. Default `'nearest'` (pre-2026-05 behavior
  used `'constant'` which produced dark halos at lamella edges).
- **`blur_sigmas`** — list of sigma values (in real-space sampling units)
  for the post-aggregation gaussian-blur previews. One blurred TIFF per
  sigma per channel: `aggregate/<channel>_<sigma>.tif`. Set to `[]` to
  skip blur previews entirely.
- **`emit_static_baseline`** — when `true`, the aggregator also writes a
  static-lattice projected-potential preview
  (`aggregate/potential_projection_static.{png,tif}`) alongside the
  phonon-averaged projection. Cheap — reuses the ground-state potential
  the aggregator already builds for the probe-shape side panel. For a
  static-lattice **scan**, run a separate job with
  `frozen_phonons = "None"`; the aggregator itself does no scan
  multislice.
- **`test_enabled`** — the aggregator preserves per-seed `outputs/`
  instead of archiving, and the worker writes
  `seed_NNNNNN_displaced.xyz` for inspection.

### `[job]`

- **`phase`** — CIF filename (resolved against `paths.folder`). Accepts
  either a single string (`"TaTe2_2310358.cif"`) or a list
  (`["TaTe2_2310358.cif", "Pm3m.cif"]`). With a list, the generator
  emits one job directory per (phase, hkl, tilt) combination with
  locked seed integers across phases so phase-to-phase comparisons
  use identical RNG draws.
- **`hkl_to_do`** — single `[h,k,l]` or list of triples.
- **`is_uvw`** — if true, the vector is a real-space direction; if false,
  the normal to that plane is aligned with Z.
- **`phonons_seed`** — RNG seed for `FrozenPhonons`. Same seed
  reproduces the same displacement bit-for-bit.
- **`inplane_angle`** — extra in-plane rotation in degrees, applied
  after the out-of-plane rotation R. Accepts a number or the string
  `'auto'` (auto-detect from `atom_to_zero`).
- **`inplane_align_hkl` / `inplane_align_axis`** — "this hkl up"
  alignment. If `inplane_align_hkl` is set, the in-plane angle is
  computed so that the projection of that Miller direction (after R is
  applied) lands on the chosen lab axis (`'x'` or `'y'`, default `'y'`).
  Overrides `inplane_angle` and the atom-to-zero auto path. Handles
  trigonal/hexagonal/triclinic cells via diffpy's metric tensor.

### `[lamella_settings]`

- **`vacancies_seed`** — RNG seed for `add_vacancies`. Separate from
  `phonons_seed` because vacancies are checkpoint-1 (atomic coords) and
  phonons are checkpoint-2 (displacements).
- **`tilt_degrees`** — if `false`, reinterprets `global_tilt_a/b` as
  **mrad** instead of degrees.

## Architecture

The pipeline naturally decomposes into three checkpoints, with
randomness only in the second and third:

1. **Atomic coordinates** `(x0, y0, z0)` — deterministic, derived from
   the TOML and the CIF. The generator writes this to `surf.xyz`
   (extxyz format, preserving the cell box) once per job.
2. **Per-phonon displacements** `{xᵢ, yᵢ, zᵢ}` — random but **strictly
   reproducible** (same seed → bit-identical displacements).
3. **Scattering output** — abTEM's internal multislice. Monte-Carlo-style
   randomness, intentionally not bit-reproducible.

Plus minor post-processing (cross-seed averaging, gaussian-blurred TIFF
variants).

The worker pipeline implements this directly:

- `generator_run.generate_run(config)` — reads the TOML, expands the
  parameter sweep, and emits one job directory per `(phase, hkl, tilt)`
  with a `seeds/seed_NNNNNN.todo` file per phonon snapshot. Also writes
  the ground-state `surf.xyz` (worker + aggregator's input) and a
  3-panel `combined.png` preview before any worker runs.
- `worker.run_one_seed(job_dir, todo_path)` — loads `surf.xyz`, applies
  `abtem.FrozenPhonons(num_configs=1, seed=K).trajectory[0]` (where K is
  the seed integer from the `.todo` filename), runs the configured
  multislice paths (scan / diffraction / CBED, each independently
  gated), writes per-seed outputs, and atomically renames `.todo` →
  `.done`. Stateless; safe to fan out across processes / GPUs.
- `aggregate.aggregate_job(job_dir)` — once all `.todo`s are `.done`,
  means the per-seed outputs into `aggregate/<channel>.{tif,zarr}` plus
  gaussian-blurred TIFF variants and a projected-potential preview
  (built from the same `surf.xyz` the worker used).

### Phonon seed convention

Workers draw independent RNG samples: each invocation runs
`FrozenPhonons(num_configs=1, sigmas=σ, seed=K)` where K is the integer
in the `seed_KKKKKK.todo` filename (range `[phonons_seed,
phonons_seed + N)`). The cross-seed mean estimates the thermal-average
observable; standard error decays as 1/√N. Setting `frozen_phonons` to
a positive int with `fph_sigma ≤ 0` is refused at worker time (it would
produce N identical frames). Use `frozen_phonons = "None"` for an
explicit single-seed baseline run with no displacement.

### Ground state file

`surf.xyz` is the worker's canonical input. The format is **extxyz** so
the lattice box survives the round-trip (plain `xyz` would drop it and
break the downstream Potential build). A user-supplied `surf.xyz` (e.g.
from an external relaxation) drops into the same slot — if the file is
present in the job dir, the worker uses it as-is rather than rebuilding
from the CIF.

## Tests and checks

```bash
pytest -q                                  # full test suite
ruff check src/ tests/                     # linter
mypy                                       # type-check src/abtem_run/
```

The suite covers FrozenPhonons reproducibility, `make_lamella`
determinism, the full pipeline integration (generator → workers →
aggregator), gaussian-blur boundary modes, the static-baseline flag,
`inplane_align_hkl` math + integration, the probe defocus/aberrations
machinery including the scherzer-with-zero-C30 warning, and the
cumulative-phonons workflow (`abtem-run-extend` + re-aggregate against
`outputs_archive/`).

mypy runs in a permissive mode (`ignore_missing_imports = true` for the
stub-less third-party deps). The baseline is clean; raise the bar
(`disallow_untyped_defs`, etc.) as annotation coverage grows.

No CI is wired yet — local `pytest -q` + `ruff` + `mypy` is the current
contract.

## abTEM-native ensemble (cross-compat bridge)

The pipeline writes per-seed `.zarr` files for resumability +
cumulative-extend support; abTEM's idiomatic on-disk representation is
a single Measurement zarr with an ensemble axis. `abtem-run-to-ensemble`
bridges the two: it reads `outputs/seed_*_<channel>.zarr` ∪
`outputs_archive/seed_*_<channel>.zarr` for each channel and writes
`aggregate/<channel>_ensemble.zarr` carrying the full N-snapshot stack
with `FrozenPhononsAxis(_ensemble_mean=True)` on the new axis.

```bash
abtem-run-to-ensemble <job_dir>                    # all channels
abtem-run-to-ensemble <job_dir> --channel haadf    # one channel only
```

The bridge is round-trip safe — `abtem.from_zarr(<channel>_ensemble.zarr)`
returns an abtem-native Measurement, and `.reduce_ensemble()` on it
yields the thermal average to floating-point precision against
`aggregate/<channel>.zarr` (the filesystem-glob mean from the regular
aggregator). Library entry: `from abtem_run import load_ensemble`.

## Acknowledgments

- **Julie M. Bekkevold** for invaluable help with ab-initio simulations
  and guidance with abTEM.
- Project **SFI/21/US/3785** for financial support.
- Worker-pipeline architecture, packaging, and iterative cleanup
  contributed by **Ivan S. Titov**.

## License

GPL-v3. See `LICENSE` at the repository root.
