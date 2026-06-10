# Post-PR #6 TODO

Items captured during the PR #6 review (worker architecture + projection
consolidation + legacy retirement) for landing after the PR closes. Kept
here as a checklist so they don't fall out of the review thread.

> **Reconciled 2026-06-10** against the merged tree (`6a5d540`). The
> `surf.xyz`-routing section and two hardening items (`static_potential`
> cache, safer `outputs/` cleanup) have landed and are checked off below.
> Chunk (h) versioned aggregates, the per-channel seed counter, output
> namespacing, and the pre-existing-tree rerun audit are still open.

## Chunk (h) — versioned aggregates

Each aggregate run gets a unique identifier; the aggregator rediscovers
the newest existing one by default and only creates a new one if asked.

- [ ] `aggregate_job` writes to `aggregate/<UTC>_<hash>/` per run, not flat `aggregate/`
- [ ] Hash = stable hash of `ctx` / `cfg.model_dump()` JSON, SHA256 truncated to 8 chars; UTC is the dir-name prefix for display/sort, NOT mixed into the hash
- [ ] Rediscovery: scan `aggregate/*_<hash>/`, reuse the one with the latest `stat().st_mtime` (no-op) unless `--force-new`
- [ ] CLI: `--force-new` flag (+ lib `force_new=True` kwarg) to skip rediscovery
- [ ] `aggregate_series`'s `n_<k>/` → filename collapse is out of scope for (h); defer to a later pass

## Build ground state once; route worker + aggregator through `surf.xyz` — ✅ landed

Landed in `cd07604` (generator writes `surf.xyz` as extxyz so the cell box
survives), `2699ed6` (worker + aggregator read it), `c136133` (Potential for
the probe grid-match), `ac8844d` (static_potential disk cache). The single
entry point is `simulation.load_ground_state_atoms`; planning, worker, and
aggregator all go through it, so planning's atoms and runtime's atoms can no
longer diverge.

- [x] Worker reads `surf.xyz` instead of `build_lamella_from_config` — `worker.run_one_seed` → `load_ground_state_atoms`; the phonon displacement applies to those atoms
- [x] Aggregator reads `surf.xyz` for the projection preview — `aggregate._load_or_build_static_potential` → `load_ground_state_atoms`
- [x] Failure mode **decided: fall back to a fresh build with a loud `warnings.warn`** (not refuse) when `surf.xyz` is missing/unreadable — `simulation.load_ground_state_atoms`
- [x] Silent-drift bug class removed — one loader shared by all three stages
- [x] `static_potential` cache landed (`ac8844d`, mtime-gated vs `surf.xyz`, atomic publish); chunk (h) versioned aggregates is the remaining "build once, reuse" rung and is still open

## Hardening + namespacing

- [ ] **Per-channel seed counter on aggregates.** `_emit_channel` returning `None` enables partial-preview-in-parallel but masks incompleteness if the caller doesn't track which channels finished. Surface "N seeds averaged into this channel" — filename suffix, sidecar metadata, or zarr attr (TBD).
- [ ] **Aggregate output namespacing.** `aggregate/` is currently a flat soup of stems (`potential_projection*`, `diff*`, `cbed*`, scan dets, `_static` variants, `_scanned.tif`). Foreseen clashes as outputs grow, and projection + diffraction land in the same dir. Audit; consider subdirs (`aggregate/projections/`, `aggregate/scans/`, `aggregate/patterns/`) or stricter prefixing. Interacts with chunk (h).
- [ ] **Pre-existing-tree rerun audit.** Behavior under partially-populated `gen_<UTC>/`, `outputs/`, `outputs_archive/`, `aggregate/` is undefined; new-seed rerun likely merges silently with old. Walk the matrix, document expected behavior, add guards.
- [x] **Cache `static_potential` to disk.** ✅ landed `ac8844d` — `aggregate._load_or_build_static_potential` saves `job_dir/static_potential.zarr`, cache-hits when it is at least as new as `surf.xyz`, and atomic-publishes via a `.tmp` rename. Lives at `job_dir` (not the aggregate dir) so it survives cleanup and is never matched by the `seed_*` averaging glob.
- [x] **Safer `outputs/` cleanup.** ✅ archive-on-aggregate landed in `84e0b67` (chunk (c)); `768531c` added the path-edge guards. `aggregate._archive_per_seed_outputs` *moves* every child of `outputs/` into `outputs_archive/` (it never `rmtree`s `outputs/` itself — only `rmdir`s the emptied dir) and refuses unless `out_dir` is a non-symlink real directory named exactly `outputs` sibling to `outputs_archive`; the worker's `_cleanup_seed_outputs` guards `is_dir`/symlink on the SIGTERM/SIGINT path too. Residual: no `seed_*`-only filter, so a stray non-seed file in `outputs/` is moved into the archive too (preserved, not deleted) — which is why the original "rmtree blows away arbitrary paths" risk can't occur.

## Dockerfile / spot-deployment review (2026-06-10)

Findings from a critical read of `Dockerfile` during the first real GPU
validation (RTX A4500 host; the GPU test pack §1–§4 all passed, so the
runtime semantics the image wraps are proven). The image *design* is sound —
`runtime` base (includes cuFFT, needed for `fft='cufft'`), selective COPY,
no-ENTRYPOINT one-shot containers, `.todo`-retry + SIGTERM partial-cleanup
spot story. These are the gaps, priority-ordered.

Scope note: there is no actual AWS infra in the repo (no ECS/Batch defs, S3
sync, or spot-interruption watcher). "AWS setup" = this generic spot-friendly
image; fleet, queue, work assignment (push model — the orchestrator must not
double-assign a `.todo`), driver/AMI choice, and result sync are all on the
operator. The header should say so explicitly.

### High — fails in real use

- [ ] **`abtem-run-generate` doesn't exist.** Referenced in the header's generator example and the CMD comment (and in `cli.py:11`'s docstring), but `[project.scripts]` defines no such entry. The fallback `python -m abtem_run.generator_run` takes *no arguments* (hardcodes `./config.toml`), so the documented `/job/config.toml` arg has no working target at all. Fix the examples to `abtem-run --generate-only --config /job/config.toml`, or add the console script (+ argparse in `generator_run.__main__`).
- [ ] **`tifffile` missing from the image.** abtem 1.0.9 declares it *optional* (lazy import in `array.py`, RuntimeError at `to_tiff`), and neither abtem-run's pyproject nor the Dockerfile installs it. The worker writes `seed_*_potproj.tif` unconditionally and *before* the zarr, so the first seed dies at output-write **after** paying for the multislice — worst case on spot billing. Masked on dev hosts where tifffile happens to be present. Right fix: add `tifffile` to `pyproject.toml` `dependencies` (hard runtime dep of worker + aggregator), which fixes the image for free.
- [ ] **No build-time patch gate.** The image exists to pin abtem for the three source-substring patches, yet nothing verifies they bind. Add after the install (imports fine without a GPU — patching only rewrites source):
  `RUN python3.11 -c "import abtem_run,sys; sys.exit(0 if all(abtem_run._PATCHES_APPLIED.values()) else 1)"`
  And pin `abtem==1.0.9` exactly — `~=1.0.9` permits a 1.0.10 where patches silently no-op and the failure surfaces as a runtime cupy crash on a spot node.

### Medium

- [ ] **Jammy's `python3.11` is the `3.11.0~rc1` universe build** — the image pins a release-candidate interpreter. `nvidia/cuda:13.0.0-runtime-ubuntu24.04` exists (tag verified on Docker Hub 2026-06-10) with a proper 3.12 default, allowed by `requires-python >=3.11`; caveat: 24.04 enforces PEP 668 → venv or `--break-system-packages`. Either way add a `python3.11 --version` / interpreter sanity check at build.
- [ ] **CUDA 13 needs an r580+ host driver** — many AWS GPU AMIs still ship r570-era drivers for CUDA 12. Document the driver floor next to the `--gpus all` example, or parameterize the file (`ARG BASE_TAG` + `ARG GPU_EXTRA`) so a `gpu-cu12` variant builds from the same Dockerfile (pyproject extras already exist).
- [ ] **Runs as root** → outputs on the mounted `/job` tree come back root-owned, breaking host-side follow-ups (extend, aggregate, rsync). Worker only writes inside the job tree, so documenting `--user "$(id -u):$(id -g)"` in the run examples suffices.
- [ ] **PID-1 signal gap.** Python is PID 1; outside the `_install_preemption_handler` window (config/lamella load; the post-`restore_handlers()`-pre-rename gap) SIGTERM has default disposition and PID 1 *ignores* it → container hangs to the grace-period SIGKILL. Consequence is mild (a completed-but-unrenamed seed recomputes), but `--init` in the run examples fixes the semantics outright.
- [ ] **Layer-cache inversion:** source is COPY'd before `pip install`, so any code edit re-resolves the whole cupy/scipy/abtem stack. Split: pinned third-party deps as their own layer, then COPY source + `pip install --no-deps .`. Also trims ECR push churn / spot cold-pull deltas.

### Minor

- [ ] WORKDIR comment says "/work is the mount point" while every example mounts `/job` (harmless — abs paths — but confusing)
- [ ] `python3.11-venv` installed, never used
- [ ] no `.dockerignore` — context ships `.git` + the unrelated tool dirs
- [ ] `DEBIAN_FRONTEND` as persistent ENV instead of per-RUN
- [ ] no OCI source/revision labels (provenance in ECR)
- [ ] `PIP_NO_CACHE_DIR=1` + `--no-cache-dir` redundant

## Diffraction/CBED finite-box fringes (2026-06-10)

Surfaced during the first GPU validation (a domain reviewer flagged the
production single-seed diffraction/CBED as "странный вид рассеяния" — odd-looking
scattering). The raw plane-wave diffraction and CBED patterns (`run_diffraction`
/ `run_cbed`) carry **concentric rings that are a finite-simulation-box artifact,
not crystal scattering**. NOT GPU-related (CPU↔GPU parity holds), NOT the crystal
lattice (the discrete Bragg spots are fine), NOT abtem's post-processing.
Confirmed by two cheap tests on static TaTe2 [010]:

- **Boundary sweep** (`borders` 4/10/20 Å → box 26/50/90 Å): ring angular spacing
  scales ~1/box (3.9 → 1.7 → 1.4 mrad), i.e. **constant in pixels (~4 px), not
  fixed in mrad**. Real Bragg/HOLZ rings would be box-invariant.
- **Mechanism test**: the rings are present in the raw `|FFT(exit_wave)|^2`, at
  the same mrad as `diffraction_patterns('valid', block_direct=True)` — so they
  live in the **multislice exit wave**, not in any resampling step.

Root cause: plane-wave illumination of a finite, laterally-cropped lamella in a
non-periodic box (`make_potential(periodic=False)`, `borders` = 4–5 Å). The finite
illuminated top-hat → shape-transform (sinc-like) fringes in the far field.

Impact + options:
- **Scan detectors (HAADF/ABF/BF) are ~unaffected** — annular angular integration
  averages over the fringe periods. The artifact is specific to the raw
  diffraction/CBED *pattern* output.
- Present in production (`borders=5` → box ~70 Å → fine fringes). Phonon averaging
  lowers their *visual* contrast (adds diffuse, smooths coherent speckle) but does
  NOT remove them — they are box-locked, config-independent.
- [ ] If clean diffraction/CBED patterns matter: enlarge the vacuum/box for the
      pattern path (fringes → finer ∝ 1/box, eventually negligible; cost = bigger
      grid), or use a dedicated large/`periodic=True` cell for pattern output
      (infinite crystal → pure Bragg, no shape transform — only correct if that's
      what the pattern should represent).
- [ ] At minimum, document that raw diffraction/CBED previews carry a finite-box
      fringe and shouldn't be over-read at low-to-mid angle.

## dask_cuda distributed path — validated + gaps (2026-06-10)

First real exercise of `gpu_related.dask_cuda = true` (GPU host; dask_cuda
26.06.00 emerged, `rmm` absent → made a soft-dep, separate commit). The path
works end-to-end: `dask-scheduler` + `dask-cuda-worker` on `:8786`, abtem's
multislice graph genuinely distributes to the CUDA worker (confirmed by the
`distributed` "Sending large graph" warning — not a local bypass), output
**bit-identical** to the single-GPU baseline (haadf/abf/bf 0.00e+00), clean
teardown. Two gaps + a pre-existing smell:

- [ ] **Workers MUST preload `abtem_run`.** The `dask_cuda=true` branch creates
      the `Client` but does nothing to ensure remote workers import `abtem_run`,
      so the monkey-patches don't apply worker-side — and `_fft_dispatch_cufft_numpy`
      executes ON the worker, so an unpatched worker would crash the cufft path.
      The validation only passed because the worker was launched with
      `dask-cuda-worker … --preload abtem_run`. A real deploy needs that flag, OR
      the pipeline should `client.register_worker_plugin(...)` (or a `WorkerPlugin`)
      that imports `abtem_run` on every worker. Currently undocumented + unhandled.
- [ ] **33 MiB graph shipped per submit.** dask warns abtem re-sends the whole
      potential/graph to the worker each call ("Sending large graph of size
      33 MiB"). Correct, but for multi-GPU it's an efficiency hit — `scatter` /
      persist the potential once instead of re-shipping per task.
- [ ] **Pre-existing** (`pipeline.py:169` `# !TODO - separate dask.distributed and
      dask_cuda`): the Client connection, the now-soft rmm allocator, and worker
      setup are yoked to one flag. The hardcoded `tcp://127.0.0.1:8786` with **no
      connection timeout** also hangs silently if no scheduler is up — give it a
      timeout and a clearer error.
