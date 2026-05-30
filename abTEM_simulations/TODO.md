# Post-PR #6 TODO

Items captured during the PR #6 review (worker architecture + projection
consolidation + legacy retirement) for landing after the PR closes. Kept
here as a checklist so they don't fall out of the review thread.

## Chunk (h) — versioned aggregates

Each aggregate run gets a unique identifier; the aggregator rediscovers
the newest existing one by default and only creates a new one if asked.

- [ ] `aggregate_job` writes to `aggregate/<UTC>_<hash>/` per run, not flat `aggregate/`
- [ ] Hash = stable hash of `ctx` / `cfg.model_dump()` JSON, SHA256 truncated to 8 chars; UTC is the dir-name prefix for display/sort, NOT mixed into the hash
- [ ] Rediscovery: scan `aggregate/*_<hash>/`, reuse the one with the latest `stat().st_mtime` (no-op) unless `--force-new`
- [ ] CLI: `--force-new` flag (+ lib `force_new=True` kwarg) to skip rediscovery
- [ ] `aggregate_series`'s `n_<k>/` → filename collapse is out of scope for (h); defer to a later pass

## Hardening + namespacing

- [ ] **Per-channel seed counter on aggregates.** `_emit_channel` returning `None` enables partial-preview-in-parallel but masks incompleteness if the caller doesn't track which channels finished. Surface "N seeds averaged into this channel" — filename suffix, sidecar metadata, or zarr attr (TBD).
- [ ] **Aggregate output namespacing.** `aggregate/` is currently a flat soup of stems (`potential_projection*`, `diff*`, `cbed*`, scan dets, `_static` variants, `_scanned.tif`). Foreseen clashes as outputs grow, and projection + diffraction land in the same dir. Audit; consider subdirs (`aggregate/projections/`, `aggregate/scans/`, `aggregate/patterns/`) or stricter prefixing. Interacts with chunk (h).
- [ ] **Pre-existing-tree rerun audit.** Behavior under partially-populated `gen_<UTC>/`, `outputs/`, `outputs_archive/`, `aggregate/` is undefined; new-seed rerun likely merges silently with old. Walk the matrix, document expected behavior, add guards.
- [ ] **Cache `static_potential` to disk.** `aggregate_job` rebuilds the static ground-state `Potential` every invocation (for the projection preview + `emit_static_baseline` scan). Save it binary once per job (e.g. `static_potential.zarr` in the versioned aggregate dir); reload on subsequent runs.
- [ ] **Safer `outputs/` cleanup.** `shutil.rmtree(out_dir)` blows away whatever path-resolution edges happen to point at, including unrelated user dirs named `output(s)`. Add guards (confirm path is exactly `job_dir/"outputs"`, no symlink, contains only the expected seed-named files) or default to no-auto-delete (opt-in flag). Coordinate with chunk-(c)'s archive-on-aggregate which moves rather than deletes.
