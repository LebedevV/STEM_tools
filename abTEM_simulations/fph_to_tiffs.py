#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Frozen-phonon zarr -> ensemble-mean TIFF + one TIFF per source size.

Standalone helper for a batch processor: given the path to an fph (frozen-phonon)
measurement zarr (e.g. saved by the old abtem path), load it, average over the
WHOLE phonon ensemble, and write TIFFs the way abtem_run's aggregator does
(see abtem_run/aggregate.py::_emit_channel) -- the mean plus one gaussian-blurred
variant per "source size" (finite-source blur).

    from fph_to_tiffs import fph_zarr_to_tiffs
    fph_zarr_to_tiffs("/path/haadf.zarr", "/path/run.toml")

It prints how many configs were averaged and, if a toml is given, warns when that
count differs from simulations.frozen_phonons -- so you can confirm the full
ensemble was reduced, not a single phonon. Source sizes are blur sigmas in the
measurement's real-space units (abtem `gaussian_filter` converts to pixels via the
measurement's own sampling); with a toml they default to simulations.blur_sigmas /
blur_boundary. Place this file where `abtem_run` is importable (e.g. next to run.py).
"""
import tomllib
import warnings
from pathlib import Path

import numpy as np

import abtem

from abtem_run.compat import apply_abtem_patches


def _read_simulation_toml_fields(toml_path):
    """Read only the [simulations] keys needed here, without validating a full job config."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    sims = data.get("simulations", {})
    return (
        sims.get("blur_sigmas"),
        sims.get("blur_boundary"),
        sims.get("frozen_phonons"),
    )


def ensemble_mean(measurement):
    """Average over ALL ensemble (frozen-phonon) axes. Returns ``(mean, n_configs)``.

    Uses abtem's ``reduce_ensemble`` when the zarr carries ensemble metadata, and
    falls back to averaging the extra leading axes if an old-format zarr stacked the
    configs without that metadata. ``n_configs`` is exactly what got averaged, so the
    caller can confirm the full ensemble was reduced (not one phonon).
    """
    ens = tuple(getattr(measurement, "ensemble_shape", ()) or ())
    if ens:
        n_configs = int(np.prod(ens))
        mean = measurement.reduce_ensemble()
    else:
        base = getattr(measurement, "base_shape", None)
        base_ndim = len(base) if base is not None else 2
        extra = measurement.array.ndim - base_ndim
        if extra > 0:
            # old-format stack with no ensemble axis metadata: average the leading dims
            n_configs = int(np.prod(measurement.array.shape[:extra]))
            mean = measurement.mean(tuple(range(extra)))
        else:
            n_configs = 1
            mean = measurement
    mean = mean.compute() if hasattr(mean, "compute") else mean
    return mean, n_configs




def _resolve_abtem_zarr_path(zarr_path):
    """Return the abTEM-readable zarr node for root or array0-based stores."""
    zarr_path = Path(zarr_path)
    array0 = zarr_path / "array0"
    if (array0 / "zarr.json").exists():
        return array0
    return zarr_path


def fph_zarr_to_tiffs(
    zarr_path,
    toml_path=None,
    *,
    out_dir=None,
    source_sizes=None,
    boundary=None,
    save_zarr=False,
):
    """Write ``<stem>.tif`` (ensemble mean) + ``<stem>_<size>.tif`` per source size.

    Returns the list of written TIFF paths. ``source_sizes`` / ``boundary`` default
    to the toml's ``simulations.blur_sigmas`` / ``blur_boundary`` when a toml is given.
    """
    # The abtem gaussian_filter boundary modes (e.g. the default 'nearest') need the
    # abtem_run compat shim; apply it explicitly here, as the abtem_run entry points do.
    apply_abtem_patches()

    expected_configs = None
    if toml_path is not None:
        toml_sizes, toml_boundary, toml_frozen_phonons = _read_simulation_toml_fields(toml_path)
        if source_sizes is None:
            source_sizes = toml_sizes
        if boundary is None:
            boundary = toml_boundary
        try:
            expected_configs = int(toml_frozen_phonons)
        except (TypeError, ValueError):
            expected_configs = None  # 'None' / disabled -> nothing to cross-check
    boundary = boundary or "nearest"
    source_sizes = source_sizes or []

    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir) if out_dir is not None else zarr_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = zarr_path.stem

    read_path = _resolve_abtem_zarr_path(zarr_path)
    if read_path != zarr_path:
        print(f"fph_to_tiffs: reading zarr array from {read_path}")
    measurement = abtem.from_zarr(str(read_path))
    mean, n_configs = ensemble_mean(measurement)
    print(f"fph_to_tiffs: {zarr_path.name}: averaged {n_configs} frozen-phonon config(s)")
    if expected_configs is not None and n_configs != expected_configs:
        warnings.warn(
            f"{zarr_path.name}: averaged {n_configs} config(s) but the toml expects "
            f"{expected_configs} (simulations.frozen_phonons) -- verify this is the full ensemble.",
            stacklevel=2,
        )
    elif n_configs <= 1:
        warnings.warn(
            f"{zarr_path.name}: only {n_configs} config -- nothing was averaged. Is this the "
            "ensemble zarr, or a single phonon / an already-reduced image?",
            stacklevel=2,
        )

    written = [out_dir / f"{stem}.tif"]
    mean.to_tiff(str(written[0]))
    #if save_zarr:
    #    mean.to_zarr(str(out_dir / f"{stem}_mean.zarr"), overwrite=True)

    for size in source_sizes:
        tag = str(size).replace(".", "-")  # filesystem-friendly, matches abtem_run
        out = out_dir / f"{stem}_{tag}.tif"
        mean.gaussian_filter(size, boundary=boundary).to_tiff(str(out))
        written.append(out)
    return written


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="fph zarr -> ensemble-mean + source-size TIFFs")
    ap.add_argument("zarr", help="path to the frozen-phonon measurement zarr")
    ap.add_argument("--toml", default=None, help="abtem_run config (blur_sigmas + blur_boundary + frozen_phonons)")
    ap.add_argument("--out", default=None, help="output dir (default: alongside the zarr)")
    ap.add_argument("--sizes", default=None, help="comma-separated source sizes (override the toml)")
    ap.add_argument("--save-zarr", action="store_true", help="also write the mean as a zarr")
    args = ap.parse_args(argv)

    sizes = [float(s) for s in args.sizes.split(",")] if args.sizes else None
    for p in fph_zarr_to_tiffs(args.zarr, args.toml, out_dir=args.out,
                               source_sizes=sizes, save_zarr=args.save_zarr):
        print(p)


if __name__ == "__main__":
    main()
