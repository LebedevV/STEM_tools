#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import abtem
import ase.io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tomli_w

from . import config as confread
from .pipeline import expand_cfg
from .simulation import build_lamella_from_config


log = logging.getLogger(__name__)


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _phase_stem(phase: str) -> str:
    phase = str(phase)
    if phase.lower().endswith(".cif"):
        return phase[:-4]
    return phase

def _strip_none(data):
    """Recursively drop dict entries whose value is None.

    TOML has no null type; tomli_w raises TypeError on None. Optional fields
    like cfg.job.inplane_align_hkl default to None and that's the natural
    pydantic representation, so the right fix is to omit them from the
    serialized TOML rather than coerce to a sentinel. The reader
    (load_config + pydantic) re-defaults them on the next load.
    """
    if isinstance(data, dict):
        return {k: _strip_none(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [_strip_none(v) for v in data]
    return data


def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        tomli_w.dump(_strip_none(data), f)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _seed_list_from_cfg(cfg_dict: dict[str, Any]) -> list[int]:
    sim = cfg_dict.get("simulations", {})
    job = cfg_dict.get("job", {})

    frozen = sim.get("frozen_phonons", None)
    seed_start = int(job.get("phonons_seed", 0))

    if frozen is None or frozen == "None":
        return [0]  # baseline (no phonons)

    n = int(frozen)
    return list(range(seed_start, seed_start + n))


def _iter_hkls(cfg_dict: dict[str, Any]) -> list[list[int]]:
    job = cfg_dict["job"]
    h = job["hkl_to_do"]
    if isinstance(h, list) and len(h) == 3 and all(isinstance(x, int) for x in h):
        return [h]
    return h  # already validated as list[list[int]]


def _tilt_str(cfg_dict: dict[str, Any]) -> str:
    ls = cfg_dict["lamella_settings"]
    a = float(ls["global_tilt_a"])
    b = float(ls["global_tilt_b"])
    # stable, filesystem-friendly
    return f"ta{a}_tb{b}".replace(" ", "")


def _emit_combined_png(lamella, cfg_frame, hkl, line_hkl, job_dir: Path) -> None:
    """3-panel atom view (XY / XZ / YZ) with the scan box overlaid on XY.
    Cheap — no GPU, no abtem multislice. Just matplotlib + ase + abtem.show_atoms.
    """
    ls = cfg_frame.lamella_settings
    borders = ls.borders
    scan_s = ls.scan_s

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    abtem.show_atoms(lamella, ax=axes[0], title="XY projection")
    # Scan-box overlay: atoms are placed with vac_xy=borders offset, scan area
    # is [2*borders, 2*borders + scan_s] in the world coords.
    rect = mpatches.Rectangle(
        (borders * 2, borders * 2),
        scan_s,
        scan_s,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
    )
    axes[0].add_patch(rect)
    abtem.show_atoms(lamella, ax=axes[1], title="Cross-section XZ", plane="xz")
    abtem.show_atoms(lamella, ax=axes[2], title="Cross-section YZ", plane="yz")

    sample_name = cfg_frame.paths.sample_name
    sg = _phase_stem(cfg_frame.job.phase)
    vec_kind = "uvw" if cfg_frame.job.is_uvw else "hkl"
    fig.suptitle(
        f"{sample_name}, {sg}, {vec_kind} [{line_hkl}]",
        fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(str(job_dir / "combined.png"), dpi=300)
    plt.close(fig)


def _emit_planning_artifacts(cfg_frame, hkl, line_hkl, job_dir: Path) -> None:
    """Build the lamella and emit surf.xyz + combined.png at planning time.

    Cheap (matplotlib + ase + dask CPU, no GPU multislice), useful as a
    sanity check before committing GPU time to the workers.
    """
    lamella = build_lamella_from_config(cfg_frame, hkl)
    # extxyz preserves the cell box; plain xyz drops it and downstream
    # Potential construction crashes on box[2]=0 (division by zero).
    ase.io.write(str(job_dir / "surf.xyz"), lamella, "extxyz")
    _emit_combined_png(lamella, cfg_frame, hkl, line_hkl, job_dir)


def _tagval(v) -> str:
    """Filesystem-friendly stringification of a sweep value (10.0 -> '10')."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _frame_sweep_tags(frames) -> list[str]:
    """One stem suffix per frame encoding the non-tilt sweep axes that vary
    across ``frames`` (empty when none vary). The job-dir stem is
    ``{phase}_{hkl}_{tilt}``; without this, a sweep over a non-tilt axis
    (frozen_phonons / fph_sigma / thickness / probability_of_vac / HT_value)
    would collide on the same stem and silently overwrite the earlier frame's
    job dir. Single-run and tilt-only sweeps keep their original names."""
    axes = (
        ("fp", lambda c: c["simulations"]["frozen_phonons"]),
        ("fs", lambda c: c["simulations"]["fph_sigma"]),
        ("th", lambda c: c["lamella_settings"]["thickness"]),
        ("pv", lambda c: c["lamella_settings"]["probability_of_vac"]),
        ("ht", lambda c: c["microscope"]["HT_value"]),
    )
    dicts = [f.model_dump() for f in frames]
    varying = [(tag, get) for tag, get in axes if len({_tagval(get(d)) for d in dicts}) > 1]
    return ["".join(f"_{tag}{_tagval(get(d))}" for tag, get in varying) for d in dicts]


def generate_run(config_path: Path = Path("config.toml")) -> Path:
    cfg0 = confread.load_config(config_path)
    frames = list(expand_cfg(cfg0))
    # Per-frame stem suffix so non-tilt sweep axes don't collide (see helper).
    sweep_tags = _frame_sweep_tags(frames)

    # Output root from TOML
    out_root = Path(cfg0.paths.folder_sim) / cfg0.paths.extr
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / f"gen_{_now_utc_compact()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "created_utc": _now_utc_compact(),
        "base_config": str(Path(config_path).resolve()),
        "n_frames": len(frames),
        "jobs": [],
    }

    for frame_idx, cfg_frame in enumerate(frames):
        cfg_dict = cfg_frame.model_dump()

        # phase can be a single string OR a list. Normalize and iterate;
        # each phase becomes its own job dir(s) with locked seed integers
        # across phases for direct phase-vs-phase comparisons.
        raw_phase = cfg_dict["job"]["phase"]
        phase_iter = raw_phase if isinstance(raw_phase, list) else [str(raw_phase)]
        is_uvw = bool(cfg_dict["job"]["is_uvw"])
        hkls = _iter_hkls(cfg_dict)
        seeds = _seed_list_from_cfg(cfg_dict)
        tilt = _tilt_str(cfg_dict)

        for phase in phase_iter:
            phase = str(phase)
            phase_name = _phase_stem(phase)

            # For each HKL, create an independent "job folder"
            for hkl in hkls:
                line_hkl = "".join(str(x) for x in hkl)

                # This is the naming analogue of: f"{sg}_{line_hkl}_{ctx.global_tilt}.toml"
                # We use (phase, line_hkl, tilt) because sg isn't known until CIF is parsed.
                stem = f"{phase_name}_{line_hkl}_{tilt}{sweep_tags[frame_idx]}"

                job_dir = run_dir / stem
                (job_dir / "seeds").mkdir(parents=True, exist_ok=True)
                (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
                (job_dir / "aggregate").mkdir(parents=True, exist_ok=True)

                # Write job-local TOML. Scalarize hkl_to_do AND phase so the
                # job dir carries a single direction and a single CIF (the
                # worker rebuilds the lamella from cfg.job.hkl_list[0] +
                # cfg.job.phase, and expects scalars).
                job_cfg_dict = dict(cfg_dict)
                job_cfg_dict["job"] = dict(cfg_dict["job"])
                job_cfg_dict["job"]["hkl_to_do"] = hkl
                job_cfg_dict["job"]["phase"] = phase
                cfg_out_path = job_dir / f"{stem}.toml"
                _atomic_write_toml(cfg_out_path, job_cfg_dict)

                # Planning artifacts: surf.xyz + combined.png. Cheap, no GPU.
                # Use the per-phase cfg_frame so make_lamella reads the right CIF.
                cfg_frame_for_phase = confread.AppConfig.model_validate(job_cfg_dict)
                _emit_planning_artifacts(cfg_frame_for_phase, hkl, line_hkl, job_dir)

                # Create one .todo per seed (or seed 0 baseline if no phonons).
                # Atomic write via tmp + os.replace; seeds/ already exists.
                for s in seeds:
                    todo = job_dir / "seeds" / f"seed_{s:06d}.todo"
                    tmp = todo.with_suffix(todo.suffix + ".tmp")
                    tmp.write_text(f"{s}\n", encoding="utf-8")
                    os.replace(tmp, todo)

                manifest["jobs"].append(
                    {
                        "frame_id": frame_idx,
                        "phase": phase,
                        "hkl": hkl,
                        "is_uvw": is_uvw,
                        "tilt": tilt,
                        "job_dir": str(job_dir.relative_to(run_dir)),
                        "cfg": str(cfg_out_path.relative_to(run_dir)),
                        "n_tasks": len(seeds),
                    }
                )

    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def main():
    """``abtem-run-generate`` console-script entry: plan the job tree from a TOML
    config (no GPU). Writes ``gen_<UTC>/`` with per-seed ``.todo`` files +
    planning artifacts (surf.xyz, combined.png)."""
    import argparse
    import sys
    from ._log import configure_default_logging
    configure_default_logging()
    parser = argparse.ArgumentParser(
        prog="abtem-run-generate",
        description="Generate the per-seed work queue + planning artifacts from a TOML config.",
    )
    parser.add_argument(
        "config", nargs="?", default="config.toml",
        help="TOML config file (default: config.toml in CWD)",
    )
    args = parser.parse_args()
    d = generate_run(Path(args.config))
    log.info(f"Generated: {d}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
