#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import json
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


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def _phase_stem(phase: str) -> str:
    phase = str(phase)
    if phase.lower().endswith(".cif"):
        return phase[:-4]
    return phase

def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        tomli_w.dump(data, f)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _cfg_to_dict(cfg) -> dict[str, Any]:
    if hasattr(cfg, "model_dump"):  # pydantic v2
        return cfg.model_dump()
    return cfg.dict()  # pydantic v1 fallback


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


def _hkl_str(hkl: list[int]) -> str:
    return "".join(str(x) for x in hkl)


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
    ase.io.write(str(job_dir / "surf.xyz"), lamella, "xyz")
    _emit_combined_png(lamella, cfg_frame, hkl, line_hkl, job_dir)


def generate_run(config_path: Path = Path("config.toml")) -> Path:
    cfg0 = confread.load_config(config_path)
    frames = list(expand_cfg(cfg0))

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
        cfg_dict = _cfg_to_dict(cfg_frame)

        phase = str(cfg_dict["job"]["phase"])
        phase_name = _phase_stem(phase)
        is_uvw = bool(cfg_dict["job"]["is_uvw"])
        hkls = _iter_hkls(cfg_dict)
        seeds = _seed_list_from_cfg(cfg_dict)
        tilt = _tilt_str(cfg_dict)

        # For each HKL, create an independent "job folder"
        for hkl in hkls:
            line_hkl = _hkl_str(hkl)

            # This is the naming analogue of: f"{sg}_{line_hkl}_{ctx.global_tilt}.toml"
            # We use (phase, line_hkl, tilt) because sg isn't known until CIF is parsed.
            stem = f"{phase_name}_{line_hkl}_{tilt}"

            job_dir = run_dir / stem
            (job_dir / "seeds").mkdir(parents=True, exist_ok=True)
            (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (job_dir / "aggregate").mkdir(parents=True, exist_ok=True)

            # Write job-local TOML. Scalarize hkl_to_do to this job's single
            # hkl: the worker rebuilds the lamella from cfg.job.hkl_list[0], so
            # each job dir must carry only its own direction, not the full sweep.
            job_cfg_dict = dict(cfg_dict)
            job_cfg_dict["job"] = dict(cfg_dict["job"])
            job_cfg_dict["job"]["hkl_to_do"] = hkl
            cfg_out_path = job_dir / f"{stem}.toml"
            _atomic_write_toml(cfg_out_path, job_cfg_dict)

            # Planning artifacts: surf.xyz + combined.png. Cheap, no GPU.
            _emit_planning_artifacts(cfg_frame, hkl, line_hkl, job_dir)

            # Create one .todo per seed (or seed 0 baseline if no phonons)
            for s in seeds:
                _atomic_write_text(job_dir / "seeds" / f"seed_{s:06d}.todo", f"{s}\n")

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


if __name__ == "__main__":
    d = generate_run(Path("config.toml"))
    print("Generated:", d)
