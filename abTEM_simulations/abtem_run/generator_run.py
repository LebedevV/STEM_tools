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
from .job_io import write_seed_todo
from .pipeline import expand_cfg
from .simulation import build_lamella_from_config


log = logging.getLogger(__name__)


def generate_run(config_path: Path = Path("config.toml")) -> Path:
	cfg0 = confread.load_config(config_path)
	frames = list(expand_cfg(cfg0))
	# Include only varying non-tilt parameters in job names, without rounding floats.
	axes = (("fp", "simulations", "frozen_phonons"), ("fs", "simulations", "fph_sigma"),
		("th", "lamella_settings", "thickness"), ("pv", "lamella_settings", "probability_of_vac"),
		("ht", "microscope", "HT_value"))
	frame_dicts = [frame.model_dump() for frame in frames]
	sweep_tags = ["" for frame in frames]
	for tag, section, field in axes:
		values = [str(data[section][field]).removesuffix(".0") for data in frame_dicts]
		if len(set(values)) > 1:
			for index, value in enumerate(values):
				sweep_tags[index] += f"_{tag}{value}"

	out_root = cfg0.paths.output_root
	out_root.mkdir(parents=True, exist_ok=True)

	created_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
	run_dir = out_root / f"gen_{created_utc}"
	run_dir.mkdir()

	manifest: dict[str, Any] = {
		"run_dir": str(run_dir),
		"created_utc": created_utc,
		"base_config": str(Path(config_path).resolve()),
		"n_frames": len(frames),
		"jobs": [],
	}

	for frame_idx, cfg_frame in enumerate(frames):
		cfg_dict = cfg_frame.model_dump()

		raw_phase = cfg_dict["job"]["phase"]
		phase_iter = raw_phase if isinstance(raw_phase, list) else [str(raw_phase)]
		is_uvw = bool(cfg_dict["job"]["is_uvw"])

		hkl_to_do = cfg_dict["job"]["hkl_to_do"]
		hkls = (
			[hkl_to_do]
			if len(hkl_to_do) == 3 and all(isinstance(x, int) for x in hkl_to_do)
			else hkl_to_do
		)

		ls = cfg_dict["lamella_settings"]
		tilt_a = str(ls["global_tilt_a"]).removesuffix(".0")
		tilt_b = str(ls["global_tilt_b"]).removesuffix(".0")
		tilt = f"ta{tilt_a}_tb{tilt_b}"
		frozen = cfg_frame.simulations.frozen_phonons
		seed_start = cfg_frame.job.phonons_seed
		seeds = [0] if frozen == "None" else list(range(seed_start, seed_start + frozen))

		for phase in phase_iter:
			phase = str(phase)
			phase_name = Path(phase).name
			if phase_name.lower().endswith(".cif"):
				phase_name = phase_name[:-4]

			for hkl in hkls:
				line_hkl = "_".join(str(x) for x in hkl)
				vec_kind = "uvw" if is_uvw else "hkl"
				stem = f"{phase_name}_{vec_kind}_{line_hkl}_{tilt}{sweep_tags[frame_idx]}"

				job_dir = run_dir / stem
				job_dir.mkdir()  # Never overwrite a colliding or duplicate job.
				(job_dir / "seeds").mkdir(parents=True, exist_ok=True)
				(job_dir / "outputs").mkdir(parents=True, exist_ok=True)
				(job_dir / "aggregate").mkdir(parents=True, exist_ok=True)

				# Job-local TOML is scalarized to one phase and one direction.
				job_cfg_dict = dict(cfg_dict)
				job_cfg_dict["job"] = dict(cfg_dict["job"])
				job_cfg_dict["job"]["hkl_to_do"] = hkl
				job_cfg_dict["job"]["phase"] = phase
				cfg_out_path = job_dir / f"{stem}.toml"
				cfg_frame_for_phase = confread.AppConfig.model_validate(job_cfg_dict)
				# TOML has no null; omit optional None fields so loading restores defaults.
				tmp = cfg_out_path.with_suffix(".toml.tmp")
				with tmp.open("wb") as f:
					tomli_w.dump(cfg_frame_for_phase.model_dump(exclude_none=True), f)
				os.replace(tmp, cfg_out_path)
				lamella = build_lamella_from_config(cfg_frame_for_phase, hkl)
				# extxyz preserves the cell box; plain xyz drops it.
				ase.io.write(str(job_dir / "surf.xyz"), lamella, "extxyz")

				fig, axes = plt.subplots(1, 3, figsize=(15, 5))
				abtem.show_atoms(lamella, ax=axes[0], title="XY projection")
				scan_s = cfg_frame_for_phase.lamella_settings.scan_s
				borders = cfg_frame_for_phase.lamella_settings.borders
				axes[0].add_patch(mpatches.Rectangle(
					(borders * 2, borders * 2), scan_s, scan_s,
					fill=False, edgecolor="red", linewidth=1.5,
				))
				abtem.show_atoms(lamella, ax=axes[1], title="Cross-section XZ", plane="xz")
				abtem.show_atoms(lamella, ax=axes[2], title="Cross-section YZ", plane="yz")
				indices = " ".join(str(x) for x in hkl)
				direction = f"uvw [{indices}]" if is_uvw else f"hkl ({indices})"
				fig.suptitle(
					f"{cfg_frame_for_phase.paths.sample_name}, {phase_name}, {direction}",
					fontsize=18,
				)
				fig.tight_layout()
				fig.savefig(str(job_dir / "combined.png"), dpi=300)
				plt.close(fig)

				for s in seeds:
					write_seed_todo(job_dir / "seeds", s)

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

	manifest_path = run_dir / "run_manifest.json"
	tmp = run_dir / "run_manifest.json.tmp"
	tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	os.replace(tmp, manifest_path)
	return run_dir


def main():
	"""Module entry point for queue generation."""
	import argparse
	from ._log import configure_default_logging
	configure_default_logging()
	parser = argparse.ArgumentParser(
		prog="python -m abtem_run.generator_run",
		description="Generate the per-seed work queue and planning artifacts.",
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
