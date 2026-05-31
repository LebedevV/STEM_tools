#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

# This code is only for reading and validating the config.toml
# Need to be edited only if new variables are added or config file is split

from pathlib import Path
from typing import Any, Literal

import tomllib
from pydantic import BaseModel, Field, field_validator

#If adding a new class of variables, add it to AppConfig, too!
class Paths(BaseModel):
	folder_sim: str = Field()
	extr: str = Field()
	folder: str = Field()
	sample_name: str = Field()
	
class Job(BaseModel):
	"""Job-defining parameters (one per config TOML)."""
	phase: str = Field()
	hkl_to_do: list[int] | list[list[int]] = Field()
	is_uvw: bool = Field()
	phonons_seed: int = Field(default=0)
	inplane_angle: float | str = Field(default=0.0)  # degrees, or 'auto'
	# "This hkl up" alignment: lands the hkl normal's in-plane projection on
	# inplane_align_axis ('x' or 'y'). Overrides inplane_angle when set.
	inplane_align_hkl: list[int] | None = Field(default=None)
	inplane_align_axis: Literal["x", "y"] = Field(default="y")

	@field_validator("hkl_to_do")
	@classmethod
	def validate_hkl_to_do(cls, v: Any):
		# Accept either [h,k,l] or [[h,k,l], ...]
		if isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) for x in v):
			return v
		if isinstance(v, list) and all(isinstance(row, list) for row in v):
			for row in v:
				if len(row) != 3 or not all(isinstance(x, int) for x in row):
					raise ValueError("Each HKL entry must be a list of 3 integers.")
			return v
		raise ValueError("hkl_to_do must be [h,k,l] or a list of [h,k,l] entries.")

	@field_validator("inplane_align_hkl")
	@classmethod
	def validate_inplane_align_hkl(cls, v: Any):
		# Reject [0,0,0] up front (would trip arctan2(0,0) downstream).
		if v is None:
			return None
		if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) for x in v)):
			raise ValueError("inplane_align_hkl must be a list of 3 ints or null")
		if all(x == 0 for x in v):
			raise ValueError("inplane_align_hkl cannot be [0,0,0] — undefined direction")
		return v

	@field_validator("inplane_angle")
	@classmethod
	def validate_inplane_angle(cls, v: Any):
		# A number (degrees), or the literal 'auto' (case-insensitive), which
		# maps to None at use time -> make_lamella's atom_to_zero auto-detect.
		if isinstance(v, str):
			if v.lower() == "auto":
				return "auto"
			raise ValueError("inplane_angle as a string must be 'auto'")
		return float(v)

	@property
	def hkl_list(self) -> list[list[int]]:
		"""hkl_to_do normalized to list-of-lists regardless of input shape."""
		if len(self.hkl_to_do) == 3 and all(isinstance(x, int) for x in self.hkl_to_do):
			return [list(self.hkl_to_do)]  # type: ignore[list-item]
		return [list(row) for row in self.hkl_to_do]

	@property
	def inplane_angle_resolved(self) -> float | None:
		"""inplane_angle as a float, or None for 'auto' (auto-detect branch)."""
		if isinstance(self.inplane_angle, str):
			return None
		return float(self.inplane_angle)

class GpuRelated(BaseModel):
	use_gpu: bool = Field()
	dask_cuda: bool = Field()
	cupy_fft_cache_size: str = Field()
	dask_chunk_size_gpu: str = Field()
	dask_chunk_size: str = Field()

class Simulations(BaseModel):
	override_sampling: float | bool = Field()
	frozen_phonons: int | str | list[int | str] = Field() #str meant to be only 'None'
	fph_sigma: float | bool | str | list[float | bool | str] = Field() #bool meant to be converted to None
	do_full_run: bool = Field()  # run the per-seed scan (probe.scan)
	# test_enabled=true: aggregator keeps outputs/ intact instead of deleting
	# it, AND the worker writes outputs/seed_NNNNNN_displaced.xyz per seed.
	test_enabled: bool = Field(default=False)
	# emit_static_baseline=true: also write aggregate/potential_projection_static.*
	# alongside the phonon-averaged projection.
	emit_static_baseline: bool = Field(default=False)
	# Boundary mode for the gaussian-blur TIFF variants. Threaded into
	# abtem.Images.gaussian_filter(boundary=...).
	blur_boundary: Literal["nearest", "constant", "reflect", "wrap"] = Field(default="nearest")
	# Gaussian-blur sigmas (real-space units) — one blurred TIFF per sigma
	# per channel: aggregate/<channel>_<sigma>.tif. [] skips the blur previews.
	blur_sigmas: list[float] = Field(default_factory=lambda: [0.025, 0.1, 0.25])

	@field_validator("blur_sigmas", mode="before")
	@classmethod
	def validate_blur_sigmas(cls, v: Any):
		if not isinstance(v, list):
			raise ValueError("blur_sigmas must be a list")
		out: list[float] = []
		for x in v:
			if isinstance(x, bool) or not isinstance(x, (int, float)):
				raise ValueError(f"blur_sigmas entries must be numeric")
			if x < 0:
				raise ValueError(f"blur_sigmas entries must be >= 0, got {x}")
			out.append(float(x))
		return out

class Microscope(BaseModel):
	HT_value: int | list[int ] = Field()
	# Plane-wave diffraction pattern, per seed. Optional extra output; off by default.
	do_diffraction: bool = Field(default=False)
	# Convergent-beam diffraction via Probe.multislice at one position. Split
	# out of do_diffraction in the worker era so they gate independently.
	do_cbed: bool = Field(default=False)
	# Which detectors to compute in probe.scan; subset of {haadf, abf, bf}.
	# Default all three.
	detectors: list[str] = Field(default_factory=lambda: ["haadf", "abf", "bf"])
	convergence_angle: float = Field(default=30.0)   # mrad
	cbed_max_angle: float | str = Field(default="valid")
	haadfinner: float = Field()
	haadfouter: float = Field()
	abfinner: float = Field()
	abfouter: float = Field()
	bfinner: float = Field()
	bfouter: float = Field()
	# Probe defocus in Å, or 'scherzer' (computed from C30 + energy).
	defocus: float | str = Field(default="scherzer")
	# Phase aberrations passed to abtem.Probe(aberrations=...). Defocus / C10
	# are rejected — use the `defocus` field above.
	aberrations: dict[str, float] = Field(default_factory=dict)

	@field_validator("defocus", mode="before")
	@classmethod
	def validate_defocus(cls, v: Any):
		# mode='before' so bool doesn't slip through as 1.0 via pydantic coercion.
		if isinstance(v, bool):
			raise ValueError("defocus cannot be a bool")
		if isinstance(v, str):
			if v.lower() == "scherzer":
				return "scherzer"
			raise ValueError(f"defocus as a string must be 'scherzer', got {v!r}")
		return float(v)

	@field_validator("aberrations")
	@classmethod
	def validate_aberrations(cls, v: Any):
		if not isinstance(v, dict):
			raise ValueError("aberrations must be a dict")
		for k in ("defocus", "C10"):
			if k in v:
				raise ValueError(f"set defocus via microscope.defocus, not aberrations[{k!r}]")
		# Lazy import so empty / unset aberrations validates without abtem.
		if v:
			from abtem.transfer import polar_aliases
			valid = (set(polar_aliases) | set(polar_aliases.values())) - {"defocus", "C10"}
			for k, val in v.items():
				if k not in valid:
					raise ValueError(f"aberrations[{k!r}] is not a known abtem aberration symbol")
				if isinstance(val, bool) or not isinstance(val, (int, float)):
					raise ValueError(f"aberrations[{k!r}] must be numeric")
		return {k: float(v) for k, v in v.items()}

	@field_validator("detectors")
	@classmethod
	def validate_detectors(cls, v: Any):
		"""Normalize to lowercase, de-duplicate, reject anything outside {haadf, abf, bf}."""
		if not isinstance(v, list):
			raise ValueError("detectors must be a list of strings")
		allowed = {"haadf", "abf", "bf"}
		normalized = [str(s).lower() for s in v]
		invalid = [s for s in normalized if s not in allowed]
		if invalid:
			raise ValueError(
				f"unknown detector(s): {invalid}; must be a subset of {sorted(allowed)}"
			)
		seen: set[str] = set()
		out = []
		for s in normalized:
			if s not in seen:
				seen.add(s)
				out.append(s)
		return out

class LamellaSettings(BaseModel):
	max_uvw: int = Field()
	sblock_size: float = Field()
	scan_s: float = Field()
	borders: float = Field()
	thickness: float | list[float] = Field()
	extra_shift_z: float = Field()
	tol: float = Field()
	atom_to_zero: str = Field()
	global_tilt_a: float | list[float] = Field()
	global_tilt_b: float | list[float] = Field()
	tilt_degrees: bool = Field()
	add_vacancies_toggle: bool = Field()
	element_to_remove: str = Field()
	probability_of_vac: float | list[float] = Field()
	vacancies_seed: int = Field(default=0)  # RNG seed for add_vacancies; distinct from job.phonons_seed

class AppConfig(BaseModel):
	paths: Paths
	gpu_related: GpuRelated
	microscope: Microscope
	lamella_settings: LamellaSettings
	simulations: Simulations
	job: Job

# Resolve a [paths] field: absolute paths pass through, relative ones resolve
# against `base` (the config file's dir). Trailing '/' keeps folder+phase /
# folder_sim+extr concatenation working.
_resolve_path_field = lambda value, base: str((Path(value) if Path(value).is_absolute() else base / value).resolve()) + "/"  # noqa: E731

def load_config(path: str | Path = 'config.toml') -> AppConfig:
	# The config-file path itself is verbatim (absolute or relative to CWD).
	# paths.folder / paths.folder_sim *inside* the config are then resolved
	# relative to the config file's directory if not absolute, so a config
	# can travel with its CIFs and output dir regardless of CWD.
	full_path = Path(path).resolve()
	config_dir = full_path.parent
	with full_path.open("rb") as f:
		data = tomllib.load(f)
	if "paths" in data:
		for key in ("folder", "folder_sim"):
			if key in data["paths"]:
				data["paths"][key] = _resolve_path_field(data["paths"][key], config_dir)
	return AppConfig.model_validate(data)
