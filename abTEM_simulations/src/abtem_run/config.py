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


# Mirror of abtem.transfer.polar_aliases (abtem 1.0.9) plus the polar symbols
# themselves. Hardcoded so config validation doesn't have to import abtem
# (slow, and runs the runtime monkey-patches). If abtem extends its supported
# aberration set, mirror new symbols here. Excludes 'defocus' / 'C10' — those
# go through the dedicated Microscope.defocus field so the 'scherzer' magic
# stays in one place.
_ABERRATION_NAMED = {
	"Cs", "C5",
	"astigmatism", "astigmatism_angle",
	"astigmatism3", "astigmatism3_angle",
	"astigmatism5", "astigmatism5_angle",
	"coma", "coma_angle",
	"coma4", "coma4_angle",
	"trefoil", "trefoil_angle",
	"trefoil4", "trefoil4_angle",
	"quadrafoil", "quadrafoil_angle",
	"quadrafoil5", "quadrafoil5_angle",
	"pentafoil", "pentafoil_angle",
	"hexafoil", "hexafoil_angle",
}
_ABERRATION_POLAR = {
	"C30", "C50",
	"C12", "phi12",
	"C32", "phi32",
	"C52", "phi52",
	"C21", "phi21",
	"C41", "phi41",
	"C23", "phi23",
	"C43", "phi43",
	"C34", "phi34",
	"C54", "phi54",
	"C45", "phi45",
	"C56", "phi56",
}
_VALID_ABERRATION_KEYS = _ABERRATION_NAMED | _ABERRATION_POLAR

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
	# "This hkl up" alignment: if set, the in-plane angle is computed so
	# the projection of the hkl normal (in lab XY, after the out-of-plane
	# rotation has been applied) lands on the chosen lab axis ('x' or 'y').
	# When set, OVERRIDES inplane_angle and the 'auto' atom-to-zero path.
	# See simulation.compute_inplane_angle_from_hkl.
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
		# Must be None, or a list of exactly three ints. [0,0,0] is undefined
		# (zero vector has no direction) so reject it here rather than letting
		# the downstream code raise from arctan2(0,0). Negative indices are
		# fine and meaningful — don't reject them.
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
	# emit_static_baseline=true: also emit a separate static-lattice (no
	# phonons) reference, kept apart from the phonon-averaged result. Gates
	# both: (a) a static-lattice projected-potential preview
	# (aggregate/potential_projection_static.*) alongside the phonon-averaged
	# projection, and (b) ONE extra static-lattice scan per job
	# (aggregate/<det>_static.{tif,zarr}). The scan path is one additional
	# multislice per job regardless of seed count; diffraction/CBED static
	# baselines are not produced (out of scope).
	emit_static_baseline: bool = Field(default=False)
	# Boundary mode for the post-aggregation gaussian-blur TIFF variants.
	# Default 'nearest' (extends edge values outward) replaces the older
	# 'constant' (pads with 0) which produced dark halos at lamella edges.
	# 'constant' remains available for byte-comparability with pre-2026-05
	# outputs. 'reflect' and 'wrap' are scipy.ndimage.gaussian_filter modes,
	# threaded straight through abtem.Images.gaussian_filter(boundary=...).
	blur_boundary: Literal["nearest", "constant", "reflect", "wrap"] = Field(default="nearest")

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
	# Probe defocus — accepts either a float in Ångström, or the literal
	# string 'scherzer' to ask abtem to compute Scherzer defocus from C30
	# (spherical aberration) and the beam energy. WARNING: 'scherzer' is a
	# silent no-op when C30 == 0 (the formula evaluates to 0); the probe
	# builder emits a runtime warning in that case so the "BF looks like
	# DF" symptom can't recur without explanation.
	defocus: float | str = Field(default="scherzer")
	# Phase-aberration coefficients passed straight through to
	# abtem.Probe(aberrations=...). All values are in Ångström (or
	# radians for the angular phi terms), matching abtem's convention.
	# Common keys: 'C30' (= spherical aberration, Cs), 'C50', 'C12'
	# (twofold astigmatism), 'phi12' (its angle), 'C32', 'C34', etc.
	# Defocus is NOT set here — use the top-level `defocus` field above
	# instead, so the 'scherzer' magic stays consistent. If a 'defocus' /
	# 'C10' key appears in this dict, it's an error.
	aberrations: dict[str, float] = Field(default_factory=dict)

	@field_validator("defocus", mode="before")
	@classmethod
	def validate_defocus(cls, v: Any):
		# A number, or the literal string 'scherzer' (case-insensitive).
		# mode='before' so we see the raw value before pydantic coerces
		# bool -> int -> float (True would otherwise slip through as 1.0).
		if isinstance(v, bool):
			raise ValueError("defocus cannot be a bool")
		if isinstance(v, str):
			if v.lower() == "scherzer":
				return "scherzer"
			raise ValueError(
				f"defocus as a string must be 'scherzer', got {v!r}"
			)
		return float(v)

	@field_validator("aberrations")
	@classmethod
	def validate_aberrations(cls, v: Any):
		# Reject defocus / C10 here — they go through the dedicated
		# `defocus` field so the 'scherzer' magic stays in one place.
		if not isinstance(v, dict):
			raise ValueError("aberrations must be a dict")
		for k in ("defocus", "C10"):
			if k in v:
				raise ValueError(
					f"set defocus via microscope.defocus, not "
					f"microscope.aberrations[{k!r}]"
				)
		# Reject keys outside abtem's supported set with a friendly message
		# listing the polar symbols (catches typos like 'C20' or 'phi21_').
		# Also reject non-numeric values up front (abtem would error later).
		for k, val in v.items():
			if k not in _VALID_ABERRATION_KEYS:
				raise ValueError(
					f"aberrations[{k!r}] is not a known abtem aberration "
					f"symbol. Polar symbols: {sorted(_ABERRATION_POLAR)!r}; "
					f"named aliases: {sorted(_ABERRATION_NAMED)!r}."
				)
			if isinstance(val, bool) or not isinstance(val, (int, float)):
				raise ValueError(
					f"aberrations[{k!r}] must be numeric, got {type(val).__name__}"
				)
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
