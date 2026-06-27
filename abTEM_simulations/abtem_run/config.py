#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

# TOML schema and validation.

from pathlib import Path
from typing import Any, Literal

import tomllib
from pydantic import BaseModel, Field, field_validator, model_validator


def _map_scalar_or_list(v: Any, validate_one):
	"""Apply a scalar validator to either a scalar config value or a sweep list."""
	if isinstance(v, list):
		if not v:
			raise ValueError("sweep lists must not be empty")
		return [validate_one(x) for x in v]
	return validate_one(v)


def _finite_number(v: Any, field_name: str) -> float:
	"""Return v as float, rejecting bools and non-numeric values early."""
	if isinstance(v, bool) or not isinstance(v, (int, float)):
		raise ValueError(f"{field_name} must be numeric, got {v!r}")
	return float(v)

#If adding a new class of variables, add it to AppConfig, too!
class Paths(BaseModel):
	folder_sim: str = Field()
	extr: str = Field()
	folder: str = Field()
	sample_name: str = Field()
	
class Job(BaseModel):
	"""Job-defining parameters (one per config TOML)."""
	phase: str | list[str] = Field()
	hkl_to_do: list[int] | list[list[int]] = Field()
	is_uvw: bool = Field()
	phonons_seed: int = Field(default=0)
	inplane_angle: float | str = Field(default=0.0)  # degrees, or 'auto'
	# "This hkl up" alignment: lands the hkl normal's in-plane projection on
	# inplane_align_axis ('x' or 'y'). Overrides inplane_angle when set.
	inplane_align_hkl: list[int] | None = Field(default=None)
	inplane_align_axis: Literal["x", "y"] = Field(default="y")

	@field_validator("phase")
	@classmethod
	def validate_phase(cls, v: Any):
		# Single CIF name (str) or non-empty list of CIF names.
		if isinstance(v, str):
			if not v.strip():
				raise ValueError("phase string must not be empty")
			return v
		if isinstance(v, list):
			if not v:
				raise ValueError("phase list must not be empty")
			for p in v:
				if not isinstance(p, str) or not p.strip():
					raise ValueError(
						f"every entry in phase list must be a non-empty string, "
						f"got {p!r}"
					)
			# de-dup while preserving order — otherwise the generator emits
			# duplicate job dirs for the same phase, which collides on disk
			# because the dir name is derived from the phase stem.
			seen: set[str] = set()
			deduped = []
			for p in v:
				if p not in seen:
					seen.add(p)
					deduped.append(p)
			return deduped
		raise ValueError(
			f"phase must be a string or a non-empty list of strings, got "
			f"{type(v).__name__}"
		)

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
	def phase_list(self) -> list[str]:
		"""`phase` normalized to a list regardless of input shape. The generator
		iterates over this; downstream per-job TOMLs always carry a single
		scalar phase."""
		if isinstance(self.phase, str):
			return [self.phase]
		return list(self.phase)

	@property
	def hkl_list(self) -> list[list[int]]:
		"""hkl_to_do normalized to list-of-lists regardless of input shape."""
		# Narrow via isinstance checks instead of relying on `# type: ignore`
		# so mypy can follow the type discrimination cleanly.
		v = self.hkl_to_do
		if len(v) == 3 and all(isinstance(x, int) for x in v):
			# Flat [h,k,l]: every element is an int.
			return [[int(x) for x in v]]  # type: ignore[arg-type]
		# Nested [[h,k,l], ...]: every element is itself a list[int].
		return [list(row) for row in v]  # type: ignore[arg-type]

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
	# per channel: <vdir>/scans/<channel>_<sigma>.tif. [] skips the blur previews.
	blur_sigmas: list[float] = Field(default_factory=lambda: [0.025, 0.1, 0.25])

	@field_validator("override_sampling", mode="before")
	@classmethod
	def validate_override_sampling(cls, v: Any):
		if v is False:
			return False
		if isinstance(v, bool):
			raise ValueError("override_sampling must be false or a positive number")
		sampling = _finite_number(v, "override_sampling")
		if sampling <= 0:
			raise ValueError("override_sampling must be > 0, or false to use the default")
		return sampling

	@field_validator("frozen_phonons", mode="before")
	@classmethod
	def validate_frozen_phonons(cls, v: Any):
		def one(x: Any):
			if isinstance(x, bool):
				raise ValueError("frozen_phonons cannot be a bool")
			if isinstance(x, str):
				if x == "None":
					return x
				raise ValueError("frozen_phonons as a string must be exactly 'None'")
			if not isinstance(x, int):
				raise ValueError(f"frozen_phonons must be an int >= 1 or 'None', got {x!r}")
			if x < 1:
				raise ValueError(f"frozen_phonons must be >= 1, got {x}")
			return x
		return _map_scalar_or_list(v, one)

	@field_validator("fph_sigma", mode="before")
	@classmethod
	def validate_fph_sigma(cls, v: Any):
		def one(x: Any):
			if x is False:
				return False
			if isinstance(x, bool):
				raise ValueError("fph_sigma must be false or a non-negative number")
			sigma = _finite_number(x, "fph_sigma")
			if sigma < 0:
				raise ValueError(f"fph_sigma must be >= 0, got {sigma}")
			return sigma
		return _map_scalar_or_list(v, one)

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
	do_diffraction: bool = Field(default=False)
	# Convergent-beam diffraction at one probe position.
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

	@field_validator("HT_value", mode="before")
	@classmethod
	def validate_HT_value(cls, v: Any):
		def one(x: Any):
			if isinstance(x, bool) or not isinstance(x, int):
				raise ValueError(f"HT_value must be an int in eV, got {x!r}")
			if x <= 0:
				raise ValueError(f"HT_value must be > 0, got {x}")
			return x
		return _map_scalar_or_list(v, one)

	@field_validator("convergence_angle", mode="before")
	@classmethod
	def validate_convergence_angle(cls, v: Any):
		angle = _finite_number(v, "convergence_angle")
		if angle <= 0:
			raise ValueError("convergence_angle must be > 0")
		return angle

	@field_validator(
		"haadfinner", "haadfouter", "abfinner", "abfouter",
		"bfinner", "bfouter", mode="before",
	)
	@classmethod
	def validate_detector_angle(cls, v: Any):
		angle = _finite_number(v, "detector angle")
		if angle < 0:
			raise ValueError("detector angles must be >= 0")
		return angle

	@model_validator(mode="after")
	def validate_detector_ranges(self):
		for name in ("haadf", "abf", "bf"):
			inner = getattr(self, f"{name}inner")
			outer = getattr(self, f"{name}outer")
			if outer <= inner:
				raise ValueError(
					f"{name} detector requires outer > inner, got "
					f"inner={inner}, outer={outer}"
				)
		return self

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

	@field_validator("max_uvw", mode="before")
	@classmethod
	def validate_max_uvw(cls, v: Any):
		if isinstance(v, bool) or not isinstance(v, int):
			raise ValueError(f"max_uvw must be an int > 0, got {v!r}")
		if v <= 0:
			raise ValueError(f"max_uvw must be > 0, got {v}")
		return v

	@field_validator("sblock_size", "scan_s", mode="before")
	@classmethod
	def validate_positive_scalar_length(cls, v: Any):
		value = _finite_number(v, "length")
		if value <= 0:
			raise ValueError("sblock_size and scan_s must be > 0")
		return value

	@field_validator("thickness", mode="before")
	@classmethod
	def validate_thickness(cls, v: Any):
		def one(x: Any):
			value = _finite_number(x, "thickness")
			if value <= 0:
				raise ValueError(f"thickness must be > 0, got {value}")
			return value
		return _map_scalar_or_list(v, one)

	@field_validator("borders", "tol", mode="before")
	@classmethod
	def validate_nonnegative_scalar_length(cls, v: Any):
		value = _finite_number(v, "length")
		if value < 0:
			raise ValueError("borders and tol must be >= 0")
		return value

	@field_validator("probability_of_vac", mode="before")
	@classmethod
	def validate_probability_of_vac(cls, v: Any):
		def one(x: Any):
			prob = _finite_number(x, "probability_of_vac")
			if not 0 <= prob <= 1:
				raise ValueError(f"probability_of_vac must be in [0, 1], got {prob}")
			return prob
		return _map_scalar_or_list(v, one)

	@field_validator("global_tilt_a", "global_tilt_b", mode="before")
	@classmethod
	def validate_global_tilt(cls, v: Any):
		return _map_scalar_or_list(v, lambda x: _finite_number(x, "global_tilt"))

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
