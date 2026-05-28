#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

# This code is only for reading and validating the config.toml
# Need to be edited only if new variables are added or config file is split

from pathlib import Path
from typing import Any

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
	do_full_run: bool = Field()

class Microscope(BaseModel):
	HT_value: int | list[int ] = Field()
	do_diffraction: bool = Field()
	convergence_angle: float = Field(default=30.0)   # mrad
	cbed_max_angle: float | str = Field(default="valid")
	haadfinner: float = Field()
	haadfouter: float = Field()
	abfinner: float = Field()
	abfouter: float = Field()
	bfinner: float = Field()
	bfouter: float = Field()
	
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

def _resolve_path_field(value: str, base: Path) -> str:
	"""
	Resolve a [paths] field: absolute paths pass through, relative ones
	resolve against `base` (the config file's directory). Returns a string
	ending in '/' so the `folder + phase` / `folder_sim + extr` concatenation
	patterns keep working.
	"""
	p = Path(value)
	resolved = p.resolve() if p.is_absolute() else (base / p).resolve()
	return str(resolved) + "/"

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
