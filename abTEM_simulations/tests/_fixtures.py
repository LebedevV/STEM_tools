"""
Shared test fixtures for the abtem-run test suite.

Single source of truth for the tiny-Pm3m config that every slow / integration
test uses. Tests call ``write_tiny_config(cfg_path, folder_sim=..., cif_dir=...,
<overrides>)`` and get the same TOML schema everywhere — adding a new schema
field means one edit here, not five.

Importable from pytest (via ``tests`` on the pythonpath, configured in
pyproject.toml) and from standalone ``python tests/test_x.py`` runs (Python
adds the script's directory to sys.path).
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


_REPO_ROOT = Path(__file__).resolve().parent.parent


# All format-string placeholders are filled in by write_tiny_config().
# Any literal curly braces in the TOML body need to be doubled ({{ }}).
TINY_TOML_TEMPLATE = dedent("""
	[paths]
	folder_sim = "{folder_sim}"
	extr = "./"
	folder = "{folder}"
	sample_name = "test"

	[gpu_related]
	use_gpu = false
	dask_cuda = false
	cupy_fft_cache_size = "256 MB"
	dask_chunk_size_gpu = "256 MB"
	dask_chunk_size = "256 MB"

	[simulations]
	override_sampling = false
	frozen_phonons = {frozen_phonons}
	fph_sigma = {fph_sigma}
	do_full_run = true
	test_enabled = {test_enabled}
	emit_static_baseline = {emit_static_baseline}
	{blur_sigmas_line}

	[microscope]
	HT_value = 200000
	do_diffraction = {do_diffraction}
	do_cbed = {do_cbed}
	detectors = {detectors}
	convergence_angle = 30.0
	cbed_max_angle = "valid"
	haadfinner = 99
	haadfouter = 200
	abfinner = 15
	abfouter = 33
	bfinner = 0.01
	bfouter = 9
	defocus = {defocus}
	aberrations = {{ {aberrations_inner} }}

	[lamella_settings]
	max_uvw = 10
	sblock_size = 12.0
	scan_s = 6.0
	borders = 1.0
	thickness = 4.0
	extra_shift_z = 0.0
	tol = 0.05
	atom_to_zero = "__skip__"
	global_tilt_a = 0.0
	global_tilt_b = 0.0
	tilt_degrees = true
	add_vacancies_toggle = false
	element_to_remove = "Pm"
	probability_of_vac = 0.0
	vacancies_seed = 0

	[job]
	phase = "Pm3m.cif"
	hkl_to_do = [0, 0, 1]
	is_uvw = false
	phonons_seed = {phonons_seed}
	inplane_angle = 0
	{inplane_align_block}
""").strip()


def setup_cif_dir(tmp: Path) -> Path:
	"""Symlink the repo's ``cifs/`` directory into ``tmp/cifs/``. Returns the path."""
	tmp = Path(tmp)
	cif_dir = tmp / "cifs"
	if not cif_dir.exists():
		cif_dir.symlink_to(_REPO_ROOT / "cifs", target_is_directory=True)
	return cif_dir


def write_tiny_config(
	cfg_path: Path,
	*,
	folder_sim: Path,
	cif_dir: Path,
	frozen_phonons: int = 2,
	fph_sigma: float = 0.05,
	phonons_seed: int = 42,
	test_enabled: bool = False,
	emit_static_baseline: bool = False,
	blur_sigmas: list[float] | None = None,
	do_diffraction: bool = False,
	do_cbed: bool = False,
	detectors: tuple[str, ...] = ("haadf",),
	inplane_align_hkl: tuple[int, int, int] | None = None,
	inplane_align_axis: str = "y",
	defocus: float | str = 0.0,
	aberrations: dict[str, float] | None = None,
) -> Path:
	"""Render the shared tiny-Pm3m TOML to ``cfg_path``.

	Args:
		cfg_path: where to write the .toml file.
		folder_sim: value for ``[paths].folder_sim`` (a trailing slash is added).
		cif_dir: value for ``[paths].folder`` (a trailing slash is added).
		Everything else maps directly to a TOML field; defaults are picked so
		that the suite's fast tests stay fast (do_full_run=true with only the
		haadf detector, no diffraction, no CBED, 2 phonons).

	Returns:
		``cfg_path`` (for convenience chaining).
	"""
	cfg_path = Path(cfg_path)
	detector_list = "[" + ", ".join(f'"{d}"' for d in detectors) + "]"
	if inplane_align_hkl is None:
		inplane_align_block = ""
	else:
		hkl_list = "[" + ", ".join(str(int(x)) for x in inplane_align_hkl) + "]"
		inplane_align_block = (
			f'inplane_align_hkl = {hkl_list}\n'
			f'\tinplane_align_axis = "{inplane_align_axis}"'
		)
	if isinstance(defocus, str):
		defocus_val = f'"{defocus}"'
	else:
		defocus_val = repr(float(defocus))
	aberrations = aberrations or {}
	aberrations_inner = ", ".join(f'{k} = {float(v)!r}' for k, v in aberrations.items())
	if blur_sigmas is None:
		blur_sigmas_line = ""
	else:
		blur_sigmas_line = f"blur_sigmas = [{', '.join(repr(float(s)) for s in blur_sigmas)}]"
	rendered = TINY_TOML_TEMPLATE.format(
		folder_sim=str(folder_sim).rstrip("/") + "/",
		folder=str(cif_dir).rstrip("/") + "/",
		frozen_phonons=int(frozen_phonons),
		fph_sigma=fph_sigma,
		phonons_seed=int(phonons_seed),
		test_enabled=str(test_enabled).lower(),
		emit_static_baseline=str(emit_static_baseline).lower(),
		blur_sigmas_line=blur_sigmas_line,
		do_diffraction=str(do_diffraction).lower(),
		do_cbed=str(do_cbed).lower(),
		detectors=detector_list,
		inplane_align_block=inplane_align_block,
		defocus=defocus_val,
		aberrations_inner=aberrations_inner,
	)
	cfg_path.write_text(rendered)
	return cfg_path


def write_tiny_ground_xyz(job_dir: Path, *, cif_dir: Path) -> Path:
	"""Build the lamella matching the tiny-Pm3m config and write surf.xyz
	at ``job_dir / 'surf.xyz'``. Mirrors the generator's planning step for
	tests that bypass the generator and invoke run_one_seed / aggregate_job
	directly. Returns the surf.xyz path."""
	import ase.io
	from abtem_run.simulation import make_lamella
	cif_path = str(Path(cif_dir) / "Pm3m.cif")
	# Geometry mirrors TINY_TOML_TEMPLATE: borders=1, scan_s=6, thickness=4,
	# sblock_size=12, atom_to_zero='__skip__' sentinel (no atom matches; skip
	# the zero-shift block), inplane_angle=0, no tilt.
	lamella = make_lamella(
		cif_path,
		[0, 0, 1],
		12.0,
		(8.0, 8.0, 4.0),
		"__skip__",
		0.05,
		10,
		is_uvw=False,
		inplane_angle=0.0,
		extra_shift_z=0.0,
		vac_xy=1.0,
		vac_z=1.0,
		global_tilt=(0.0, 0.0),
		tilt_degrees=True,
	)
	surf_path = Path(job_dir) / "surf.xyz"
	# extxyz preserves the cell box — matches generator_run.py's write so
	# the worker's read sees a non-zero z-extent for Potential construction.
	ase.io.write(str(surf_path), lamella, format="extxyz")
	return surf_path
