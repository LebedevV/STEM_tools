"""
Tests for Q4 (V. Lebedev, 2026-05-20): ``cfg.job.inplane_align_hkl`` +
``cfg.job.inplane_align_axis`` "this hkl up" in-plane alignment.

Covers three layers:
  - The pydantic Job model: rejects [0,0,0] and bad shapes; accepts
    negative indices.
  - The helper ``simulation.compute_inplane_angle_from_hkl`` against
    analytical answers on a cubic cell.
  - End-to-end via ``make_lamella(inplane_align_hkl=..., axis=...)``
    on a tiny Pm3m lamella — should override inplane_angle and produce
    a non-trivial in-plane rotation that we can check by comparing
    atom positions to the un-aligned baseline.

Fast (~3-5s). Run via:

    PYTHONPATH=src python3 tests/test_inplane_align.py
    PYTHONPATH=src pytest tests/test_inplane_align.py
"""
from __future__ import annotations

import numpy as np
import pytest

import abtem_run  # noqa: F401 — patches abtem at import
from abtem_run.config import Job
from abtem_run.simulation import compute_inplane_angle_from_hkl


# --------------------------------------------------------------------------- #
# Layer 1: pydantic validation
# --------------------------------------------------------------------------- #


def _base_job_kwargs(**overrides):
	# Minimum kwargs to construct a Job model.
	base = dict(phase="Pm3m.cif", hkl_to_do=[0, 0, 1], is_uvw=False)
	base.update(overrides)
	return base


def test_job_accepts_default_none():
	"""Default is None; field is optional."""
	job = Job(**_base_job_kwargs())
	assert job.inplane_align_hkl is None
	assert job.inplane_align_axis == "y"


def test_job_accepts_negative_indices():
	"""Negative Miller indices are valid."""
	job = Job(**_base_job_kwargs(inplane_align_hkl=[-1, 0, 2]))
	assert job.inplane_align_hkl == [-1, 0, 2]


def test_job_rejects_zero_vector():
	"""[0,0,0] is undefined — pydantic should refuse it."""
	with pytest.raises(Exception):  # pydantic ValidationError wraps ValueError
		Job(**_base_job_kwargs(inplane_align_hkl=[0, 0, 0]))


def test_job_rejects_wrong_length():
	"""Must be exactly 3 ints."""
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_align_hkl=[1, 0]))
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_align_hkl=[1, 0, 0, 0]))


def test_job_rejects_floats():
	"""Indices must be ints, not floats."""
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_align_hkl=[1.5, 0, 0]))


def test_job_rejects_bool_in_inplane_align_hkl():
	"""bool is a subclass of int in Python — without mode='before' on the
	validator, [True, False, True] would silently slide through as
	[1, 0, 1]. Regression for the audit pass after v6_pre5."""
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_align_hkl=[True, False, True]))


def test_job_rejects_bool_for_inplane_angle():
	"""Same audit-pass regression for inplane_angle: True must not
	silently become 1.0."""
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_angle=True))


def test_job_rejects_bad_axis():
	"""Axis is restricted to 'x' or 'y'."""
	with pytest.raises(Exception):
		Job(**_base_job_kwargs(inplane_align_axis="z"))


# --------------------------------------------------------------------------- #
# Layer 2: helper math on a cubic cell
# --------------------------------------------------------------------------- #


# Cubic cell — orthonormal reciprocal lattice, so v_real direction = hkl direction
# (up to scaling), which means we can reason about angles trivially.
_CUBIC = (4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
_IDENT = np.eye(3)


def test_helper_cubic_100_axis_x():
	"""[1,0,0] under identity rotation projects onto +X; angle to land on +X is 0."""
	ang = compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1, 0, 0], axis="x")
	assert ang == pytest.approx(0.0, abs=1e-9)


def test_helper_cubic_100_axis_y():
	"""[1,0,0] on +Y axis means rotate by 0 - 90 = -90 deg."""
	ang = compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1, 0, 0], axis="y")
	assert ang == pytest.approx(-90.0, abs=1e-9)


def test_helper_cubic_010_axis_y():
	"""[0,1,0] already on +Y under identity; angle to land on +Y is 0."""
	ang = compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [0, 1, 0], axis="y")
	assert ang == pytest.approx(0.0, abs=1e-9)


def test_helper_cubic_110_axis_x():
	"""[1,1,0] direction is at 45° from +X; aligning to +X needs rot_angle = 45°."""
	ang = compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1, 1, 0], axis="x")
	assert ang == pytest.approx(45.0, abs=1e-9)


def test_helper_cubic_negative_h():
	"""[-1,0,0] is at 180°; axis='x' returns 180° (vector ends back on +X)."""
	ang = compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [-1, 0, 0], axis="x")
	assert ang == pytest.approx(180.0, abs=1e-9)


def test_helper_rejects_parallel_to_view():
	"""Under identity R, viewing direction = +Z; aligning [0,0,1] is undefined."""
	with pytest.raises(ValueError, match="undefined"):
		compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [0, 0, 1], axis="x")


def test_helper_rejects_zero_vector():
	"""Zero vector has no direction."""
	with pytest.raises(ValueError, match=r"\[0,0,0\]"):
		compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [0, 0, 0], axis="x")


def test_helper_rejects_bad_axis():
	with pytest.raises(ValueError, match="axis"):
		compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1, 0, 0], axis="z")


def test_helper_rejects_bad_shape():
	with pytest.raises(ValueError, match="3 ints"):
		compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1, 0], axis="x")
	with pytest.raises(ValueError, match="3 ints"):
		compute_inplane_angle_from_hkl(_IDENT, _CUBIC, [1.0, 0, 0], axis="x")


# --------------------------------------------------------------------------- #
# Layer 3: integration through make_lamella
# --------------------------------------------------------------------------- #


def test_make_lamella_align_hkl_matches_explicit_inplane_angle():
	"""inplane_align_hkl=[1,1,0]@x on a cubic cell should compute the same
	angle (45°) as setting inplane_angle=45.0 directly — and produce a
	byte-identical lamella. This proves the override path is consistent
	with the existing inplane_angle path (V.'s spec: "amending R as if it
	was a given in-plane angle")."""
	from pathlib import Path
	from abtem_run.simulation import make_lamella

	cif_path = str((Path(__file__).resolve().parent.parent / "cifs" / "Pm3m.cif"))
	common = dict(
		cif_path=cif_path,
		hkl=[0, 0, 1],
		sblock_size=12.0,
		lamella_sizes=(8.0, 8.0, 4.0),
		atom_to_zero="__skip__",  # skip atom-to-zero auto-rotation
		tol=0.05,
		max_uvw=10,
		is_uvw=False,
		extra_shift_z=0.0,
		vac_xy=1.0,
		vac_z=1.0,
		global_tilt=(0.0, 0.0),
		tilt_degrees=True,
	)

	via_align = make_lamella(**common, inplane_align_hkl=[1, 1, 0], inplane_align_axis="x")
	via_explicit = make_lamella(**common, inplane_angle=45.0)

	pa = via_align.get_positions()
	pe = via_explicit.get_positions()
	assert pa.shape == pe.shape, (
		f"shape mismatch: align={pa.shape}, explicit={pe.shape} "
		"— the alignment branch must compute the same angle as the explicit path"
	)
	assert np.allclose(pa, pe, atol=1e-9), (
		f"positions diverge between align and explicit-45deg paths; "
		f"max |Δ| = {np.max(np.abs(pa - pe)):.3e}"
	)


def test_make_lamella_align_hkl_differs_from_no_rotation():
	"""Sanity: inplane_align_hkl=[1,1,0]@x is not the same as
	inplane_angle=0 — alignment actually rotates."""
	from pathlib import Path
	from abtem_run.simulation import make_lamella

	cif_path = str((Path(__file__).resolve().parent.parent / "cifs" / "Pm3m.cif"))
	common = dict(
		cif_path=cif_path,
		hkl=[0, 0, 1],
		sblock_size=12.0,
		lamella_sizes=(8.0, 8.0, 4.0),
		atom_to_zero="__skip__",
		tol=0.05,
		max_uvw=10,
		is_uvw=False,
		extra_shift_z=0.0,
		vac_xy=1.0,
		vac_z=1.0,
		global_tilt=(0.0, 0.0),
		tilt_degrees=True,
	)

	zero = make_lamella(**common, inplane_angle=0.0)
	aligned = make_lamella(**common, inplane_align_hkl=[1, 1, 0], inplane_align_axis="x")
	# atom count may differ — the rotated crop picks up different atoms — but
	# at minimum the two lamellas are not byte-identical.
	if zero.get_positions().shape == aligned.get_positions().shape:
		assert not np.allclose(zero.get_positions(), aligned.get_positions(), atol=1e-9), (
			"align path produced identical lamella to inplane_angle=0 — override didn't fire"
		)


def _run_all():
	# Standalone-script entry point; pytest-only tests are gated by usefixtures
	# or pytest.raises so they error out cleanly here too.
	import inspect
	mod = inspect.getmodule(_run_all)
	for name, fn in inspect.getmembers(mod, inspect.isfunction):
		if not name.startswith("test_"):
			continue
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {name}: {e}")
			return 1
		except Exception as e:
			print(f"ERROR {name}: {type(e).__name__}: {e}")
			return 1
		else:
			print(f"PASS  {name}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
