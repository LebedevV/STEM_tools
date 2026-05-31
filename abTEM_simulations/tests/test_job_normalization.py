"""
Tests for the helper properties on config.Job that cli.main now relies on
to wire [job] through (Phase 2 #3, the long-standing TODO from 0225f62).

Runnable two ways:
    PYTHONPATH=src python3 tests/test_job_normalization.py
    PYTHONPATH=src pytest tests/test_job_normalization.py
"""
from __future__ import annotations

from pydantic import ValidationError

from abtem_run.config import Job


def _job(**overrides):
	"""Construct a Job with minimal valid defaults."""
	defaults = dict(
		phase="TaTe2_2310358.cif",
		hkl_to_do=[0, 0, 1],
		is_uvw=False,
		phonons_seed=15,
	)
	defaults.update(overrides)
	return Job(**defaults)


def test_hkl_list_wraps_single_triple():
	"""[0,0,1] (a single hkl) becomes [[0,0,1]]."""
	job = _job(hkl_to_do=[0, 0, 1])
	assert job.hkl_list == [[0, 0, 1]]


def test_hkl_list_passes_through_list_of_lists():
	"""[[1,1,0],[0,0,1]] stays a list of lists."""
	job = _job(hkl_to_do=[[1, 1, 0], [0, 0, 1]])
	assert job.hkl_list == [[1, 1, 0], [0, 0, 1]]


def test_hkl_list_returns_independent_copies():
	"""Mutating the returned list must NOT affect the model's hkl_to_do."""
	job = _job(hkl_to_do=[0, 0, 1])
	hkls = job.hkl_list
	hkls[0][0] = 99
	# Original is_uvw / hkl_to_do should be untouched.
	assert job.hkl_to_do == [0, 0, 1]


# --- inplane_angle ---


def test_inplane_angle_default_is_zero():
	job = _job()
	assert job.inplane_angle == 0.0
	assert job.inplane_angle_resolved == 0.0


def test_inplane_angle_accepts_numeric():
	assert _job(inplane_angle=45).inplane_angle_resolved == 45.0
	assert _job(inplane_angle=-12.5).inplane_angle_resolved == -12.5


def test_inplane_angle_auto_maps_to_none():
	"""'auto' (any case) becomes None at resolution, triggering
	make_lamella's auto-detect branch."""
	for s in ("auto", "AUTO", "Auto"):
		job = _job(inplane_angle=s)
		assert job.inplane_angle == "auto"
		assert job.inplane_angle_resolved is None


def test_inplane_angle_rejects_other_strings():
	"""Any string other than 'auto' must fail validation."""
	for s in ("none", "nan", "0", ""):
		try:
			_job(inplane_angle=s)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for inplane_angle={s!r}")


# --------------------------------------------------------------------------- #
# Direction 6: multiple phases per job
# --------------------------------------------------------------------------- #


def test_phase_list_wraps_single_string():
	"""Scalar phase string normalizes to a 1-element list."""
	j = _job(phase="TaTe2_2310358.cif")
	assert j.phase_list == ["TaTe2_2310358.cif"]


def test_phase_list_passes_through_list():
	j = _job(phase=["a.cif", "b.cif", "c.cif"])
	assert j.phase_list == ["a.cif", "b.cif", "c.cif"]


def test_phase_list_dedups_preserving_order():
	"""Duplicate CIFs would collide on the per-phase job-dir name. The
	validator removes duplicates while preserving first-seen order."""
	j = _job(phase=["a.cif", "b.cif", "a.cif", "c.cif", "b.cif"])
	assert j.phase_list == ["a.cif", "b.cif", "c.cif"]


def test_phase_rejects_empty_string():
	try:
		_job(phase="")
	except ValidationError:
		pass
	else:
		raise AssertionError("expected ValidationError for empty phase string")


def test_phase_rejects_empty_list():
	try:
		_job(phase=[])
	except ValidationError:
		pass
	else:
		raise AssertionError("expected ValidationError for empty phase list")


def test_phase_rejects_non_string_entries_in_list():
	try:
		_job(phase=["a.cif", 42, "c.cif"])
	except ValidationError:
		pass
	else:
		raise AssertionError("expected ValidationError for non-string entry")


def _run_all():
	for fn in (
		test_hkl_list_wraps_single_triple,
		test_hkl_list_passes_through_list_of_lists,
		test_hkl_list_returns_independent_copies,
		test_inplane_angle_default_is_zero,
		test_inplane_angle_accepts_numeric,
		test_inplane_angle_auto_maps_to_none,
		test_inplane_angle_rejects_other_strings,
		test_phase_list_wraps_single_string,
		test_phase_list_passes_through_list,
		test_phase_list_dedups_preserving_order,
		test_phase_rejects_empty_string,
		test_phase_rejects_empty_list,
		test_phase_rejects_non_string_entries_in_list,
	):
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {fn.__name__}: {e}")
			return 1
		except Exception as e:
			print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
			return 1
		else:
			print(f"PASS  {fn.__name__}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
