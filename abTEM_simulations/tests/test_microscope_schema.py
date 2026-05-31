"""
Schema tests for the new worker-era fields on `Microscope` and `Simulations`:

- microscope.detectors (list[str], validator: subset of {haadf, abf, bf})
- microscope.do_diffraction default flipped to False
- microscope.do_cbed (new, default False)
- simulations.test_enabled (new, default False)

Runnable two ways:
    PYTHONPATH=src python3 tests/test_microscope_schema.py
    PYTHONPATH=src pytest tests/test_microscope_schema.py
"""
from __future__ import annotations

from pydantic import ValidationError

from abtem_run.config import Microscope, Simulations


def _microscope(**overrides):
	"""Construct a Microscope with minimum required fields."""
	defaults = dict(
		HT_value=200000,
		haadfinner=99,
		haadfouter=200,
		abfinner=15,
		abfouter=33,
		bfinner=0.01,
		bfouter=9,
	)
	defaults.update(overrides)
	return Microscope(**defaults)


def _sim(**overrides):
	defaults = dict(
		override_sampling=False,
		frozen_phonons="None",
		fph_sigma=False,
		do_full_run=False,
	)
	defaults.update(overrides)
	return Simulations(**defaults)


# --- detectors ---


def test_detectors_default_is_all_three():
	assert _microscope().detectors == ["haadf", "abf", "bf"]


def test_detectors_accepts_subset():
	assert _microscope(detectors=["haadf", "abf"]).detectors == ["haadf", "abf"]
	assert _microscope(detectors=["haadf"]).detectors == ["haadf"]
	assert _microscope(detectors=["bf", "haadf"]).detectors == ["bf", "haadf"]


def test_detectors_normalizes_case():
	"""Mixed case in TOML should normalize to lowercase."""
	assert _microscope(detectors=["HAADF", "ABF"]).detectors == ["haadf", "abf"]
	assert _microscope(detectors=["HaAdF", "BF"]).detectors == ["haadf", "bf"]


def test_detectors_deduplicates():
	assert _microscope(detectors=["haadf", "haadf", "abf"]).detectors == ["haadf", "abf"]


def test_detectors_rejects_unknown():
	for bad in (["xyz"], ["haadf", "xyz"], [""], ["haa df"]):
		try:
			_microscope(detectors=bad)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for detectors={bad!r}")


def test_detectors_rejects_non_list():
	for bad in ("haadf", {"haadf": True}, 42):
		try:
			_microscope(detectors=bad)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for detectors={bad!r}")


# --- defaults flipped / added ---


def test_do_diffraction_defaults_false():
	"""Previously a required field; now defaults to False to match the
	worker-era 'opt in to diffraction' convention."""
	assert _microscope().do_diffraction is False


def test_do_cbed_defaults_false():
	"""New field split out of do_diffraction in the worker era."""
	assert _microscope().do_cbed is False


def test_do_cbed_can_be_set():
	assert _microscope(do_cbed=True).do_cbed is True


def test_test_enabled_defaults_false():
	"""New on Simulations. Diagnostic-only switch."""
	assert _sim().test_enabled is False


def test_test_enabled_can_be_set():
	assert _sim(test_enabled=True).test_enabled is True


# --- frozen_phonons validator (rejects 0 / negative / non-int strings) ---


def test_frozen_phonons_accepts_positive_int():
	assert _sim(frozen_phonons=1).frozen_phonons == 1
	assert _sim(frozen_phonons=8).frozen_phonons == 8


def test_frozen_phonons_accepts_none_string():
	assert _sim(frozen_phonons="None").frozen_phonons == "None"


def test_frozen_phonons_accepts_list_of_valid():
	assert _sim(frozen_phonons=[1, 2, "None"]).frozen_phonons == [1, 2, "None"]


def test_frozen_phonons_rejects_zero():
	"""0 would generate an empty seed queue; almost certainly a typo."""
	for bad in (0, -1, -5):
		try:
			_sim(frozen_phonons=bad)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for frozen_phonons={bad}")


def test_frozen_phonons_rejects_other_strings():
	for bad in ("none", "0", "1", ""):
		try:
			_sim(frozen_phonons=bad)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for frozen_phonons={bad!r}")


def test_frozen_phonons_rejects_bool():
	"""bool is a subclass of int in Python; explicitly rule it out."""
	for bad in (True, False):
		try:
			_sim(frozen_phonons=bad)
		except ValidationError:
			pass
		else:
			raise AssertionError(f"expected ValidationError for frozen_phonons={bad!r}")


def test_frozen_phonons_rejects_bad_list_entry():
	"""A list with one bad entry should fail validation."""
	try:
		_sim(frozen_phonons=[1, 0, 2])
	except ValidationError:
		pass
	else:
		raise AssertionError("expected ValidationError for list containing 0")


def _run_all():
	for fn in (
		test_detectors_default_is_all_three,
		test_detectors_accepts_subset,
		test_detectors_normalizes_case,
		test_detectors_deduplicates,
		test_detectors_rejects_unknown,
		test_detectors_rejects_non_list,
		test_do_diffraction_defaults_false,
		test_do_cbed_defaults_false,
		test_do_cbed_can_be_set,
		test_test_enabled_defaults_false,
		test_test_enabled_can_be_set,
		test_frozen_phonons_accepts_positive_int,
		test_frozen_phonons_accepts_none_string,
		test_frozen_phonons_accepts_list_of_valid,
		test_frozen_phonons_rejects_zero,
		test_frozen_phonons_rejects_other_strings,
		test_frozen_phonons_rejects_bool,
		test_frozen_phonons_rejects_bad_list_entry,
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
