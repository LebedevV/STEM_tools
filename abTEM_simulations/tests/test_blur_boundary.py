"""
Tests for the simulations.blur_boundary config field + the monkey-patch
that broadens abtem.Images.gaussian_filter's allowed boundary modes
(`_gaussian_filter_boundary_modes` in src/abtem_run/__init__.py).

V. Lebedev picked 'nearest' as the new default on 2026-05-20 (Q2 answer);
abtem 1.0.9 doesn't natively support 'nearest', so the patch is the only
reason this works.

Runnable two ways:

    PYTHONPATH=src python3 tests/test_blur_boundary.py
    PYTHONPATH=src pytest tests/test_blur_boundary.py
"""
from __future__ import annotations

import numpy as np

import abtem
import abtem_run  # noqa: F401 — triggers monkey-patches at import


def _img():
	return abtem.Images(np.random.default_rng(0).random((16, 16)).astype("float32"), sampling=(0.1, 0.1))


def test_patch_applied():
	"""The boundary-modes patch must land against this abtem build."""
	assert abtem_run._PATCHES_APPLIED.get("_gaussian_filter_boundary_modes") is True, (
		f"_gaussian_filter_boundary_modes patch missing/failed; "
		f"_PATCHES_APPLIED = {abtem_run._PATCHES_APPLIED!r}"
	)


def test_nearest_mode_accepted():
	"""Previously raised bare ValueError(); patch makes it work."""
	out = _img().gaussian_filter(0.1, boundary="nearest")
	assert out.array.shape == (16, 16)
	assert np.all(np.isfinite(out.array))


def test_constant_mode_still_works():
	"""Don't regress the modes abtem already accepted."""
	out = _img().gaussian_filter(0.1, boundary="constant")
	assert out.array.shape == (16, 16)


def test_unknown_mode_raises_with_message():
	"""Rejection path now carries a useful message (was bare ValueError())."""
	try:
		_img().gaussian_filter(0.1, boundary="banana")
	except ValueError as e:
		assert "banana" in str(e), f"expected mode name in message, got {e!r}"
	else:
		raise AssertionError("'banana' should have been rejected")


def test_config_default_is_nearest():
	"""V.'s pick: nearest is the default; constant remains reachable."""
	from abtem_run.config import AppConfig

	# Build a minimal AppConfig-equivalent payload to exercise the default.
	# Easier: use the AppConfig defaults via model_fields.
	field = AppConfig.model_fields["simulations"].annotation.model_fields["blur_boundary"]
	assert field.default == "nearest", f"expected 'nearest' default, got {field.default!r}"


def _run_all():
	for fn in (
		test_patch_applied,
		test_nearest_mode_accepted,
		test_constant_mode_still_works,
		test_unknown_mode_raises_with_message,
		test_config_default_is_nearest,
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
