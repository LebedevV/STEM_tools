"""
Tests for the conditional abtem monkey-patch logic in abtem_run/__init__.py.

The current local abtem build is expected to need both patches (abtem
1.0.9-ish, where the original bugs live), so we just verify they're
recorded as applied. The more interesting test exercises the no-op path:
running the patch helper against a synthetic module whose source string
doesn't match — should leave the function untouched and record the patch
as not applied.

Runnable two ways:
    PYTHONPATH=src python3 tests/test_abtem_patches.py
    PYTHONPATH=src pytest tests/test_abtem_patches.py
"""
from __future__ import annotations

import types

from abtem_run import _PATCHES_APPLIED, _apply_substitution_patch


def test_patches_recorded_on_import():
	"""All three patches must be either applied or explicitly recorded as
	skipped. Just importing the package can't leave _PATCHES_APPLIED in an
	indeterminate state."""
	assert "_partition_args_meta" in _PATCHES_APPLIED
	assert "_fft_dispatch_cufft_numpy" in _PATCHES_APPLIED
	assert "_gaussian_filter_boundary_modes" in _PATCHES_APPLIED
	# Values must be plain bools (False = no-op, True = patched).
	for v in _PATCHES_APPLIED.values():
		assert isinstance(v, bool)


def test_patches_apply_against_current_abtem():
	"""On the dev environment the local abtem build matches the targets,
	so all three patches should land. If this fails locally, the abtem
	version has drifted and the patch targets need refresh — but it's not
	a correctness failure of the patch logic itself."""
	assert _PATCHES_APPLIED.get("_partition_args_meta") is True, (
		f"_partition_args_meta did not apply: {_PATCHES_APPLIED}"
	)
	assert _PATCHES_APPLIED.get("_fft_dispatch_cufft_numpy") is True, (
		f"_fft_dispatch_cufft_numpy did not apply: {_PATCHES_APPLIED}"
	)
	assert _PATCHES_APPLIED.get("_gaussian_filter_boundary_modes") is True, (
		f"_gaussian_filter_boundary_modes did not apply: {_PATCHES_APPLIED}"
	)


def test_patch_is_noop_when_target_absent():
	"""If the abtem source doesn't contain the string the patch is looking
	for, the patch must silently no-op (return False) and leave the
	function untouched."""

	# Build a synthetic module with a function whose source DOES NOT
	# contain the target string. The patch helper should return False
	# and not touch the function.
	module = types.ModuleType("fake_abtem")

	def f(x):
		return x + 1

	module.f = f
	original = module.f

	applied = _apply_substitution_patch(
		name="probe_noop",
		module=module,
		owner=module,
		attr_name="f",
		target="this_string_does_not_appear_in_f",
		replacement="...",
	)

	assert applied is False, "patch should have no-op'd on missing target"
	assert module.f is original, "function must not be replaced when target absent"


def _run_all():
	for fn in (
		test_patches_recorded_on_import,
		test_patches_apply_against_current_abtem,
		test_patch_is_noop_when_target_absent,
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
