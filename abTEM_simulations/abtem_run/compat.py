#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
abTEM compatibility patches for the tested local runtime.

These patches target known abTEM behavior in the pinned/tested environment.
They are not a strategy for supporting arbitrary abTEM internals. Prefer a
pinned environment first; update or remove these shims deliberately.

Patches applied are recorded in `_PATCHES_APPLIED` for diagnostics:
    >>> from abTEM_simulations.abtem_run.compat import apply_abtem_patches, _PATCHES_APPLIED
    >>> apply_abtem_patches()
    >>> _PATCHES_APPLIED

Patches:
  1. _partition_args_meta — TEMP: CuPy/Dask metadata workaround. Recheck after
     abTEM's CuPy-related review/fix.
  2. _gaussian_filter_boundary_modes — KEEP until upstreamed: abTEM wraps
     scipy.ndimage.gaussian_filter but restricts boundary modes too narrowly
     for BF-style blurred outputs. Candidate for an upstream PR.

The former _fft_dispatch_cufft_numpy patch is intentionally not applied: CPU-only
code paths now run under a temporary CPU/FFT config instead of teaching abTEM to
treat ``fft="cufft"`` as a valid NumPy FFT backend.

TODO: direct module commands such as
`python -m abTEM_simulations.abtem_run.worker ...` bypass `run.py` and may need
an explicit AWS/headless-Docker entrypoint that applies these patches before
importing runtime modules.
"""
import inspect
import textwrap
import warnings


_PATCHES_APPLIED: dict[str, bool] = {}
_PATCHES_ATTEMPTED = False


def _apply_substitution_patch(
	name: str,
	module,
	owner,
	attr_name: str,
	target: str,
	replacement: str,
) -> bool:
	"""Source-substitute one known abTEM function body."""
	import numpy as np

	try:
		fn = getattr(owner, attr_name)
		src_raw = inspect.getsource(fn)
	except (AttributeError, TypeError, OSError):
		_PATCHES_APPLIED[name] = False
		return False

	src = textwrap.dedent(src_raw)
	if target not in src:
		_PATCHES_APPLIED[name] = False
		return False

	patched_src = src.replace(target, replacement)
	ns = {**vars(module), "np": np}
	try:
		exec(patched_src, ns)
		setattr(owner, attr_name, ns[attr_name])
	except Exception as e:  # noqa: BLE001 - compatibility shim should warn, not mask
		warnings.warn(
			f"abtem_run: monkey-patch {name!r} failed to apply: {e!r}. "
			"Expect runtime errors on the code path this patch addresses.",
			stacklevel=2,
		)
		_PATCHES_APPLIED[name] = False
		return False

	_PATCHES_APPLIED[name] = True
	return True


def apply_abtem_patches() -> dict[str, bool]:
	"""Apply local abTEM compatibility patches once and return patch status."""
	global _PATCHES_ATTEMPTED
	if _PATCHES_ATTEMPTED:
		return dict(_PATCHES_APPLIED)
	_PATCHES_ATTEMPTED = True

	try:
		import abtem.array as _ab_array
	except ImportError:
		_PATCHES_APPLIED["_partition_args_meta"] = False
	else:
		_apply_substitution_patch(
			name="_partition_args_meta",
			module=_ab_array,
			owner=_ab_array.ArrayObject,
			attr_name="_partition_args",
			target="meta=xp.array((), object)",
			replacement="meta=np.array((), dtype=object)",
		)

	try:
		import abtem.measurements as _ab_meas
	except ImportError:
		_PATCHES_APPLIED["_gaussian_filter_boundary_modes"] = False
	else:
		_apply_substitution_patch(
			name="_gaussian_filter_boundary_modes",
			module=_ab_meas,
			owner=_ab_meas._BaseMeasurement2D,
			attr_name="gaussian_filter",
			target=(
				"    elif boundary in (\"reflect\", \"constant\"):\n"
				"        mode = boundary\n"
				"    else:\n"
				"        raise ValueError()"
			),
			replacement=(
				"    elif boundary in (\n"
				"        \"reflect\", \"constant\", \"nearest\", \"mirror\", \"wrap\",\n"
				"        \"grid-constant\", \"grid-mirror\", \"grid-wrap\",\n"
				"    ):\n"
				"        mode = boundary\n"
				"    else:\n"
				"        raise ValueError(\n"
				"            f\"unknown gaussian_filter boundary mode: {boundary!r}; \"\n"
				"            \"must be one of {periodic, reflect, constant, nearest, \"\n"
				"            \"mirror, wrap, grid-constant, grid-mirror, grid-wrap}\"\n"
				"        )"
			),
		)

	return dict(_PATCHES_APPLIED)


def bootstrap() -> None:
	"""Entry-point setup: apply the abTEM patches, then configure logging.

	Every console script and the ``run.py`` / ``python -m`` entries call this
	before any abTEM work, so the patches land no matter which entry point
	started the process. Idempotent — both steps guard against re-running.
	"""
	apply_abtem_patches()
	from ._log import configure_default_logging
	configure_default_logging()


__all__ = ["apply_abtem_patches", "bootstrap", "_PATCHES_APPLIED"]
