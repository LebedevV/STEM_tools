#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
abtem-run package init.

Applies two compatibility monkey-patches to abtem at import time, targeting
abtem 1.0.9 internals. Each patch is CONDITIONAL: if the source string to be
replaced isn't present (the abtem build doesn't need the patch, or its
internal layout changed), the patch is a no-op. This is what lets
`pip install abtem-run` work against a range of abtem builds rather than
only the exact 1.0.9 the author developed against.

Patches applied are recorded in `_PATCHES_APPLIED` for diagnostics:
    >>> from abtem_run import _PATCHES_APPLIED
    >>> _PATCHES_APPLIED
    {'_partition_args_meta': True, '_fft_dispatch_cufft_numpy': True}

Both patches:
  1. _partition_args_meta — cupy rejects `object` dtype in da.blockwise
     meta. Rewrite ArrayObject._partition_args to use np.array(..., dtype=object).
  2. _fft_dispatch_cufft_numpy — fft="cufft" + numpy array crashes
     _fft_dispatch's else branch. Add a numpy fallback before the raise.
"""
import inspect
import textwrap
import warnings


_PATCHES_APPLIED: dict[str, bool] = {}


def _apply_substitution_patch(
	name: str,
	module,
	owner,
	attr_name: str,
	target: str,
	replacement: str,
) -> bool:
	"""
	Rewrite `owner.<attr_name>` by source-substituting `target` -> `replacement`
	and re-exec'ing in the host module's namespace. Returns True if the patch
	landed, False if it was skipped (target not found / source unavailable /
	function missing on this abtem build).

	`module` is the abtem submodule whose namespace the rewritten function
	must execute in (so its free variables resolve correctly). `owner` is
	the class or module on which the attribute is reassigned — often equal
	to `module`, but for class methods (`ArrayObject._partition_args`) it's
	the class.

	`np` is injected into the exec namespace so the replacement code can use
	`np.array(...)` and `np.fft.<func>(...)` regardless of whether the host
	module imported numpy under that name.
	"""
	import numpy as np  # local import: only needed when actually patching

	try:
		fn = getattr(owner, attr_name)
		src_raw = inspect.getsource(fn)
	except (AttributeError, TypeError, OSError):
		# function gone / not introspectable on this abtem build
		_PATCHES_APPLIED[name] = False
		return False

	src = textwrap.dedent(src_raw)
	if target not in src:
		# abtem doesn't have the broken line we expected — no work to do
		_PATCHES_APPLIED[name] = False
		return False

	patched_src = src.replace(target, replacement)
	ns = {**vars(module), "np": np}
	try:
		exec(patched_src, ns)
		patched_fn = ns[attr_name]
		setattr(owner, attr_name, patched_fn)
	except Exception as e:  # noqa: BLE001 — narrow logging, not a silent swallow
		warnings.warn(
			f"abtem-run: monkey-patch {name!r} failed to apply: {e!r}. "
			"Falling back to the unpatched function; expect "
			"runtime errors on the code path this patch addresses.",
			stacklevel=2,
		)
		_PATCHES_APPLIED[name] = False
		return False

	_PATCHES_APPLIED[name] = True
	return True


def _apply_all_patches() -> None:
	"""Import abtem submodules lazily so failure to import doesn't poison
	the entire package import — we still want config / Job / load_config
	to be usable in environments without abtem (e.g. unit tests of the
	schema)."""
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
		import abtem.core.fft as _ab_fft
	except ImportError:
		_PATCHES_APPLIED["_fft_dispatch_cufft_numpy"] = False
	else:
		_apply_substitution_patch(
			name="_fft_dispatch_cufft_numpy",
			module=_ab_fft,
			owner=_ab_fft,
			attr_name="_fft_dispatch",
			target="        else:\n            raise RuntimeError()",
			replacement=(
				"        elif config.get(\"fft\") == \"cufft\":\n"
				"            return getattr(np.fft, func_name)(x, **kwargs)\n"
				"        else:\n"
				"            raise RuntimeError()"
			),
		)


_apply_all_patches()


# --------------------------------------------------------------------------- #
# Public library API
#
# Re-export the most-used names so callers can do `from abtem_run import X`
# instead of digging through submodule paths. These imports MUST come after
# the monkey-patches above — any abtem usage triggered by the submodule
# inits is then already patched. (Importing the package now requires abtem;
# for schema-only use without abtem, import abtem_run.config directly.)
# --------------------------------------------------------------------------- #

from .config import AppConfig, Job, load_config
from .simulation import (
	add_probe,
	add_scan,
	add_vacancies,
	make_lamella,
)
from .pipeline import (
	RunContext,
	expand_cfg,
	make_potential,
	resolve_context,
)
from .generator_run import generate_run
from .worker import run_one_seed
from .aggregate import aggregate_job
from .to_ensemble import load_ensemble, to_ensemble_files
from .cli import main, run_pipeline

__all__ = [
	# config
	"AppConfig",
	"Job",
	"load_config",
	# simulation (geometry)
	"add_probe",
	"add_scan",
	"add_vacancies",
	"make_lamella",
	# pipeline (shared infrastructure for the worker path)
	"RunContext",
	"expand_cfg",
	"make_potential",
	"resolve_context",
	# worker pipeline
	"generate_run",
	"run_one_seed",
	"aggregate_job",
	# abTEM cross-compat bridge (per-seed zarrs -> ensemble Measurement)
	"load_ensemble",
	"to_ensemble_files",
	# convenience wrapper
	"main",
	"run_pipeline",
	# diagnostics
	"_PATCHES_APPLIED",
]
