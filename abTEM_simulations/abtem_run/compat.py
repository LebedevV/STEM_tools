#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""abTEM compatibility shims and the policy for applying them.

These shims exist only because the pinned abTEM build lacks two behaviors this
package needs. They are a stopgap, not an architecture:

  1. _partition_args_meta — CuPy/Dask-safe partition metadata (GPU runs).
  2. _gaussian_filter_boundary_modes — a wider gaussian_filter boundary
     allow-list (e.g. 'nearest') for blurred outputs.

Policy:
  * A missing behavior is an ENVIRONMENT problem; the durable fix is upstreaming
    to abTEM (or pinning a build that has it), not patching around it.
  * Never apply the shims as an import side-effect, and never scatter patch calls
    across entry points. Importing this module changes nothing.
  * The user CLI (cli.main) applies them once, explicitly, behind a consent gate
    (ensure_patched_environment): detect what's missing, say why, and apply only
    on [y/N] / --apply-patches / ABTEM_RUN_APPLY_PATCHES.
  * Direct module / worker entries run UNPATCHED by design — the parallel/AWS
    path runs in a properly-patched abTEM environment (e.g. a Docker image).

Applied status is recorded in _PATCHES_APPLIED for diagnostics.
"""
import importlib
import inspect
import logging
import sys
import textwrap
import warnings


log = logging.getLogger("abtem_run")

_PATCHES_APPLIED: dict[str, bool] = {}
_PATCHES_ATTEMPTED = False


# One source of truth for both detection and application of each shim.
_PATCH_SPECS = (
	{
		"name": "_partition_args_meta",
		"module": "abtem.array",
		"owner": "ArrayObject",
		"attr": "_partition_args",
		"reason": "CuPy/Dask-safe partition metadata (GPU runs)",
		"target": "meta=xp.array((), object)",
		"replacement": "meta=np.array((), dtype=object)",
	},
	{
		"name": "_gaussian_filter_boundary_modes",
		"module": "abtem.measurements",
		"owner": "_BaseMeasurement2D",
		"attr": "gaussian_filter",
		"reason": "wider gaussian_filter boundary modes (e.g. 'nearest')",
		"target": (
			"    elif boundary in (\"reflect\", \"constant\"):\n"
			"        mode = boundary\n"
			"    else:\n"
			"        raise ValueError()"
		),
		"replacement": (
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
	},
)

_REASONS = {spec["name"]: spec["reason"] for spec in _PATCH_SPECS}


def _resolve_target(spec):
	"""Return (module, owner, dedented_source) for a spec, or None if the target
	function isn't importable/inspectable in this environment."""
	try:
		module = importlib.import_module(spec["module"])
		owner = getattr(module, spec["owner"])
		src = textwrap.dedent(inspect.getsource(getattr(owner, spec["attr"])))
	except (ImportError, AttributeError, TypeError, OSError):
		return None
	return module, owner, src


def detect_applicable_patches() -> list[str]:
	"""Names of shims whose target is present in the current abTEM (the
	environment is unpatched for that behavior) and not already applied.

	Pure inspection — modifies nothing.
	"""
	applicable: list[str] = []
	for spec in _PATCH_SPECS:
		if _PATCHES_APPLIED.get(spec["name"]):
			continue
		resolved = _resolve_target(spec)
		if resolved is not None and spec["target"] in resolved[2]:
			applicable.append(spec["name"])
	return applicable


def _apply_one(spec) -> bool:
	"""Source-substitute one known abTEM function body."""
	import numpy as np

	resolved = _resolve_target(spec)
	if resolved is None or spec["target"] not in resolved[2]:
		_PATCHES_APPLIED[spec["name"]] = False
		return False

	module, owner, src = resolved
	patched_src = src.replace(spec["target"], spec["replacement"])
	ns = {**vars(module), "np": np}
	try:
		exec(patched_src, ns)
		setattr(owner, spec["attr"], ns[spec["attr"]])
	except Exception as e:  # noqa: BLE001 - compatibility shim should warn, not mask
		warnings.warn(
			f"abtem_run: monkey-patch {spec['name']!r} failed to apply: {e!r}. "
			"Expect runtime errors on the code path this patch addresses.",
			stacklevel=2,
		)
		_PATCHES_APPLIED[spec["name"]] = False
		return False

	_PATCHES_APPLIED[spec["name"]] = True
	return True


def apply_abtem_patches() -> dict[str, bool]:
	"""Apply the abTEM compatibility shims once and return their status.

	Prefer ``ensure_patched_environment`` at the CLI; call this directly only when
	the decision to patch is already made (e.g. a Docker/AWS entrypoint).
	"""
	global _PATCHES_ATTEMPTED
	if _PATCHES_ATTEMPTED:
		return dict(_PATCHES_APPLIED)
	_PATCHES_ATTEMPTED = True
	for spec in _PATCH_SPECS:
		_apply_one(spec)
	return dict(_PATCHES_APPLIED)


def ensure_patched_environment(assume_yes: bool | None = None) -> None:
	"""Consent-gated application of the compat shims, for the user CLI.

	No-op when the current abTEM already has the behaviors. Otherwise:
	  * assume_yes=True  -> apply (--apply-patches / ABTEM_RUN_APPLY_PATCHES)
	  * assume_yes=False -> run against the bare environment, with a warning
	  * assume_yes=None  -> ask on a TTY; on a non-TTY, fail fast with guidance
	"""
	applicable = detect_applicable_patches()
	if not applicable:
		return
	missing = "; ".join(_REASONS[name] for name in applicable)

	if assume_yes is None:
		if not sys.stdin.isatty():
			raise SystemExit(
				f"env check failed: abTEM lacks compat behavior(s): {missing}.\n"
				"Re-run with --apply-patches (or ABTEM_RUN_APPLY_PATCHES=1), pass "
				"--no-patches to run anyway, or use a patched abTEM environment."
			)
		reply = input(
			f"env check: abTEM lacks compat behavior(s): {missing}.\n"
			"Apply local compat shims for this run? [y/N] "
		)
		assume_yes = reply.strip().lower() in ("y", "yes")

	if not assume_yes:
		log.warning(
			"running without abTEM compat shims (%s); "
			"expect failures on those code paths.",
			missing,
		)
		return

	apply_abtem_patches()
	landed = ", ".join(name for name, ok in _PATCHES_APPLIED.items() if ok)
	log.info(
		"applied abTEM compat shims: %s "
		"(environment lacked them; the durable fix is upstreaming to abTEM).",
		landed or "none",
	)


__all__ = [
	"apply_abtem_patches",
	"detect_applicable_patches",
	"ensure_patched_environment",
	"_PATCHES_APPLIED",
]
