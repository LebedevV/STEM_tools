#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Compatibility fixes for the pinned abTEM 1.0.9 dependency.

Why these fixes exist:

1. ArrayObject._partition_args creates Dask metadata with xp.array((), object).
   On the GPU, xp is CuPy, which does not support object-dtype arrays. This is
   partition metadata, not the simulated wave or potential, so a NumPy object
   array supplies the metadata without moving the numerical calculation to CPU.

2. _BaseMeasurement2D.gaussian_filter accepts only reflect/constant in its
   non-periodic boundary branch. Our finite-source blur also uses nearest/wrap.
   The underlying filtering backend supports these modes, but abTEM rejects them
   before the filter runs. We extend that allow-list and retain the separate
   periodic branch. Boundary choice affects image edges; the fix enables the
   requested choice rather than changing the multislice calculation.

These are targeted source substitutions, not general fixes for arbitrary abTEM
versions. An absent target can mean either an existing fix or changed upstream
code; persistent patching checks for the replacement before treating it as fixed.
Revisit these workarounds when changing abTEM versions and remove them once the
upstream functions provide the required behavior.

Two application methods are retained:

* python -m abtem_run.compat asks before patch_abtem_source() rewrites the installed
  third-party abTEM files. Run it once per Python environment so separately started
  workers and aggregators inherit the fixes. Reinstalling abTEM can erase them.
  Start a new Python process afterwards; already-imported functions do not change.
* apply_abtem_patches() changes functions in memory for this process only. The
  local run.py driver offers this option when persistent fixes are absent.

Importing this module never patches anything. These scripts themselves do not
need installation. _PATCHES_APPLIED records only the in-memory patch status.
"""
import importlib
import inspect
import logging
import os
from pathlib import Path
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
	the caller has already chosen to apply the fixes.
	"""
	global _PATCHES_ATTEMPTED
	if _PATCHES_ATTEMPTED:
		return dict(_PATCHES_APPLIED)
	_PATCHES_ATTEMPTED = True
	for spec in _PATCH_SPECS:
		_apply_one(spec)
	return dict(_PATCHES_APPLIED)


def patch_abtem_source() -> dict[str, bool]:
	"""Persist the fixes in installed abTEM source for future Python processes.

	The caller chooses whether to modify the environment; the module CLI asks first.
	Check every target before writing. Already-patched functions are accepted, while
	unknown source is rejected so an abTEM version change cannot silently skip a fix.
	Each file is published with an atomic rename, preserving its permission bits.
	"""
	results = {}
	edits = []
	for spec in _PATCH_SPECS:
		module = importlib.import_module(spec["module"])
		fn = getattr(getattr(module, spec["owner"]), spec["attr"])
		filename = inspect.getsourcefile(fn)
		lines, start = inspect.getsourcelines(fn)
		source = textwrap.dedent("".join(lines))
		if spec["replacement"] in source:
			results[spec["name"]] = True
			continue
		if filename is None or spec["target"] not in source:
			raise RuntimeError(f"Unrecognized abTEM source for {spec['name']}; no files changed.")
		path = Path(filename)
		indent = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
		patched = textwrap.indent(source.replace(spec["target"], spec["replacement"]), indent)
		file_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
		file_lines[start - 1:start - 1 + len(lines)] = [patched]
		edits.append((spec["name"], path, "".join(file_lines)))

	for name, path, source in edits:
		tmp = path.with_suffix(path.suffix + ".tmp")
		try:
			tmp.write_text(source, encoding="utf-8")
			tmp.chmod(path.stat().st_mode)
			os.replace(tmp, path)
		finally:
			tmp.unlink(missing_ok=True)
		results[name] = True
	return results


def ensure_patched_environment(assume_yes: bool | None = None) -> None:
	"""Consent-gated application of the compat shims, for the user CLI.

	No-op when the current abTEM already has the behaviors. Otherwise:
	  * assume_yes=True  -> apply (--apply-patches / ABTEM_RUN_APPLY_PATCHES)
	  * assume_yes=False -> run against the bare environment, with a warning (--no-patches)
	  * assume_yes=None  -> ask on a TTY (declining aborts the run); non-TTY fails fast
	"""
	applicable = detect_applicable_patches()
	if not applicable:
		return
	missing = "; ".join(_REASONS[name] for name in applicable)
	# These changes handle partition metadata and enable the requested blur boundary mode.
	hint = (
		f"abTEM is missing: {missing}.\n"
		"These fixes handle GPU partition metadata and blur boundary modes. "
		"They apply only to this process (ABTEM_RUN_APPLY_PATCHES=1 skips this prompt)."
	)

	if assume_yes is None:
		if not sys.stdin.isatty():
			raise SystemExit(
				f"env check failed: {hint}\nRe-run with --apply-patches "
				"(or ABTEM_RUN_APPLY_PATCHES=1), or --no-patches to run unpatched anyway."
			)
		if input(f"{hint}\nApply for this run? [y/N] ").strip().lower() not in ("y", "yes"):
			# fail fast on an interactive decline -- proceeding would only crash later in
			# the run. (--no-patches is the explicit "run unpatched anyway".)
			raise SystemExit(
				"aborting: these shims are needed for this run. Re-run and answer y, pass "
				"--apply-patches, or --no-patches to run unpatched anyway."
			)
		assume_yes = True

	if not assume_yes:  # explicit --no-patches
		log.warning(
			"running without abTEM compat shims (%s); expect failures on those code paths.",
			missing,
		)
		return

	apply_abtem_patches()
	landed = ", ".join(name for name, ok in _PATCHES_APPLIED.items() if ok)
	log.info(
		"applied abTEM compat shims: %s (the durable fix is upstreaming to abTEM).",
		landed or "none",
	)


def warn_if_unpatched() -> None:
	"""Warn when a direct module call needs compatibility fixes."""
	applicable = detect_applicable_patches()
	if applicable:
		log.warning(
			"abTEM is missing %s. Run python -m abtem_run.compat once in this environment, "
			"then restart the worker or aggregator. Alternatively, python run.py offers "
			"in-memory fixes for its own process.",
			"; ".join(_REASONS[name] for name in applicable),
		)


__all__ = [
	"apply_abtem_patches",
	"patch_abtem_source",
	"detect_applicable_patches",
	"ensure_patched_environment",
	"warn_if_unpatched",
	"_PATCHES_APPLIED",
]


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--yes", action="store_true", help="apply source fixes without prompting")
	args = parser.parse_args()
	print("This modifies the installed third-party abTEM source in this Python environment:")
	for spec in _PATCH_SPECS:
		print(f"  {spec['module']}.{spec['owner']}.{spec['attr']}: {spec['reason']}")
	if not args.yes:
		if not sys.stdin.isatty():
			parser.error("Use --yes to explicitly approve source changes in a non-interactive run.")
		if input("Apply these persistent fixes? [y/N] ").strip().lower() not in ("y", "yes"):
			raise SystemExit("No changes made.")
	try:
		status = patch_abtem_source()
	except (ImportError, AttributeError, TypeError, OSError, RuntimeError) as error:
		raise SystemExit(f"Could not finish patching abTEM: {error}. Correct the issue and rerun.")
	for name in status:
		print(f"  {name}: fixed or already present")
	print("Start a new Python process for simulations. Recheck after reinstalling or upgrading abTEM.")
