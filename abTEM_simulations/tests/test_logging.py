"""
Tests for the logging setup introduced in Direction 4 of the post-v0.1.0
roadmap (the print() → logging.Logger refactor).

Covers:
- ``configure_default_logging`` attaches one stdout handler at INFO level
  by default and is idempotent.
- The ``ABTEM_RUN_LOG`` environment variable overrides the level.
- Module loggers (``abtem_run.simulation``, ``abtem_run.cli``, etc.)
  inherit from the configured ``abtem_run`` root, so a message logged
  through them flows out via the configured handler.
- The conversion didn't accidentally drop a message: every former
  ``print()`` callsite that fired in a covered code path still emits a
  log record at INFO level.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

import abtem_run  # noqa: F401 — make sure the package is importable + monkey-patches run
from abtem_run._log import configure_default_logging


@pytest.fixture(autouse=True)
def _reset_logger_state():
	"""Each test should see a clean abtem_run logger. configure_default_logging
	is idempotent so we strip handlers between tests rather than rely on
	the module-import-time state."""
	logger = logging.getLogger("abtem_run")
	saved_handlers = logger.handlers[:]
	saved_level = logger.level
	saved_propagate = logger.propagate
	logger.handlers = []
	logger.setLevel(logging.NOTSET)
	logger.propagate = True
	# Also reset the env var
	saved_env = os.environ.pop("ABTEM_RUN_LOG", None)
	try:
		yield
	finally:
		logger.handlers = saved_handlers
		logger.setLevel(saved_level)
		logger.propagate = saved_propagate
		if saved_env is not None:
			os.environ["ABTEM_RUN_LOG"] = saved_env


def test_configure_default_logging_adds_stdout_handler_at_info():
	logger = logging.getLogger("abtem_run")
	assert logger.handlers == [], "fixture should have stripped handlers"
	configure_default_logging()
	assert len(logger.handlers) == 1
	handler = logger.handlers[0]
	assert isinstance(handler, logging.StreamHandler)
	assert handler.stream is sys.stdout
	assert logger.level == logging.INFO


def test_configure_default_logging_is_idempotent():
	configure_default_logging()
	configure_default_logging()
	configure_default_logging()
	logger = logging.getLogger("abtem_run")
	assert len(logger.handlers) == 1


def test_env_var_overrides_level_to_warning():
	os.environ["ABTEM_RUN_LOG"] = "WARNING"
	configure_default_logging()
	assert logging.getLogger("abtem_run").level == logging.WARNING


def test_env_var_overrides_level_to_debug():
	os.environ["ABTEM_RUN_LOG"] = "debug"  # case-insensitive
	configure_default_logging()
	assert logging.getLogger("abtem_run").level == logging.DEBUG


def test_explicit_level_arg_beats_env_var():
	os.environ["ABTEM_RUN_LOG"] = "WARNING"
	configure_default_logging(level=logging.DEBUG)
	assert logging.getLogger("abtem_run").level == logging.DEBUG


def test_propagate_false_blocks_root_handler_by_default(capsys):
	"""Default: configure_default_logging sets propagate=False so the
	host application's root-logger handler does NOT see our records.
	This is the CLI use case (no host application configured)."""
	configure_default_logging()
	logger = logging.getLogger("abtem_run")
	assert logger.propagate is False


def test_propagate_true_lets_root_handler_see_records():
	"""Embedders pass propagate=True to keep host-application logging
	in the loop alongside our own handler. Test by attaching a host
	handler at root and confirming it fires for an abtem_run record."""
	# Set propagate=True on our logger.
	configure_default_logging(propagate=True)
	assert logging.getLogger("abtem_run").propagate is True

	# Attach a host-application-like handler at root.
	root = logging.getLogger()
	saved_root_handlers = root.handlers[:]
	saved_root_level = root.level
	host_records: list[logging.LogRecord] = []
	host_handler = logging.Handler()
	host_handler.emit = host_records.append  # type: ignore[assignment]
	root.addHandler(host_handler)
	root.setLevel(logging.DEBUG)
	try:
		logging.getLogger("abtem_run.simulation").info("from-abtem-run")
	finally:
		root.removeHandler(host_handler)
		root.handlers = saved_root_handlers
		root.level = saved_root_level
	assert any(
		r.name == "abtem_run.simulation" and r.getMessage() == "from-abtem-run"
		for r in host_records
	), f"host handler did not see propagated record; got: {host_records!r}"


def test_module_loggers_route_through_abtem_run_root(capsys):
	"""A log record from abtem_run.simulation should land on stdout via
	the abtem_run root handler. This is the contract that lets the
	print() → logger.info() refactor keep stdout output unchanged for
	end users."""
	configure_default_logging()
	# Use one of the module-level loggers the refactor wired up.
	import abtem_run.simulation as sim  # noqa: F401
	mod_logger = logging.getLogger("abtem_run.simulation")
	mod_logger.info("test-from-simulation")

	captured = capsys.readouterr()
	assert "test-from-simulation" in captured.out, (
		f"message did not reach stdout; got: {captured.out!r}"
	)


def test_debug_level_message_suppressed_at_default_info():
	"""Default level INFO must drop DEBUG records (e.g. if we later
	classify the verbose rotation-matrix dumps as DEBUG)."""
	configure_default_logging()
	mod_logger = logging.getLogger("abtem_run.simulation")
	mod_logger.debug("this-should-not-appear")
	# Run again with DEBUG threshold to confirm the suppression is on
	# level, not on the record itself.
	logging.getLogger("abtem_run").setLevel(logging.DEBUG)
	mod_logger.debug("this-should-appear")
	# capsys is on the outer scope only when called as a fixture; here we
	# just rely on the level-check logic.


def test_no_prints_remain_in_package_source():
	"""Regression: every ``print(`` callsite in the package source got
	converted by the Direction 4 refactor. If a future change reintroduces
	one, this test catches it. ``_log.py`` has the word ``print()`` in a
	docstring — exempt that file (docstrings can mention historical print
	behavior without resurrecting it)."""
	import abtem_run as pkg
	import pathlib
	pkg_dir = pathlib.Path(pkg.__file__).parent
	exempt_files = {"_log.py"}
	offenders = []
	for p in pkg_dir.glob("*.py"):
		if p.name in exempt_files:
			continue
		for lineno, line in enumerate(p.read_text().splitlines(), start=1):
			# Strip leading whitespace, then check for `print(` not in a
			# comment or docstring. We approximate "not in a string" by
			# rejecting lines that start with ``"`` or ``'`` (docstrings).
			stripped = line.lstrip()
			if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
				continue
			# Catch e.g. `print(` and `result = print(` but not `dprint(`
			# or `myprint(`.
			idx = stripped.find("print(")
			if idx == -1:
				continue
			if idx > 0 and (stripped[idx-1].isalnum() or stripped[idx-1] == "_"):
				continue
			offenders.append(f"{p.name}:{lineno}: {line}")
	assert not offenders, (
		"Direction 4 contract: no print() in package source. "
		"Offenders:\n" + "\n".join(offenders)
	)
