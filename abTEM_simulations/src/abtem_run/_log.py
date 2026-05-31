"""
Logging setup for abtem_run.

Default behavior (when a CLI entry point calls ``configure_default_logging()``):
- INFO and above stream to stdout with a bare ``%(message)s`` formatter,
  matching the historical "everything goes through print()" output that
  external tooling (test harnesses, log scrapers, etc.) may rely on.
- Logger is namespaced under ``abtem_run`` so the package can be silenced
  or re-routed independently of any host application's own logging.

Environment override: ``ABTEM_RUN_LOG=debug`` (or ``warning`` / ``error``)
raises or lowers the threshold without touching code.

Library callers that want their own routing can either skip
``configure_default_logging`` entirely (no handlers, no output) or
configure handlers themselves on ``logging.getLogger("abtem_run")``.
"""
from __future__ import annotations

import logging
import os
import sys


_DEFAULT_FORMAT = "%(message)s"


def configure_default_logging(level: int | str | None = None) -> None:
	"""Attach a stdout handler to the ``abtem_run`` logger. Idempotent —
	repeated calls are no-ops.

	Args:
		level: integer log level (e.g. ``logging.INFO``), case-insensitive
		       string name (``"INFO"``), or ``None`` to read
		       ``ABTEM_RUN_LOG`` from the environment (default ``"INFO"``).
	"""
	logger = logging.getLogger("abtem_run")
	if logger.handlers:
		return

	if level is None:
		level = os.environ.get("ABTEM_RUN_LOG", "INFO").upper()
	if isinstance(level, str):
		level = getattr(logging, level.upper(), logging.INFO)

	handler = logging.StreamHandler(sys.stdout)
	handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
	logger.addHandler(handler)
	logger.setLevel(level)
	logger.propagate = False
