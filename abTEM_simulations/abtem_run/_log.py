"""Default stdout logging for local abtem_run scripts.

Set ``ABTEM_RUN_LOG=debug|info|warning|error`` to change verbosity.
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
