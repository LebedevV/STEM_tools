"""Sphinx configuration for abtem-run."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable so autodoc can introspect it.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))


# -- Project ----------------------------------------------------------------
project = "abtem-run"
author = "Vasily A. Lebedev"
copyright = "Vasily A. Lebedev — GPL-v3"

try:
	from importlib.metadata import version as _v
	release = _v("abtem-run")
except Exception:
	release = "0.1.0"
version = release


# -- General ----------------------------------------------------------------
extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",        # Google/Numpy docstring parsing
	"sphinx.ext.intersphinx",
	"sphinx.ext.viewcode",
	"myst_parser",                # Markdown source support
]

source_suffix = {
	".md": "markdown",
	".rst": "restructuredtext",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
	"members": True,
	"show-inheritance": True,
	"undoc-members": False,
}
# abtem-run depends on heavy scientific libs that take a while to import
# (abtem itself triggers our runtime monkey-patches). Don't mock — let
# the build actually exercise the import path so autodoc gets real
# type info.

# V. Lebedev's legacy docstrings (the older bits of simulation.py, the
# pre-pydantic config docs) are plain text with indentation, not
# reStructuredText. RST trips on them with "Unexpected indentation"
# and "Block quote ends without a blank line" warnings. Suppress those
# specifically — the rendered output still reads fine, and rewriting
# every legacy docstring is out of scope for the docs scaffolding.
suppress_warnings = [
	"docutils",
	"misc.highlighting_failure",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

intersphinx_mapping = {
	"python": ("https://docs.python.org/3", None),
	"numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = [
	"colon_fence",     # ::: blocks
	"deflist",         # term : definition
	"linkify",         # auto-link URLs
	"tasklist",        # - [ ] checkboxes
]
myst_heading_anchors = 3


# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"abtem-run {release}"
html_show_sourcelink = False
