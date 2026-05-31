"""
Tests for probe defocus + aberrations machinery (PR #8 / physics-fixes).

Background: a hardcoded ``defocus='scherzer'`` plus default ``C30=0`` made
Scherzer evaluate to 0 silently, giving an in-focus probe and the
"BF looks like DF" inverted-center artifact on thin crystals.

Current API:
- ``microscope.defocus`` accepts a number (Å) or the literal string
  ``'scherzer'``.
- ``microscope.aberrations`` is a pass-through dict to abtem's CTF.
- ``simulation.add_probe(ctx, potential, defocus=None)`` resolves
  ``'scherzer'`` to a number ourselves (abtem 1.0.9 reads C30 from
  partial state as it iterates the aberrations dict, so order matters).
  Warns when the resolved defocus is ``'scherzer'`` and C30=0 —
  applies to both ctx-driven and explicit-override paths.

Covers:
- pydantic ``Microscope`` validation (defocus magic, dict pass-through,
  rejection of defocus / unknown / non-numeric keys);
- ``add_probe`` warning + resolved defocus + aberration pass-through
  via ``probe.ctf``.
"""
from __future__ import annotations

import warnings

import pytest

import abtem
import ase
import abtem_run  # noqa: F401 — package import triggers monkey-patches
from abtem_run.config import Microscope
from abtem_run.simulation import add_probe


# --------------------------------------------------------------------------- #
# Module-level tiny potential for grid.match — atoms/geometry don't matter,
# only that the grid is well-defined.
# --------------------------------------------------------------------------- #


def _tiny_potential():
	atoms = ase.Atoms("Si", positions=[(1.0, 1.0, 1.0)], cell=(2.0, 2.0, 2.0), pbc=False)
	return abtem.Potential(atoms, gpts=(32, 32), slice_thickness=1.0)


_POT = _tiny_potential()


# --------------------------------------------------------------------------- #
# Pydantic — Microscope
# --------------------------------------------------------------------------- #


def _base_microscope(**overrides):
	"""Minimum kwargs to instantiate Microscope."""
	base = dict(
		HT_value=200000,
		haadfinner=99, haadfouter=200,
		abfinner=15, abfouter=33,
		bfinner=0.0, bfouter=9,
	)
	base.update(overrides)
	return base


def test_default_defocus_is_scherzer_string():
	m = Microscope(**_base_microscope())
	assert m.defocus == "scherzer"
	assert m.aberrations == {}


def test_defocus_accepts_numeric_angstrom():
	m = Microscope(**_base_microscope(defocus=-100.0))
	assert m.defocus == -100.0


def test_defocus_rejects_other_strings():
	with pytest.raises(Exception):
		Microscope(**_base_microscope(defocus="overfocus"))


def test_defocus_rejects_bool():
	with pytest.raises(Exception):
		Microscope(**_base_microscope(defocus=True))


def test_aberrations_pass_through():
	m = Microscope(**_base_microscope(aberrations={"C30": 1e7, "C12": 5.0}))
	assert m.aberrations == {"C30": 1e7, "C12": 5.0}


def test_aberrations_rejects_defocus_key():
	"""defocus must go through the top-level field, not the dict."""
	for bad_key in ("defocus", "C10"):
		with pytest.raises(Exception):
			Microscope(**_base_microscope(aberrations={bad_key: -100.0}))


def test_aberrations_rejects_non_numeric():
	with pytest.raises(Exception):
		Microscope(**_base_microscope(aberrations={"C30": "scherzer"}))


def test_aberrations_rejects_bool_values():
	"""bool is a subclass of int and dict[str, float] would silently
	coerce True -> 1.0. Enforce mode='before' so booleans never sneak in."""
	with pytest.raises(Exception):
		Microscope(**_base_microscope(aberrations={"C30": True}))


def test_aberrations_rejects_unknown_keys():
	"""Aberrations dict must only contain abtem-supported symbols (polar or
	named). Catches typos before the user runs."""
	with pytest.raises(Exception, match="not a known abtem"):
		Microscope(**_base_microscope(aberrations={"C20": 1e7}))
	with pytest.raises(Exception, match="not a known abtem"):
		Microscope(**_base_microscope(aberrations={"phi100": 0.5}))


def test_aberrations_accepts_named_aliases():
	"""abtem accepts both polar ('C30') and named ('Cs') aliases."""
	for key in ("C30", "Cs", "C12", "astigmatism", "phi12", "astigmatism_angle"):
		m = Microscope(**_base_microscope(aberrations={key: 1.0}))
		assert key in m.aberrations


# --------------------------------------------------------------------------- #
# add_probe — composing it all
# --------------------------------------------------------------------------- #


class _Ctx:
	"""Tiny RunContext shim for unit-test isolation (no CIF needed)."""
	def __init__(self, defocus="scherzer", aberrations=None, HT=200000, conv=30.0):
		self.defocus = defocus
		self.aberrations = aberrations or {}
		self.HT_value = HT
		self.convergence_angle = conv


def test_non_c30_aberration_reaches_probe_ctf():
	"""A non-C30 aberration (C12 twofold astigmatism + phi12 angle) must
	flow through add_probe to the resulting Probe.ctf."""
	probe = add_probe(_Ctx(defocus=0.0, aberrations={"C12": 50.0, "phi12": 0.5}), _POT)
	assert probe.ctf.aberration_coefficients["C12"] == 50.0
	assert probe.ctf.aberration_coefficients["phi12"] == 0.5


def test_non_c30_aberration_changes_probe():
	"""Smoke: adding a non-C30 aberration produces a different probe
	wavefunction. Catches the silently-dropped-coefficient regression."""
	import numpy as np
	probe_no_astig = add_probe(_Ctx(defocus=0.0, aberrations={}), _POT)
	probe_astig = add_probe(_Ctx(defocus=0.0, aberrations={"C12": 100.0}), _POT)
	a1 = np.asarray(probe_no_astig.build().compute().array)
	a2 = np.asarray(probe_astig.build().compute().array)
	assert not np.allclose(a1, a2), (
		"C12 aberration did not change the probe — pass-through is broken"
	)


def test_add_probe_warns_on_scherzer_with_zero_c30():
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		probe = add_probe(_Ctx(defocus="scherzer", aberrations={}), _POT)
	assert any("scherzer" in str(w.message).lower() for w in caught), (
		f"expected a scherzer-with-zero-C30 warning, got: "
		f"{[str(w.message) for w in caught]}"
	)
	# Probe still built (warning, not error).
	assert probe.ctf.defocus == 0.0


def test_add_probe_no_warning_when_c30_nonzero():
	import math
	from abtem.transfer import scherzer_defocus
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		probe = add_probe(_Ctx(defocus="scherzer", aberrations={"C30": 1e7}), _POT)
	relevant = [w for w in caught if "scherzer" in str(w.message).lower()]
	assert not relevant, f"unexpected scherzer warning: {[str(w.message) for w in relevant]}"
	# Scherzer at 200 kV / Cs=1 mm ~613 A — pin the exact formula value.
	expected = float(scherzer_defocus(1e7, 200000))
	assert math.isclose(probe.ctf.defocus, expected, rel_tol=1e-9)
	assert probe.ctf.defocus > 100.0


def test_add_probe_no_warning_for_explicit_zero_defocus():
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		probe = add_probe(_Ctx(defocus=0.0, aberrations={}), _POT)
	relevant = [w for w in caught if "scherzer" in str(w.message).lower()]
	assert not relevant
	assert probe.ctf.defocus == 0.0


def test_add_probe_passes_aberrations_through():
	"""C30 and a sample non-spherical aberration both surface on the CTF."""
	probe = add_probe(_Ctx(defocus=-50.0, aberrations={"C30": 1e7, "C12": 5.0}), _POT)
	assert probe.ctf.defocus == -50.0
	assert probe.ctf.C30 == 1e7
	assert probe.ctf.aberration_coefficients["C12"] == 5.0


def test_add_probe_order_independent_for_scherzer():
	"""abtem 1.0.9 had order-sensitive aberration parsing. add_probe
	resolves 'scherzer' before passing to abtem, so dict order can't
	affect the resolved defocus."""
	probe_a = add_probe(_Ctx(defocus="scherzer", aberrations={"C30": 1e7, "C12": 5.0}), _POT)
	probe_b = add_probe(_Ctx(defocus="scherzer", aberrations={"C12": 5.0, "C30": 1e7}), _POT)
	assert probe_a.ctf.defocus == probe_b.ctf.defocus
	assert probe_a.ctf.defocus > 100.0


def test_add_probe_explicit_defocus_override_also_warns():
	"""add_probe(ctx, pot, defocus='scherzer') with C30=0 fires the same
	warning the ctx-driven path does — an explicit 'scherzer' override
	is the same misconfiguration."""
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		# ctx.defocus = 0 (no warning normally), but defocus='scherzer'
		# explicit override + C30=0 should warn.
		add_probe(_Ctx(defocus=0.0, aberrations={}), _POT, defocus="scherzer")
	assert any("scherzer" in str(w.message).lower() for w in caught), (
		f"expected scherzer warning via explicit defocus, got: "
		f"{[str(w.message) for w in caught]}"
	)


def test_add_probe_no_warning_when_explicit_defocus_is_numeric():
	"""Numeric explicit defocus never warns, even if ctx would have."""
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		add_probe(_Ctx(defocus="scherzer", aberrations={}), _POT, defocus=-50.0)
	relevant = [w for w in caught if "scherzer" in str(w.message).lower()]
	assert not relevant, f"unexpected scherzer warning: {[str(w.message) for w in relevant]}"
