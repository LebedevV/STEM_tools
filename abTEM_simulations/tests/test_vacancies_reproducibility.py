"""
Reproducibility test for add_vacancies — closes V. Lebedev's last
'seed transparency' hole. Vacancies live on the first-checkpoint
(atomic coordinates) of the three-checkpoint architecture, so they
must be bit-for-bit reproducible given the same seed.

Runnable two ways:
    PYTHONPATH=src python3 tests/test_vacancies_reproducibility.py
    PYTHONPATH=src pytest tests/test_vacancies_reproducibility.py
"""
from __future__ import annotations

import ase.build
import numpy as np

from abtem_run.simulation import add_vacancies


_TARGET_EL = "Cl"


def _build():
	# Tiny mixed-element supercell so we have atoms to drop (Cl) AND atoms
	# to keep (Na). NaCl rocksalt repeated 3x is 54 atoms total (27 Na + 27 Cl).
	atoms = ase.build.bulk("NaCl", crystalstructure="rocksalt", a=5.64)
	return atoms.repeat((3, 3, 3))


def test_same_seed_same_vacancy_pattern():
	"""Same seed + same surf + same (el, prob) -> bit-identical output."""
	atoms = _build()
	a = add_vacancies(atoms, _TARGET_EL, 0.5, seed=42)
	b = add_vacancies(atoms, _TARGET_EL, 0.5, seed=42)

	assert len(a) == len(b), f"atom counts differ: {len(a)} vs {len(b)}"
	assert a.get_chemical_symbols() == b.get_chemical_symbols()
	assert np.array_equal(a.positions, b.positions), (
		f"positions differ; max |Δ| = {np.max(np.abs(a.positions - b.positions))}"
	)


def test_different_seeds_diverge():
	"""seed=42 vs seed=43 should drop different atoms (with very high probability)."""
	atoms = _build()
	a = add_vacancies(atoms, _TARGET_EL, 0.5, seed=42)
	c = add_vacancies(atoms, _TARGET_EL, 0.5, seed=43)

	# With ~27 Cl atoms at p=0.5, the chance of both seeds producing identical
	# masks is ~2^-27. If it happens, the seed isn't actually wired.
	assert len(a) != len(c) or not np.array_equal(a.positions, c.positions), (
		"seed appears to be ignored: identical output for seed=42 and seed=43"
	)


def test_default_seed_is_zero_and_reproducible():
	"""Omitting `seed` must default to 0, and that path must be reproducible too."""
	atoms = _build()
	a = add_vacancies(atoms, _TARGET_EL, 0.5)         # default seed
	b = add_vacancies(atoms, _TARGET_EL, 0.5, seed=0)  # explicit seed=0

	assert len(a) == len(b)
	assert np.array_equal(a.positions, b.positions), "default seed differs from seed=0"


def test_prob_zero_drops_nothing():
	"""prob=0 must leave the atoms object untouched."""
	atoms = _build()
	out = add_vacancies(atoms, _TARGET_EL, 0.0, seed=42)
	assert len(out) == len(atoms)


def test_prob_one_drops_all_matching():
	"""prob=1 must remove every atom of the target element, none of the rest."""
	atoms = _build()
	n_target_before = sum(1 for s in atoms.get_chemical_symbols() if s == _TARGET_EL)
	out = add_vacancies(atoms, _TARGET_EL, 1.0, seed=42)
	n_target_after = sum(1 for s in out.get_chemical_symbols() if s == _TARGET_EL)
	assert n_target_before > 0, f"test setup didn't produce any {_TARGET_EL} atoms"
	assert n_target_after == 0, f"prob=1 left {n_target_after} {_TARGET_EL} atoms in place"
	# Non-target atoms must be untouched.
	expected_kept = len(atoms) - n_target_before
	assert len(out) == expected_kept


def _run_all():
	for fn in (
		test_same_seed_same_vacancy_pattern,
		test_different_seeds_diverge,
		test_default_seed_is_zero_and_reproducible,
		test_prob_zero_drops_nothing,
		test_prob_one_drops_all_matching,
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
