"""
Tombstone for the retired ``simulations.legacy_phonons`` flag.

The codebase used to support two phonon-seed conventions:

  - Independent (the only mode now): per-seed ``FrozenPhonons(num_configs=1,
    seed=K_i)``. The integer in the seed_NNNNNN.todo filename IS the seed.
  - Batched (legacy, retired): one shared ``FrozenPhonons(num_configs=N,
    seed=base).trajectory[idx]``. All N draws shared one RNG state.

For idx=0 they match (first draw from same seed). For idx>0 they diverge
(batched picks the post-RNG-state snapshot, independent uses a fresh seed).
This test pins the divergence so any future revival of batched mode comes
with explicit acknowledgement that it is NOT byte-equivalent to current
behavior.

    PYTHONPATH=src python3 tests/phonons/test_independent_vs_batched_divergence.py
    PYTHONPATH=src pytest tests/phonons/test_independent_vs_batched_divergence.py
"""
from __future__ import annotations

import abtem
import ase.build
import numpy as np


def _tiny_atoms():
	# NaCl supercell — ~64 atoms × 3 components × σ=0.1 makes accidental
	# divergence-collisions astronomically unlikely.
	return ase.build.bulk("NaCl", crystalstructure="rocksalt", a=5.64).repeat((2, 2, 2))


def _independent(atoms, seed: int):
	return abtem.FrozenPhonons(
		atoms, num_configs=1, sigmas=0.1, seed=seed,
	).to_atoms_ensemble().trajectory[0]


def _batched_trajectory(atoms, base_seed: int, n: int):
	return abtem.FrozenPhonons(
		atoms, num_configs=n, sigmas=0.1, seed=base_seed,
	).to_atoms_ensemble().trajectory


def test_idx_zero_matches_between_modes():
	"""First draw from seed K matches between conventions."""
	atoms = _tiny_atoms()
	K = 100
	batched = _batched_trajectory(atoms, base_seed=K, n=8)
	indep = _independent(atoms, seed=K)
	assert np.array_equal(batched[0].positions, indep.positions), (
		"idx=0 must match — first draw from the same seed in both conventions"
	)


def test_idx_one_diverges_between_modes():
	"""idx=1 batched (post-one-draw RNG state) != independent at seed=K+1.

	Both are valid Monte-Carlo samples but they hit different atomic
	configurations. Retiring legacy mode means losing byte-equivalence
	with pre-v6 outputs; this assertion makes that explicit.
	"""
	atoms = _tiny_atoms()
	K = 100
	batched = _batched_trajectory(atoms, base_seed=K, n=8)
	indep_K1 = _independent(atoms, seed=K + 1)
	assert not np.array_equal(batched[1].positions, indep_K1.positions), (
		"idx>0 unexpectedly matched — batched and independent modes should "
		"draw from different RNG states once past the first snapshot"
	)


def _run_all():
	for fn in (
		test_idx_zero_matches_between_modes,
		test_idx_one_diverges_between_modes,
	):
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {fn.__name__}: {e}")
			return 1
		else:
			print(f"PASS  {fn.__name__}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
