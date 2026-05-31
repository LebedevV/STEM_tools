"""
Smoke test for V. Lebedev's stage-2 reproducibility goal:

    "Same TOML should produce the same atomic displacement arrays bit-for-bit"
    -- project-architecture-intent memory

Checks abtem.FrozenPhonons in isolation. If this fails, no downstream
checkpoint can be reproducible — the test is the floor that everything
else builds on.

Runnable two ways:

    PYTHONPATH=src python3 tests/test_phonon_reproducibility.py
    PYTHONPATH=src pytest tests/test_phonon_reproducibility.py
"""
from __future__ import annotations

import ase.build
import numpy as np

import abtem


def _snapshots(atoms, num_configs: int, sigmas: float, seed: int) -> list[np.ndarray]:
    fph = abtem.FrozenPhonons(
        atoms, num_configs=num_configs, sigmas=sigmas, seed=seed
    )
    ensemble = fph.to_atoms_ensemble()
    return [a.positions.copy() for a in ensemble.trajectory]


def _build_atoms():
    # Tiny diamond-Si supercell — enough atoms to exercise sigma displacement
    # without making the test slow. Choice of element and lattice is arbitrary;
    # all we care about is that displacements are reproducible.
    return ase.build.bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))


def test_same_seed_same_displacements():
    """Calling FrozenPhonons twice with identical args must produce identical positions."""
    atoms = _build_atoms()
    args = dict(num_configs=4, sigmas=0.1, seed=42)

    snaps_a = _snapshots(atoms, **args)
    snaps_b = _snapshots(atoms, **args)

    assert len(snaps_a) == len(snaps_b) == args["num_configs"]
    for i, (a, b) in enumerate(zip(snaps_a, snaps_b)):
        assert a.shape == b.shape, f"shape differs on snapshot {i}"
        assert np.array_equal(a, b), (
            f"snapshot {i} differs bit-for-bit; "
            f"max |Δ| = {np.max(np.abs(a - b))}"
        )


def test_different_seed_different_displacements():
    """Sanity check: different seeds should not collide (would mean seed is ignored)."""
    atoms = _build_atoms()
    snaps_a = _snapshots(atoms, num_configs=4, sigmas=0.1, seed=42)
    snaps_c = _snapshots(atoms, num_configs=4, sigmas=0.1, seed=43)

    # At least one snapshot must differ — if all match, the seed isn't actually wired.
    all_match = all(np.array_equal(a, c) for a, c in zip(snaps_a, snaps_c))
    assert not all_match, (
        "seed appears to be ignored: snapshots are identical for seed=42 and seed=43"
    )


def test_displacements_are_actually_applied():
    """sigmas>0 with a seed must move atoms away from their lattice positions."""
    atoms = _build_atoms()
    snaps = _snapshots(atoms, num_configs=2, sigmas=0.1, seed=42)
    rest = atoms.positions

    for i, s in enumerate(snaps):
        delta = np.max(np.abs(s - rest))
        assert delta > 1e-6, (
            f"snapshot {i} is bit-identical to rest positions — "
            f"sigma displacement not applied (max |Δ| = {delta})"
        )


def _run_all():
    for fn in (
        test_same_seed_same_displacements,
        test_different_seed_different_displacements,
        test_displacements_are_actually_applied,
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
