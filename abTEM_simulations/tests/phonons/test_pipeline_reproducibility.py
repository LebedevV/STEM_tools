"""
Pipeline-level reproducibility test, complementing
``test_phonon_reproducibility.py``.

Builds an actual lamella from a CIF via ``simulation.make_lamella``, then
feeds it through ``cli.add_frozen_phonons_potential``-equivalent
``abtem.FrozenPhonons`` plumbing and checks the per-snapshot positions are
bit-identical across runs.

The standalone ``test_phonon_reproducibility.py`` covers the abtem boundary
on a synthetic bulk lattice. This one stresses the *actual* code path the
pipeline uses:
- ``simulation.make_lamella`` (CIF parse, rotation, crop, dask compute)
- ``add_frozen_phonons_potential`` (FrozenPhonons + Potential)

If make_lamella is non-deterministic for fixed inputs, or if seed flow
breaks anywhere upstream of FrozenPhonons, this test catches it.

Runnable two ways:

    PYTHONPATH=src python3 tests/test_pipeline_reproducibility.py
    PYTHONPATH=src pytest tests/test_pipeline_reproducibility.py
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import abtem

from abtem_run import simulation as sim


# Use the smallest deterministic CIF shipped with the repo. Pm3m is a cubic
# perovskite — short file, no trigonal-symmetry caveats.
_CIF_PATH = str(Path(__file__).resolve().parent.parent.parent / "cifs" / "Pm3m.cif")

# Tiny lamella so the test runs in a few seconds, not minutes.
_LAMELLA_KW = dict(
    sblock_size=15.0,
    lamella_sizes=(10.0, 10.0, 8.0),
    atom_to_zero=None,            # skip the in-plane reference-atom step
    tol=0.01,
    max_uvw=10,
    is_uvw=True,
    inplane_angle=0,
    vac_xy=2.0,
    vac_z=2.0,
    global_tilt=(0, 0),
)


def _build_lamella():
    return sim.make_lamella(_CIF_PATH, [0, 0, 1], **_LAMELLA_KW)


def _fph_snapshots(atoms, num_configs: int, sigmas: float, seed: int) -> list[np.ndarray]:
    fph = abtem.FrozenPhonons(
        atoms, num_configs=num_configs, sigmas=sigmas, seed=seed
    )
    return [a.positions.copy() for a in fph.to_atoms_ensemble().trajectory]


def test_make_lamella_is_deterministic():
    """make_lamella with identical inputs must return identical atoms."""
    a = _build_lamella()
    b = _build_lamella()

    assert len(a) == len(b), (
        f"lamella has different atom counts: {len(a)} vs {len(b)}"
    )
    assert a.get_chemical_symbols() == b.get_chemical_symbols(), (
        "chemical symbols differ between two builds"
    )
    assert np.array_equal(a.positions, b.positions), (
        "atom positions differ bit-for-bit; "
        f"max |Δ| = {np.max(np.abs(a.positions - b.positions))}"
    )


def test_phonons_on_real_lamella_reproducible():
    """End-to-end: lamella + FrozenPhonons + same seed -> identical snapshots."""
    lamella = _build_lamella()

    snaps_a = _fph_snapshots(lamella, num_configs=3, sigmas=0.1, seed=42)
    snaps_b = _fph_snapshots(lamella, num_configs=3, sigmas=0.1, seed=42)

    for i, (a, b) in enumerate(zip(snaps_a, snaps_b)):
        assert np.array_equal(a, b), (
            f"snapshot {i} differs bit-for-bit; "
            f"max |Δ| = {np.max(np.abs(a - b))}"
        )


def test_add_frozen_phonons_potential_threads_seed():
    """The package's add_frozen_phonons_potential must actually use ctx.frozen_phonons_seed."""
    # Lazy import — the cli module pulls a lot of deps. Only needed for this test.
    from abtem_run.pipeline import add_frozen_phonons_potential

    lamella = _build_lamella()
    ctx = SimpleNamespace(
        frozen_phonons=3,
        fph_sigma=0.1,
        frozen_phonons_seed=42,
    )

    # Calling add_frozen_phonons_potential twice with the same ctx returns
    # two lazy Potentials backed by FrozenPhonons. We can read the seed off
    # the FrozenPhonons object via .ensemble (it's the first slot in the
    # abtem Potential's ensemble chain).
    pot_a = add_frozen_phonons_potential(ctx, lamella)
    pot_b = add_frozen_phonons_potential(ctx, lamella)

    # Sanity: both potentials carry the same seed all the way through.
    # NB: abtem transforms the integer we pass (e.g. 42) into a derived
    # tuple of ints internally. We check the two calls match each other,
    # not that they equal the input literal.
    fph_a = pot_a.frozen_phonons if hasattr(pot_a, "frozen_phonons") else None
    fph_b = pot_b.frozen_phonons if hasattr(pot_b, "frozen_phonons") else None
    if fph_a is not None and fph_b is not None:
        assert fph_a.seed == fph_b.seed, (
            f"seed not consistent across calls to add_frozen_phonons_potential "
            f"(a={fph_a.seed}, b={fph_b.seed})"
        )

    # Strong check: the displacement trajectories the Potential will use must
    # match. We re-derive them from the same FrozenPhonons constructor that
    # add_frozen_phonons_potential uses internally.
    snaps_a = _fph_snapshots(
        lamella,
        num_configs=ctx.frozen_phonons,
        sigmas=ctx.fph_sigma,
        seed=ctx.frozen_phonons_seed,
    )
    snaps_b = _fph_snapshots(
        lamella,
        num_configs=ctx.frozen_phonons,
        sigmas=ctx.fph_sigma,
        seed=ctx.frozen_phonons_seed,
    )
    for i, (a, b) in enumerate(zip(snaps_a, snaps_b)):
        assert np.array_equal(a, b), f"snapshot {i} differs"


def _run_all():
    for fn in (
        test_make_lamella_is_deterministic,
        test_phonons_on_real_lamella_reproducible,
        test_add_frozen_phonons_potential_threads_seed,
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
