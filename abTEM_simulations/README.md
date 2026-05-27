# abtem-run

STEM/TEM image simulation pipeline built on top of [`abtem`](https://github.com/abTEM/abTEM)
and [`ase`](https://gitlab.com/ase/ase). Given a CIF file and a crystallographic direction
(hkl or uvw), it builds a rotated, cropped "lamella" supercell, generates the projected
potential (optionally with frozen phonons), and simulates diffraction patterns, CBED, and
HAADF / ABF / BF scans.

Outputs are TIFFs, PNG previews, Zarr arrays, and per-run TOML config dumps written under
the directory configured by `paths.folder_sim + paths.extr` in the run config.

## Status

Pre-1.0 research code. The pipeline runs end-to-end on the author's setup but is **not
yet portable** — the abtem monkey-patches at import time are pinned to a specific abtem
build, and several config paths are absolute. See the "Limitations" section below.

## Install (development)

```bash
git clone https://github.com/LebedevV/STEM_tools.git
cd STEM_tools/abTEM_simulations
python -m venv venv
./venv/bin/pip install -e .
```

For GPU runs you also need `cupy`, `dask-cuda`, and `rmm` matched to your CUDA toolkit
(installed manually — see `pyproject.toml` for details).

## Run

```bash
cd STEM_tools/abTEM_simulations  # CWD must contain config.toml
./venv/bin/abtem-run          # console-script installed by pip
# or equivalently:
./venv/bin/python -m abtem_run
```

`abtem-run` loads `config.toml` from the current working directory. To change
which phase / direction runs, edit the hardcoded literal inside `main()` in
`src/abtem_run/cli.py` — `[job].phase` / `.hkl_to_do` / `.is_uvw` are parsed but
not yet consumed at runtime (TODO). `[job].phonons_seed` IS wired.

## Configuration

All runtime parameters live in a TOML config file validated by Pydantic models in
`src/abtem_run/config.py`. See the inline comments in `config.toml` for an
annotated example.

Several scalar fields accept lists for parameter sweeps (`frozen_phonons`, `fph_sigma`,
`thickness`, `global_tilt_a`, `global_tilt_b`, `probability_of_vac`, `HT_value`); the
pipeline iterates the Cartesian product of all such fields.

## Limitations

- **abtem monkey-patches at import time** target abtem 1.0.9 internals. Newer or older
  versions may not need them and may break. Making these conditional is on the TODO list.
- Several config paths (`paths.folder_sim`, `paths.folder`) are absolute and assume a
  specific filesystem layout.
- `[job].phase` / `.hkl_to_do` / `.is_uvw` are parsed but not consumed at runtime —
  see the TODO in `src/abtem_run/cli.py`. (`[job].phonons_seed` IS wired.)
- `add_vacancies` is non-deterministic (unseeded `random.random()`) — runs with
  `add_vacancies_toggle = true` cannot be reproduced exactly. TODO in
  `src/abtem_run/simulation.py`.
- `add_probe` uses `defocus="scherzer"` by default. With abtem's default `C30=0`,
  Scherzer evaluates to zero — the BF channel may show an inverted "BF looks like
  DF" pattern. Verify BF output or pass an explicit numeric defocus.
- No test suite. No CI.

## Authorship

Original code by **Vasily A. Lebedev**.

Iterative cleanup and packaging work in this repository was carried out by Ivan S. Titov.

## License

GPL-v3. See the `LICENSE` file at the repository root.
