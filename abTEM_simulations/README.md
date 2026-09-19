This code meant to:

 - import crystallographic data from a given cif file
 - expand the lattice to create a large set (superblock) of atoms
 - rotate a superblock to direct a given uvw (or normal to hkl, depends on settings) along Z
 - crop a superblock to the given 'lamella' sizes
 - perform a simulation using abTEM routines

The main benefit of this code is the possibility to deal with non-orthogonal space groups
Trigonal symmentry has not been assessed yet; looking forward to hear any feedback on that! 

Known issues and TODO:
 - (done) move most variables to JSON  (now a validated config.toml)
 - (done) add a support of variables as lists to iterate over all combinations  (expand_cfg sweeps frozen_phonons / fph_sigma / thickness / global_tilt_a / global_tilt_b / probability_of_vac / HT_value; phase and hkl_to_do also accept lists)
 - (done) fix the issue with random_seed for vacancies generator
 - add a possibility of different types of vacancies simultaneously  (still open: one element_to_remove per job)
 - (partial) add a way to import an ase set  (worker + aggregator read job_dir/surf.xyz, so a hand-placed extxyz is honored when those stages run directly; the generator overwrites it on a full run, and there is no config field for an external path yet)
 - (partial) output file names to reflect was it uvw or hkl  (captured in combined.png title + run_manifest.json per job; the job-dir / output filenames still don't encode it)
 - (done) first frame is simulated separately from frozen phonons, and this simulation is just repeated later on.
	resolved by the worker redesign: every seed shares one code path (static lattice = fph_sigma off + a single seed); the static projection is opt-in via simulations.emit_static_baseline
 - (done) dry\_run should be implemented as a flag in a full_run  (python run.py --generate-only)
 - (done) separated lib file to be created
 - (done) gaussian blur is not handling borders correctly  (simulations.blur_boundary: nearest | constant | reflect | wrap, threaded into Images.gaussian_filter)
 - BF images to be confirmed  (add_probe warns on defocus='scherzer' with C30=0; empirical confirmation still pending)
 - (done) for the in-plane rotation to add 'this hkl up' functionality  (job.inplane_align_hkl + job.inplane_align_axis)
 - (done) check imported libraries  (no unused imports across the package — ruff --select F is clean apart from one f-string nit; ruff can be run separately)

Run from the `abTEM_simulations` directory with Python 3.11 or newer.
These scripts are not installed as a package. Third-party dependencies are
listed in `requirements.txt` and can be installed into your Python environment:

    python -m pip install -r requirements.txt

Before the first simulation, apply the abTEM compatibility fixes once in the
Python environment used for simulations:

    python -m abtem_run.compat

This command explains the fixes and asks before modifying the installed
**third-party abTEM source**, not these scripts. It fixes GPU partition metadata
(CuPy cannot create the object arrays used for this metadata) and enables the
Gaussian blur boundary modes used by the output routines. The detailed rationale
is retained in `abtem_run/compat.py`; `python -m abtem_run.compat --help` displays it.
Use `--yes` only when you want to skip the confirmation.

Start a new Python process afterwards. The fixes then apply to separate workers
and aggregators as well. Reinstalling abTEM can overwrite them; rerun the command
if needed. After upgrading abTEM, review compatibility first: the persistent
patcher refuses unrecognized function source rather than guessing a replacement.

Run a simulation directly from this directory:

    python run.py --config config.toml

To inspect the generated structures before running simulations:

    python run.py --config config.toml --generate-only
    python run.py --resume /path/to/gen_<UTC>

The local driver runs the seeds in sequence and averages their outputs.
If the persistent fixes have not been applied, the driver can instead apply
in-memory fixes for its current process. It asks first; `--apply-patches` accepts
them without a prompt. In-memory fixes do not carry over to later processes.

Individual stages can also be run from this directory:

    python -m abtem_run.generator_run config.toml
    python -m abtem_run.worker /path/to/job /path/to/job/seeds/seed_000000.todo
    python -m abtem_run.aggregate /path/to/job
    python -m abtem_run.extend /path/to/job --add 10
    python -m abtem_run.to_ensemble /path/to/job

Use `python run.py --resume /path/to/gen_<UTC>` for the normal workflow.
Direct worker and aggregate commands expect the one-time compatibility setup;
they warn if known fixes are missing and do not apply them automatically.

GPU runs need CuPy matched to your CUDA toolkit. Dask-CUDA is only needed
when `gpu_related.dask_cuda` is enabled; RMM is optional.

Acknowledgements:
 - Julie M. Bekkevold for the invalueable help with ab-initio simulations and guidance with abTEM code
 - Project SFI/21/US/3785 for financial support
 - Iterative cleanup and packaging work carried out by Ivan S. Titov

This patch was prepared with AI assistance and is subject to author review and amendment.
