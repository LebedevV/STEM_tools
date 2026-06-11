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
 - (done) dry\_run should be implemented as a flag in a full_run  (abtem-run --generate-only)
 - (done) separated lib file to be created
 - (done) gaussian blur is not handling borders correctly  (simulations.blur_boundary: nearest | constant | reflect | wrap, threaded into Images.gaussian_filter)
 - BF images to be confirmed  (add_probe warns on defocus='scherzer' with C30=0; empirical confirmation still pending)
 - (done) for the in-plane rotation to add 'this hkl up' functionality  (job.inplane_align_hkl + job.inplane_align_axis)
 - (done) check imported libraries  (no unused imports across the package — ruff --select F is clean apart from one f-string nit; ruff in dev deps)

Install (development):

	pip install -e .

then run from a directory containing config.toml:

	abtem-run

For GPU runs you also need cupy, dask-cuda, and rmm matched to your CUDA toolkit.

Acknowledgements:
 - Julie M. Bekkevold for the invalueable help with ab-initio simulations and guidance with abTEM code
 - Project SFI/21/US/3785 for financial support
 - Iterative cleanup and packaging work carried out by Ivan S. Titov
