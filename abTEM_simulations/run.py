from abTEM_simulations.abtem_run.compat import apply_abtem_patches

apply_abtem_patches()

from abtem_run.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
