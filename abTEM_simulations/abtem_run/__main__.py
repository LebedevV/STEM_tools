from .compat import apply_abtem_patches

apply_abtem_patches()

from .cli import main

if __name__ == "__main__":
	raise SystemExit(main())
