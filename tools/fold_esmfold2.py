"""Compatibility entry point; implementation lives in the installed package."""

from functools import partial

from foldjet.models.io.cli import run

main = partial(run, "esmfold2")

if __name__ == "__main__":
    raise SystemExit(main())
