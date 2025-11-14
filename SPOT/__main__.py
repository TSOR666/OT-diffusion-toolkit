"""Command line entrypoint for ``python -m SPOT``."""
from __future__ import annotations

from .selftest import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
