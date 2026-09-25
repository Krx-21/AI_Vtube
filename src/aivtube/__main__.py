"""``python -m aivtube``: delegates to ``aivtube.cli:main`` (imported lazily)."""

import sys


def _main() -> int:
    from aivtube.cli import main

    return main()


if __name__ == "__main__":
    sys.exit(_main())
