"""``python -m aivtube``: delegates to ``aivtube.cli:main`` (imported lazily)."""

import sys


def _main() -> int:
    from aivtube.cli import main

    code: int = main()
    return code


if __name__ == "__main__":
    sys.exit(_main())
