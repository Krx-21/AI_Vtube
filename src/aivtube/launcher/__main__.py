"""``python -m aivtube.launcher [run flags]``: the same as ``aivtube run``."""

import sys

from aivtube.launcher.main import main

if __name__ == "__main__":
    sys.exit(main())
