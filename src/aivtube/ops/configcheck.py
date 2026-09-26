"""Full config validation for the launcher preflight, run in a subprocess (§2.6 step 1).

The launcher is stdlib-only, so it cannot import pydantic; it runs
``python -m aivtube.ops.configcheck --root DIR [--profile P] [--overrides JSON]`` instead. The
result is one JSON object on stdout: ``{"ok": bool, "errors": [{"path", "message_en",
"message_th", "hint", "hint_th", "source"}]}``; the exit code is 0 (valid) or 2 (invalid).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["check_config", "main"]


def _issue(err: Any) -> dict[str, Any]:
    return {
        "path": err.path,
        "message_en": err.message_en,
        "message_th": err.message_th,
        "hint": err.hint,
        "hint_th": err.hint_th,
        "source": err.source,
    }


def check_config(
    root: Path,
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load and cross-check the app config and every character; ``[]`` when valid."""
    from aivtube.config import ConfigError, load_characters, load_config

    try:
        cfg = load_config(root, profile=profile, cli_overrides=cli_overrides, env=env)
        load_characters(cfg)
    except ConfigError as exc:
        return [_issue(i) for i in exc.issues]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aivtube.ops.configcheck")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--overrides", default="{}", help="JSON object of dotted CLI overrides")
    args = parser.parse_args(argv)
    try:
        overrides = json.loads(args.overrides)
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a JSON object")
    except ValueError as exc:
        print(json.dumps({"ok": False, "errors": [{"path": "--overrides", "message_en": str(exc),
                                                   "message_th": str(exc), "hint": "",
                                                   "hint_th": "", "source": None}]}))
        return 2
    errors = check_config(args.root, profile=args.profile, cli_overrides=overrides)
    # ASCII-escaped JSON survives any console code page on Windows
    sys.stdout.write(json.dumps({"ok": not errors, "errors": errors}) + "\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
