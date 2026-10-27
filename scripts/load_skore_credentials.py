#!/usr/bin/env python3
"""Load Skore Hub credentials from the nearest ``.skore`` file.

Sets ``SKORE_HUB_URI`` and ``SKORE_HUB_API_KEY`` so ``skore.login``
does not open a browser. Never prints the API key.

Always call this immediately before ``login(mode="hub")``. Do not
inline ``json.loads(Path(".skore")...)`` and do not wrap ``login``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _find_skore(path: Path | None = None) -> Path:
    """Return the nearest ``.skore``, walking up from ``path``, CWD, or this file."""
    if path is not None:
        dest = Path(path)
        if dest.is_file():
            return dest
        raise FileNotFoundError(
            f"Missing {dest}. Run: python scripts/install_skore.py"
        )
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for folder in [start, *start.parents]:
            if folder in seen:
                continue
            seen.add(folder)
            candidate = folder / ".skore"
            if candidate.is_file():
                return candidate
            if (folder / ".git").exists():
                break
    raise FileNotFoundError(
        "Missing .skore. Run: python scripts/install_skore.py"
    )


def load_skore_credentials(path: Path | None = None) -> dict:
    """Read the nearest ``.skore`` and export hub URI + API key into ``os.environ``.

    Parameters
    ----------
    path :
        Optional path to ``.skore``. When omitted, walk up from the
        current working directory, then from this module.

    Returns
    -------
    dict
        Public fields (``hub_url``, ``workspace``, ``workspace_id``).
        The API key is exported to ``os.environ`` only.
    """
    dest = _find_skore(path)
    try:
        cfg = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{dest} is not valid JSON") from exc
    if not isinstance(cfg, dict) or not cfg.get("hub_url") or not cfg.get("api_key"):
        raise ValueError(
            f"{dest} is missing hub_url or api_key. "
            "Run: python scripts/install_skore.py"
        )
    os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {key: value for key, value in cfg.items() if key != "api_key"}


if __name__ == "__main__":
    cfg = load_skore_credentials()
    print(f"hub_url={cfg['hub_url']}")
    print(f"workspace={cfg['workspace']}")
