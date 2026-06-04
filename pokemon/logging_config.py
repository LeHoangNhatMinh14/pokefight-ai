"""
Central logging setup for the PokeTeam AI app.

Call `configure_logging()` once at startup (app.py does this) to get:

- Console output with a compact `HH:MM:SS LEVEL [module] message` format.
- A rotating file at `logs/poketeam.log` (2 MB per file, 3 backups kept).
- Sensible quieting of noisy third-party loggers (urllib3, werkzeug).

Each module obtains a logger with `logger = logging.getLogger(__name__)`,
which gives log lines like:
    14:23:01 INFO  [app] request synergy pokemon=pikachu max_legendaries=0
    14:23:04 INFO  [recommenders.recommender] recommend_teammates pikachu depth=1 -> 3.85s
    14:23:04 INFO  [app] response synergy team_size=6 duration=3.92s

To bump verbosity for one run from the shell:
    POKETEAM_LOG_LEVEL=DEBUG python app.py
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union

_CONFIGURED = False


def configure_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure the root logger. Safe to call more than once."""
    global _CONFIGURED

    if level is None:
        level = os.environ.get("POKETEAM_LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Clear handlers each call so Flask's debug reloader doesn't stack duplicates.
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console -> stdout.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file. Default location: <repo_root>/logs/poketeam.log.
    if log_file is None:
        log_file = Path(__file__).resolve().parent / "logs" / "poketeam.log"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet noisy libraries unless you specifically asked for DEBUG.
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests_cache").setLevel(logging.WARNING)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    _CONFIGURED = True
    root.debug("logging configured: level=%s file=%s", level, log_file)
    return root
