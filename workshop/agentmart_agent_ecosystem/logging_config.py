"""Shared logging for the AgentMart LangGraph ecosystem and its A2A server.

Every entry point (``agentmart_ecosystem.py`` CLI, ``a2a_server.py``, the
scenario suite) uses the same logger namespace (:py:mod:`~agentmart`) and the
same handlers, so a run's hops land in one place whether it came over HTTP or
the CLI.

Handlers:

* ``logs/agentmart.log`` — rotating file (1 MB, 3 backups), DEBUG+.
* stderr console — level from ``AGENTMART_LOG_LEVEL`` (default INFO), or the
  ``--verbose`` / ``--log-level`` flags on the entry points.

``setup_logging`` is idempotent: a second call returns the same configuration,
so the A2A server and a graph run in the same process share one handler set.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "agentmart"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "agentmart.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

_configured = False
_console_handler: logging.Handler | None = None


def setup_logging(level: str | int | None = None, *, log_file: Path | None = None,
                  console: bool = True) -> logging.Logger:
    """Configure the ``agentmart`` logger tree and return its root logger.

    Idempotent for handlers (a second call in the same process reuses them, so
    the A2A server and a graph run in one process share a single handler set),
    but the ``level`` is re-applied to the console handler on every call, so
    ``--verbose`` between processes still takes effect.

    Parameters:
        level: Log level name/number; console threshold. Defaults to the
            ``AGENTMART_LOG_LEVEL`` environment variable (``INFO`` if unset).
        log_file: Override the log file location (used by tests).
        console: Also emit to stderr.

    The rotating file always records at ``DEBUG`` so a postmortem never misses
    a hop even when the console is quiet.
    """
    global _configured, _console_handler

    if level is None:
        level = os.getenv("AGENTMART_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger(LOGGER_NAME)

    if not _configured:
        root = logging.getLogger(LOGGER_NAME)
        root.setLevel(logging.DEBUG)
        root.propagate = False
        fmt = logging.Formatter(FORMAT)

        target = log_file or LOG_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        if console:
            stream = logging.StreamHandler()
            stream.setFormatter(fmt)
            root.addHandler(stream)
            _console_handler = stream
        _configured = True

    if _console_handler is not None:
        _console_handler.setLevel(level)
    return root


def bind_agent(name: str) -> logging.Logger:
    """A child logger per agent so graph nodes log under ``agentmart.<agent>``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")