"""
Package-wide logging for sparkmobility.

Call ``configure_logging()`` to route ``sparkmobility.*`` loggers to the
console (and optionally a file). Models auto-call this on first use unless
the user has already attached a handler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER_NAME = "sparkmobility"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Attach a stream (and optional file) handler to the sparkmobility logger.

    Idempotent: repeat calls update the level but don't duplicate handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_file = str(log_file)
        have_same_file = any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == str(Path(log_file).resolve())
            for h in logger.handlers
        )
        if not have_same_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def configure_if_needed(level: str = "INFO") -> None:
    """Attach a default StreamHandler if nothing is wired up yet.

    Called from model ``__init__`` so a bare ``Model().fit(...)`` in a Jupyter
    cell produces visible progress without the user calling
    :func:`configure_logging` first.
    """
    logger = logging.getLogger(LOGGER_NAME)
    # Respect existing user config — both direct handlers and any handler on
    # the root logger that would already catch our records.
    if logger.handlers:
        return
    root = logging.getLogger()
    if root.handlers:
        return
    configure_logging(level=level)
