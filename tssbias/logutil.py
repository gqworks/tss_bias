"""Process-wide logger for pipeline steps."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "tssbias") -> logging.Logger:
    """Return a logger that writes step lines to stderr.

    Inputs:
        name: logger name, usually ``tssbias.<command>``.
    Outputs:
        Logger at INFO with a single stream handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger
