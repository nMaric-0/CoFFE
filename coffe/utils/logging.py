"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
