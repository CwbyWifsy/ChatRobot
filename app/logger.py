import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import settings


def configure_logging(log_dir: Optional[Path] = None) -> None:
    log_directory = log_dir or settings.log_directory
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file = log_directory / "interactions.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


configure_logging()
