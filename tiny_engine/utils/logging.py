# utils/logging.py
"""Logging and metrics tracking."""

import logging
import sys

class Log:
    @staticmethod
    def setup(
        level: int,
        filename: str | None = None
    ) -> None:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

        if filename is not None:
            handlers.append(logging.FileHandler(filename=filename, encoding="utf-8"))

        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s | %(levelname)-8s |"
                "%(name)s:%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers
        )


    @staticmethod
    def get_logger(
        name: str,
        level: int | None = None
    ) -> logging.Logger:
        logger = logging.getLogger(name)

        if level is not None:
            logger.setLevel(level)

        return logger




