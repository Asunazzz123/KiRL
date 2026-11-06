"""Process logging configuration and scoped task context; no device operations."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_context: ContextVar[dict[str, Any]] = ContextVar("kirl_log_context", default={})
_fields = {
    "run_id", "role", "phase", "algorithm", "rank", "step",
    "iteration_id", "policy_version", "event", "all_ranks",
}


def _validate(fields: dict[str, Any]) -> None:
    unknown = fields.keys() - _fields
    if unknown:
        raise ValueError(f"Unknown logging context fields: {sorted(unknown)}")
    for name, value in fields.items():
        if value is not None and type(value) not in (str, int, float, bool):
            raise TypeError(f"Logging context {name!r} must be a CPU scalar")
    if "rank" in fields and (type(fields["rank"]) is not int or fields["rank"] < 0):
        raise ValueError("rank must be a non-negative integer")
    if "all_ranks" in fields and type(fields["all_ranks"]) is not bool:
        raise TypeError("all_ranks must be bool")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in Log.get_context().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True


class _RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.rank == 0 or record.levelno >= logging.WARNING or record.all_ranks


class _TaskAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Preserve per-event extra fields on Python versions before merge_extra.
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


class Log:
    @staticmethod
    def get_context() -> dict[str, Any]:
        """Return a snapshot, resolving torchrun rank without importing torch."""
        defaults = {name: "-" for name in _fields}
        defaults.update(rank=int(os.environ.get("RANK", "0")), all_ranks=False)
        return {**defaults, **_context.get()}

    @staticmethod
    def set_context(**fields: Any) -> None:
        """Set worker defaults at entry; use bind() for temporary work."""
        _validate(fields)
        _context.set({**_context.get(), **fields})

    @staticmethod
    @contextmanager
    def bind(**fields: Any):
        """Override context in this thread/async task and restore it on exit."""
        _validate(fields)
        token = _context.set({**_context.get(), **fields})
        try:
            yield
        finally:
            _context.reset(token)

    @staticmethod
    def setup(
        level: int = logging.INFO,
        filename: str | None = None,
        *,
        main_rank_only: bool = True,
        **context: Any,
    ) -> None:
        """
        Explicitly configure root handlers at the worker entry point.
        """
        Log.set_context(**context)
        rank = Log.get_context()["rank"]
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | run=%(run_id)s rank=%(rank)s "
            "role=%(role)s algorithm=%(algorithm)s phase=%(phase)s "
            "step=%(step)s iteration=%(iteration_id)s policy=%(policy_version)s "
            "event=%(event)s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.addFilter(_ContextFilter())
        if main_rank_only:
            console.addFilter(_RankFilter())
        handlers: list[logging.Handler] = [console]
        if filename is not None:
            path = Path(filename)
            if int(os.environ.get("WORLD_SIZE", "1")) > 1 or rank != 0:
                path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.addFilter(_ContextFilter())
            handlers.append(file_handler)
        for handler in handlers:
            handler.setFormatter(formatter)
        logging.basicConfig(level=level, handlers=handlers, force=True)

    @staticmethod
    def get_logger(
        name: str, level: int | None = None, **context: Any,
    ) -> logging.Logger | logging.LoggerAdapter:
        """Get a module logger, optionally binding fixed component metadata."""
        _validate(context)
        logger = logging.getLogger(name)
        if level is not None:
            logger.setLevel(level)
        return _TaskAdapter(logger, context) if context else logger
