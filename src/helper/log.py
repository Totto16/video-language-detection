import logging
import sys
from enum import Enum
from logging import Formatter, Logger, StreamHandler, getLogger
from typing import Optional, Self

__GLOBAL__LOGGER__NAME = "__global__logger__"


class LogLevel(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"

    @staticmethod
    def from_str(inp: str) -> Optional["LogLevel"]:
        for level in LogLevel:
            if str(level).lower() == inp.lower():
                return level

        return None

    @property
    def underlying(self: Self) -> int:
        match self:
            case LogLevel.CRITICAL:
                return logging.CRITICAL
            case LogLevel.ERROR:
                return logging.ERROR
            case LogLevel.WARNING:
                return logging.WARNING
            case LogLevel.INFO:
                return logging.INFO
            case LogLevel.DEBUG:
                return logging.DEBUG
            case LogLevel.NOTSET:
                return logging.NOTSET
            case _:
                msg = "UNREACHABLE!"
                raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return str(self.name).lower()

    def __repr__(self: Self) -> str:
        return self.__str__()


def get_logger() -> Logger:
    return getLogger(__GLOBAL__LOGGER__NAME)


def setup_custom_logger(level: LogLevel = LogLevel.DEBUG) -> Logger:
    formatter = Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # don't log anything from the global logger (speechbrain spits many things in here)
    root_logger = getLogger()
    root_logger.setLevel(logging.ERROR)

    logger = get_logger()
    logger.propagate = False  # don't propagate to the root handler

    if logger.hasHandlers():
        msg = "Logger already initialized"
        raise RuntimeError(msg)

    logger.setLevel(level.underlying)
    logger.addHandler(handler)
    return logger
