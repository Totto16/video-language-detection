import logging
import os
import sys
from enum import Enum
from logging import Logger, StreamHandler, getLogger
from typing import Optional, Self, assert_never
from warnings import filterwarnings

import colorlog

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

    @staticmethod
    def __underlying_impl(value: "LogLevel") -> int:
        match value:
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
                assert_never(value)

    @property
    def underlying(self: Self) -> int:
        return LogLevel.__underlying_impl(self)

    def __str__(self: Self) -> str:
        return str(self.name).lower()

    def __repr__(self: Self) -> str:
        return self.__str__()


def get_logger() -> Logger:
    return getLogger(__GLOBAL__LOGGER__NAME)


def setup_global_logger() -> None:
    # don't log anything from the global logger (speechbrain spits many things in here)
    root_logger = getLogger()
    root_logger.setLevel(logging.ERROR)

    filterwarnings("ignore")

    # set speechbrain log level
    os.environ["SB_LOG_LEVEL"] = "WARNING"


def setup_custom_logger(level: LogLevel = LogLevel.DEBUG) -> Logger:
    formatter = colorlog.ColoredFormatter(
        fmt="%(blue)s%(asctime)s%(reset)s - %(log_color)s%(levelname)s%(reset)s - %(green)s%(module)s%(reset)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_purple",
        },
    )

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    setup_global_logger()

    logger = get_logger()
    logger.propagate = False  # don't propagate to the root handler

    if logger.hasHandlers():
        msg = "Logger already initialized"
        raise RuntimeError(msg)

    logger.setLevel(level.underlying)
    logger.addHandler(hdlr=console_handler)
    return logger
