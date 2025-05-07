import logging
import os
import sys
from enum import Enum
from logging import Formatter, Logger, LogRecord, StreamHandler, getLogger
from typing import Any, Optional, Self, override
from warnings import filterwarnings

from prompt_toolkit.formatted_text import FormattedText, to_plain_text

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
            case _:  # mypy is stupid in match statements :(
                msg = "UNREACHABLE!"
                raise RuntimeError(msg)

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


Color = str
ColorDict = dict[int, Color]
ColorMappings = dict[str, Color]


class FormatAsColor:
    __color: Color
    __value: Any
    __fmt_string: str

    def __init__(self: Self, color: Color, value: Any, fmt_string: str = "%s") -> None:
        self.__color = color
        self.__value = value
        self.__fmt_string = fmt_string

    def __str__(self: Self) -> str:
        formatted_text = FormattedText(
            [(self.__color, self.__fmt_string % self.__value)],
        )

        def get_text(tpl: tuple[str, str] | tuple[str, str, Any]) -> str:
            print(tpl)
            return tpl[1]

        return to_plain_text(formatted_text)


class ColorFormatter(Formatter):
    __color_mappings: ColorMappings
    __log_level_colors: ColorDict

    def __init__(
        self: Self,
        fmt: str,
        *,
        color_mappings: Optional[ColorMappings],
        log_level_colors: Optional[ColorDict] = None,
    ) -> None:
        super().__init__(fmt=fmt)
        self.__color_mappings = color_mappings if color_mappings is not None else {}
        self.__log_level_colors = (
            log_level_colors
            if log_level_colors is not None
            else self.__get_default_log_level_colors()
        )

    def __get_default_log_level_colors(self: Self) -> ColorDict:
        return {
            logging.CRITICAL: "bold fg:ansimagenta",
            logging.ERROR: "bold fg:ansired",
            logging.WARNING: "fg:ansiyellow",
            logging.INFO: "fg:ansicyan",
            logging.DEBUG: "fg:ansigrey",
        }

    def __modify_record(self: Self, record: LogRecord) -> LogRecord:
        log_level = record.__getattribute__("levelname")
        if log_level is not None:
            log_level_color = self.__log_level_colors.get(log_level, "")
            record.__setattr__("levelname", FormatAsColor(log_level_color, log_level))

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        for mapped_name, color_for_name in self.__color_mappings.items():
            value = record.__getattribute__(mapped_name)
            if value is not None:
                record.__setattr__(mapped_name, FormatAsColor(color_for_name, value))

        return record

    @override
    def format(self: Self, record: LogRecord) -> str:
        modified_record = self.__modify_record(record)
        return super().format(modified_record)


def setup_custom_logger(level: LogLevel = LogLevel.DEBUG) -> Logger:
    formatter = ColorFormatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        color_mappings={"asctime": "fg:ansiblue", "module": "fg:ansigreen"},
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
