import logging

from pytest_subtests import SubTests

from helper.log import LogLevel, get_logger, setup_custom_logger


def test_loglevel_from_str(subtests: SubTests) -> None:
    wrong_names: list[str] = ["dsadsad", "sd", "CRITICALsas", "sNOTSET", "hello world"]
    for wrong_name in wrong_names:
        with subtests.test():
            assert LogLevel.from_str(wrong_name) is None, "None was returned"

    correct_names: list[tuple[str, LogLevel]] = [
        ("CRITICAL", LogLevel.CRITICAL),
        ("ERROR", LogLevel.ERROR),
        ("WARNING", LogLevel.WARNING),
        ("INFO", LogLevel.INFO),
        ("DEBUG", LogLevel.DEBUG),
        ("NOTSET", LogLevel.NOTSET),
        ("critical", LogLevel.CRITICAL),
        ("warNiNg", LogLevel.WARNING),
        ("wArNiNg", LogLevel.WARNING),
        ("WARNINg", LogLevel.WARNING),
    ]
    for correct_name, correct_level in correct_names:
        with subtests.test():
            assert (
                LogLevel.from_str(correct_name) is correct_level
            ), "correct LogLevel was returned"


def test_loglevel_underlying(subtests: SubTests) -> None:
    correct_names: list[tuple[LogLevel, int]] = [
        (LogLevel.CRITICAL, logging.CRITICAL),
        (LogLevel.ERROR, logging.ERROR),
        (LogLevel.WARNING, logging.WARNING),
        (LogLevel.INFO, logging.INFO),
        (LogLevel.DEBUG, logging.DEBUG),
        (LogLevel.NOTSET, logging.NOTSET),
    ]
    for level, result in correct_names:
        with subtests.test():
            assert level.underlying == result, "LogLevel to underlying works correctly"


def test_get_logger() -> None:
    assert get_logger() is not None, "get_logger returns not None"
    assert isinstance(get_logger(), logging.Logger), "get_logger returns Logger"


def test_setup_custom_logger() -> None:
    assert setup_custom_logger() is not None, "get_logger returns not None"
    assert isinstance(
        setup_custom_logger(),
        logging.Logger,
    ), "get_logger returns Logger"
