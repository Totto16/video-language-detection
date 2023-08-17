from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger

__GLOBAL__LOGGER__NAME = "__global__logger__"


def get_logger() -> Logger:
    return getLogger(__GLOBAL__LOGGER__NAME)


def setup_custom_logger(level: int = DEBUG) -> Logger:
    formatter = Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    handler = StreamHandler()
    handler.setFormatter(formatter)

    logger = get_logger()
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
