from logging import Logger
from pathlib import Path

from backend import Address, Backend, BackendOptions
from helper.log import get_logger

logger: Logger = get_logger()


def start_gui(options: BackendOptions) -> int:
    return 1


def launch_gui(_config: Path) -> int:
    logger.error("ERROR: TODO")
    address = Address(host="127.0.0.1", port=4433)
    options = BackendOptions(address=address)
    backend = Backend(options)

    backend.start()

    result = start_gui(options)
    success = backend.stop(10.0)

    return result if success else 1
