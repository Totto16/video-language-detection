import asyncio
from logging import Logger
from pathlib import Path

from backend import Address, Backend, BackendOptions
from helper.log import get_logger

logger: Logger = get_logger()


async def start_gui(options: BackendOptions, backend: Backend) -> int:
    backend.shutdown_app(10.0)
    return 1


async def start_all(options: BackendOptions) -> int:
    backend = Backend(options)

    results = await asyncio.gather(
        backend.run(),
        start_gui(options=options, backend=backend),
    )

    return results[1]


def launch_gui(_config: Path) -> int:
    address = Address(host="127.0.0.1", port=4433)
    options = BackendOptions(address=address)

    return asyncio.run(start_all(options))
