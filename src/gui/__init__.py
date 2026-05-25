import asyncio
from logging import Logger

from backend import Backend, BackendOptions
from config import FinalConfig
from helper.log import get_logger

logger: Logger = get_logger()


RUN_GUI = ["TODO", "arguments"]


async def start_gui(options: BackendOptions, backend: Backend) -> int:
    [process] = (
        await asyncio.subprocess.create_subprocess_exec(
            RUN_GUI[0],
            *RUN_GUI[1:],
            options.address.host,
            str(options.address.port),
        ),
    )

    app_ok = await process.wait()

    success = backend.shutdown_app(10.0)
    return 0 if success and app_ok else 1


async def start_all(options: BackendOptions, configs: list[FinalConfig]) -> int:
    backend = Backend(options=options, configs=configs)

    [_, result] = await asyncio.gather(
        backend.run(),
        start_gui(options=options, backend=backend),
    )

    return result


def launch_gui(options: BackendOptions, configs: list[FinalConfig]) -> int:
    return asyncio.run(start_all(options=options, configs=configs))
