from dataclasses import dataclass
from threading import Thread
from typing import Self

import requests
import uvicorn
from fastapi import FastAPI
import asyncio
import os
import signal


def register_routes(app) -> None:
    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: str | None = None):
        return {"item_id": item_id, "q": q}

    @app.get("/shutdown/")
    def shutdown():
        os.kill(os.getpid(), signal.SIGUSR2)
        return fastapi.Response(status_code=200, content="Server shutting down...")


@dataclass
class Address:
    host: str
    port: int

    def __str__(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendOptions:
    address: Address


async def async_loop(server: uvicorn.Server) -> None:
    
    
    await asyncio.create_task(server.serve())

    await server.shutdown()


def start_app(options: BackendOptions) -> None:
    app = FastAPI()
    register_routes(app)
    config = uvicorn.Config(
        app=app,
        host=options.address.host,
        port=options.address.port,
        log_level="info",
        lifespan="on",
        reload=False,
    )
    server = uvicorn.Server(config=config)

    asyncio.run(async_loop(server))


def stop_app(address: Address, timeout: float) -> bool:
    stop_result = requests.get(f"{address:s}/shutdown", timeout=timeout)
    return stop_result.status_code == 200


class Backend:
    __handle: Thread
    __options: BackendOptions

    def __init__(self: Self, options: BackendOptions) -> None:
        self.__options = options
        self.__handle = Thread(
            target=start_app,
            name="backend-thread",
            args=((options,)),
        )

    def start(self: Self) -> None:
        self.__handle.start()

    def stop(self: Self, timeout: float) -> bool:
        success = stop_app(self.__options.address, timeout)
        self.__handle.join(timeout)

        if self.__handle.is_alive():
            return False

        return success
