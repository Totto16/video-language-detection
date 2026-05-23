from dataclasses import dataclass
from typing import Self

import requests
import uvicorn
from fastapi import FastAPI, Response


def register_routes(app: FastAPI, backend: "Backend") -> None:
    @app.get("/shutdown/")
    async def shutdown() -> Response:
        await backend.shutdown()
        return Response(status_code=200, content="Server shutting down")


@dataclass
class Address:
    host: str
    port: int

    def __str__(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendOptions:
    address: Address


class Backend:
    __options: BackendOptions
    __server: uvicorn.Server

    def __init__(self: Self, options: BackendOptions) -> None:
        self.__options = options

        app = FastAPI()
        register_routes(app, self)

        config = uvicorn.Config(
            app=app,
            host=options.address.host,
            port=options.address.port,
            log_level="info",
            lifespan="on",
            reload=False,
        )

        self.__server = uvicorn.Server(config=config)

    async def shutdown(self: Self) -> None:
        if not self.__server:
            return

        await self.__server.shutdown()

    def shutdown_app(self: Self, timeout: float) -> bool:
        try:
            stop_result = requests.get(
                f"{self.__options.address!s}/shutdown",
                timeout=timeout,
            )
        except RuntimeError:
            return False
        except requests.exceptions.RequestException:
            return False
        else:
            return stop_result.status_code == 200

    async def run(self: Self) -> None:
        await self.__server.serve()
        await self.__server.shutdown()
