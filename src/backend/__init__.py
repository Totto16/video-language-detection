import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, Coroutine, Self, override

import requests
import uvicorn
from fastapi import Depends, FastAPI, Response, WebSocket

from config import FinalConfig
from helper.result import Result


class BackendRef:
    __backend: "Backend"

    def __init__(self: Self, backend: "Backend") -> None:
        self.__backend = backend

    async def wait_for_ready(self: Self) -> "Backend":
        await self.__backend.ready()
        return self.__backend


ProcessResult = Result[Any, str]


# TODO. maybe use abc.abstractmethod instead of this paradigm for abstract classed?
class MissingOverrideError(RuntimeError):
    pass


class WebsocketHandler:
    __ws: WebSocket

    def __init__(self: Self, websocket: WebSocket) -> None:
        self.__ws = websocket

    async def process_data(self: Self, _data: Any) -> ProcessResult:
        raise MissingOverrideError

    async def send_data(self: Self, data: Any) -> None:
        await self.__ws.send_json({"type": "ok", "data": data})

    async def send_error(self: Self, error: str) -> None:
        await self.__ws.send_json(
            {"type": "error", "error": error},
        )

    async def process(self: Self) -> None:
        while True:
            try:
                data = await self.__ws.receive_json()
                result = await self.process_data(data)
                if result.is_ok():
                    await self.send_data(result.get_ok())
                else:
                    await self.send_error(result.get_err())

            except RuntimeError as err:
                await self.__ws.send_json({"type": "error", "error": str(err)})


class SingleManager(WebsocketHandler):

    @override
    async def process_data(self: Self, _data: Any) -> ProcessResult:
        return ProcessResult.err("Nothing can be written in this cases")


class Manager:
    __instances: list[SingleManager]

    def __init__(self: Self) -> None:
        self.__instances = []

    def add(self: Self, websocket: WebSocket) -> SingleManager:
        manager = SingleManager(websocket)
        self.__instances.append(manager)
        return manager

    async def send_data(self: Self, data: Any) -> None:
        futures: list[Coroutine[Any, Any, None]] = [
            instance.send_data(data) for instance in self.__instances
        ]

        await asyncio.gather(*futures)


def register_routes(app: FastAPI, backend_ref: BackendRef) -> None:

    async def retreive_backend() -> "Backend":
        return await backend_ref.wait_for_ready()

    @app.get("/shutdown/")
    async def shutdown(
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> Response:
        await backend.shutdown()
        return Response(status_code=200, content="Server shutting down")

    @app.websocket("/manager/ws/")
    async def manager_ws(
        websocket: WebSocket,
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> None:
        await websocket.accept()
        manager = backend.add_manager(websocket)
        await manager.process()


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
    __ready: asyncio.Event
    __manager: Manager

    def __init__(self: Self, options: BackendOptions) -> None:
        self.__options = options
        self.__manager = Manager()
        self.__ready = asyncio.Event()

        app = FastAPI(dependencies=[Depends(self.ready)])
        register_routes(app, BackendRef(backend=self))

        config = uvicorn.Config(
            app=app,
            host=options.address.host,
            port=options.address.port,
            log_level="info",
            lifespan="on",
            reload=False,
        )

        self.__server = uvicorn.Server(config=config)
        self.__ready.set()

    async def ready(self: Self) -> None:
        if self.__ready.is_set():
            return

        await self.__ready.wait()

    def add_manager(self: Self, websocket: WebSocket) -> SingleManager:
        return self.__manager.add(websocket)

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


async def start_all(options: BackendOptions) -> int:
    backend = Backend(options)

    await backend.run()
    return 0


def launch_api(_configs: list[FinalConfig], options: BackendOptions) -> int:
    return asyncio.run(start_all(options))
