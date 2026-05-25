import asyncio
from collections.abc import Coroutine
import json
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Annotated, Any, Optional, Self, override

import requests
import uvicorn
from fastapi import Depends, FastAPI, Response, WebSocket
from fastapi.responses import JSONResponse

from classifier import Classifier, Model, voxlingua107_ecapa_model
from config import FinalConfig
from content.base_class import Content, LanguageScanner, ScanSummaryDetailed, Scanner
from content.general import NameParser
from content.language_picker import LanguagePicker, get_picker_from_config
from content.metadata.config import get_metadata_scanner_from_config
from content.metadata.scanner import MetadataScanner
from content.scanner import get_scanner_from_config
from content.summary import LanguageDict, MetadataDict, Summary
from entry import CustomNameParser
from helper.base import AnyType, ManagerInterface, parse_contents
from helper.devices import DeviceManager
from helper.result import Result
from main import AllContent


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


class ScanManager(WebsocketHandler):

    @override
    async def process_data(self: Self, _data: Any) -> ProcessResult:
        # TODO: use pydantic to get e.g. status request
        return ProcessResult.err("Nothing can be written in this cases")


class ScannerManager:
    __instances: list[ScanManager]

    def __init__(self: Self) -> None:
        self.__instances = []

    def add(self: Self, websocket: WebSocket) -> ScanManager:
        manager = ScanManager(websocket)
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

    @app.websocket("/scan/managers/ws/")
    async def scan_manager_ws(
        websocket: WebSocket,
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> None:
        await websocket.accept()
        manager = backend.scanner.add_manager(websocket)
        await manager.process()

    @app.websocket("/scan/start")
    async def scan_start(
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> Response:
        # TODO. support one / multiple / some configs
        configs: Optional[list[str]] = None
        result = backend.scanner.start(configs)

        if result.is_err():
            return JSONResponse(status_code=422, content={"error": result.get_err()})

        return JSONResponse(status_code=200, content={"ok": True})

    @app.websocket("/scan/status")
    async def scan_status(
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> Response:
        status = backend.scanner.status()
        return JSONResponse(status_code=200, content={"status": status})


@dataclass
class Address:
    host: str
    port: int

    def __str__(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendOptions:
    address: Address


class ScannerState(Enum):
    stopped = "stopped"
    running = "running"
    finished = "finished"


ScannerStartResult = Result[None, str]

type SummaryTuple = tuple[LanguageDict, MetadataDict, ScanSummaryDetailed]


class BackendScanner:
    __manager: ScannerManager
    __configs: list[FinalConfig]
    __state: ScannerState

    def __init__(
        self: Self,
        configs: list[FinalConfig],
    ) -> None:
        self.__configs = configs
        self.__manager = ScannerManager()
        self.__state = ScannerState.stopped

    def add_manager(self: Self, websocket: WebSocket) -> ScanManager:
        return self.__manager.add(websocket)

    async def __launch_scanner_in_background(
        self: Self,
        config: FinalConfig,
        name_parser: NameParser,
        all_content_type: AnyType,
        config_paramaters: Optional[tuple[int, int]],
        manager: ManagerInterface,
    ) -> SummaryTuple:
        device_manager: DeviceManager = DeviceManager()

        model: Model = voxlingua107_ecapa_model

        classifier = Classifier(
            device_manager=device_manager,
            model=model,
            options=config.classifier,
        )
        language_scanner = LanguageScanner(classifier=classifier)
        metadata_scanner: MetadataScanner = get_metadata_scanner_from_config(
            config.metadata,
        )
        scanner: Scanner = get_scanner_from_config(
            config.scanner,
            language_scanner,
            metadata_scanner=metadata_scanner,
        )

        # TODO: note: we need to have a picker that is configured over ws not over terminal
        language_picker: LanguagePicker = get_picker_from_config(config.picker)

        general_info: list[str] = [
            x
            for x in [
                f"Config: {config.config_name}",
                (
                    None
                    if config_paramaters is None
                    else f"Config progress: {config_paramaters[0]+1} / {config_paramaters[1]}"
                ),
                f"Config type: {config.config_type.value}",
            ]
            if x is not None
        ]

        contents: list[Content] = parse_contents(
            root_folder=config.parser.root_folder,
            options={
                "ignore_files": config.parser.ignore_files,
                "video_formats": config.parser.video_formats,
                "trailer_names": config.parser.trailer_names,
                "parse_error_is_exception": config.parser.exception_on_error,
            },
            save_file=config.general.target_file,
            name_parser=name_parser,
            scanner=scanner,
            language_picker=language_picker,
            all_content_type=all_content_type,
            general_info=general_info,
            config_type=config.config_type,
            manager=manager,
        )

        language_summary, metadata_summary = Summary.combine_summaries(
            content.summary() for content in contents
        )

        scan_summary = language_scanner.summary_manager.get_detailed_summary()

        return (language_summary, metadata_summary, scan_summary)

    async def __start_coroutine(
        self: Self, configs: list[FinalConfig]
    ) -> list[SummaryTuple]:
        result: list[SummaryTuple] = []

        manager = None

        # TODO: set current configs and configs to process, support arguments
        for index, config in enumerate(configs):
            name_parser = CustomNameParser(season_special_names=config.parser.special)

            config_paramaters: Optional[tuple[int, int]] = (
                None if len(self.__configs) == 1 else (index, len(self.__configs))
            )

            summary = await self.__launch_scanner_in_background(
                config=config,
                name_parser=name_parser,
                all_content_type=AllContent,
                config_paramaters=config_paramaters,
                manager=manager,
            )

            result.append(summary)

        return result

    def __start_impl(self: Self, configs: Optional[list[str]]) -> ScannerStartResult:

        if configs is not None:
            # TODO: implement
            return ScannerStartResult.err("TODO")

        self.__status = ScannerState.running

        task: asyncio.Task[list[SummaryTuple]] = asyncio.Task(
            self.__start_coroutine(configs=self.__configs),
        )

        def done(task: asyncio.Task[list[SummaryTuple]]) -> None:
            result: list[SummaryTuple] = task.result()

            # set result to the state
            self.__status = ScannerState.finished

        task.add_done_callback(done)

        return ScannerStartResult.ok(None)

    def start(self: Self, configs: Optional[list[str]]) -> ScannerStartResult:
        if self.__state != ScannerState.stopped:
            return ScannerStartResult.err("Scanner is already running")

        if configs is None:
            return self.__start_impl(configs)

        # TODO: implement
        return ScannerStartResult.err("TODO")

    def status(self: Self) -> str:
        return self.__state.value


class Backend:
    __options: BackendOptions
    __server: uvicorn.Server
    __ready: asyncio.Event
    __scanner: BackendScanner

    def __init__(
        self: Self, options: BackendOptions, configs: list[FinalConfig]
    ) -> None:
        self.__options = options
        self.__scanner = BackendScanner(configs)
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

    async def shutdown(self: Self) -> None:
        if not self.__server:
            return

        await self.__server.shutdown()

    @property
    def scanner(self: Self) -> BackendScanner:
        return self.__scanner

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


async def start_all(options: BackendOptions, configs: list[FinalConfig]) -> int:
    backend = Backend(options=options, configs=configs)

    await backend.run()
    return 0


def launch_api(options: BackendOptions, configs: list[FinalConfig]) -> int:
    return asyncio.run(start_all(options, configs))
