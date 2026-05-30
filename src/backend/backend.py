import asyncio
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    Self,
    TypedDict,
    Unpack,
    assert_never,
    cast,
    override,
)

import pydantic
import requests
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from content.base_class import Content, LanguageScanner, Scanner, ScanSummaryDetailed
from content.general import NameParser
from content.language import Language
from content.language_picker import (
    ChoiceColor,
    ChoiceInterface,
    ChoiceManagerInterface,
    ChoiceTitle,
    LanguagePicker,
    ManualSelectResult,
    PredictionBestSelectResult,
    SelectedType,
    SelectResult,
    get_picker_from_config,
)
from content.metadata.config import get_metadata_scanner_from_config
from content.prediction import PredictionBest
from content.scanner import get_scanner_from_config
from content.summary import LanguageDict, MetadataDict, Summary
from helper.base import (
    AnyType,
    parse_contents,
)
from helper.classifier import Classifier, Model, voxlingua107_ecapa_model
from helper.config import ConfigFilter, ConfigFilterItem, FinalConfig, filter_configs
from helper.devices import DeviceManager
from helper.manager import (
    CounterInterface,
    CounterOptions,
    ManagerInterface,
    NumberLike,
    StatusBarGetOptions,
    StatusBarInterface,
    StatusBarInterfaceUpdateOptions,
    number_like_convert_to_serializable,
)
from helper.parser import CustomNameParser
from helper.result import Result
from main import AllContent

if TYPE_CHECKING:

    from content.metadata.scanner import MetadataScanner


class BackendRef:
    __backend: "Backend"

    def __init__(self: Self, backend: "Backend") -> None:
        self.__backend = backend

    async def wait_for_ready(self: Self) -> "Backend":
        await self.__backend.ready()
        return self.__backend


ProcessResult = Result[Any, str]


class WebsocketHandler(ABC):
    __ws: WebSocket

    def __init__(self: Self, websocket: WebSocket) -> None:
        super().__init__()
        self.__ws = websocket

    @abstractmethod
    async def process_data(self: Self, _data: Any) -> Optional[ProcessResult]: ...

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
                if result is not None:
                    if result.is_ok():
                        await self.send_data(result.get_ok())
                    else:
                        await self.send_error(result.get_err())
            except WebSocketDisconnect:
                return
            except RuntimeError as err:
                await self.__ws.send_json({"type": "error", "error": str(err)})


class ManagerWsChoiceMessageAskQuestionReply(TypedDict, total=True):
    type: Literal["reply"]
    reply: Literal["ask_question"]
    result: Optional["ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult"]
    id: str


IncomingWsData = ManagerWsChoiceMessageAskQuestionReply


ProcessResultTyped = Result["OutgoingWsData", str]


class ScanManager(WebsocketHandler):
    __parent_ref: "ScannerManager"

    def __init__(
        self: Self,
        parent_ref: "ScannerManager",
        websocket: WebSocket,
    ) -> None:
        super().__init__(websocket=websocket)
        self.__parent_ref = parent_ref

    async def __process_data(
        self: Self,
        data: IncomingWsData | dict[str, Any],
    ) -> Optional[ProcessResultTyped]:
        match data["type"]:
            case "reply":
                match data["reply"]:
                    case "ask_question":
                        res_data: Optional[SelectResult] = (
                            None
                            if data["result"] is None
                            else deserialize_select_result(data["result"])
                        )
                        id: uuid.UUID = uuid.UUID(data["id"])
                        result = self.__parent_ref.process_ask_question_reply(
                            res_data,
                            id,
                        )
                        if result is None:
                            response: ManagerWsChoiceMessageQuestionReplyReceived = {
                                "type": "choice",
                                "data": {"type": "reply_received", "id": data["id"]},
                            }
                            return ProcessResultTyped.ok(response)

                        return ProcessResultTyped.err(result)
                    case _:
                        return ProcessResultTyped.err("Invalid reply data received")
            case _:
                return ProcessResultTyped.err("Invalid data received")

    @override
    async def process_data(self: Self, data: Any) -> Optional[ProcessResult]:
        # TODO: use pydantic to validate the incoming data
        typed_data: IncomingWsData | dict[str, Any] = cast(
            IncomingWsData | dict[str, Any],
            data,
        )
        return await self.__process_data(typed_data)


class ManagerWsGlobalMessageGeneric[D](TypedDict, total=True):
    type: Literal["global"]
    data: D


class ManagerWsGlobalMessageStopData(TypedDict, total=True):
    type: Literal["stop"]


ManagerWsGlobalMessageStop = ManagerWsGlobalMessageGeneric[
    ManagerWsGlobalMessageStopData
]


class CounterTypeCounter(TypedDict, total=True):
    type: Literal["counter"]
    options: CounterOptions


class CounterTypeStatusBar(TypedDict, total=True):
    type: Literal["status_bar"]
    options: StatusBarGetOptions


CounterType = Literal["counter", "status_bar"]

CounterInstanceType = CounterTypeCounter | CounterTypeStatusBar


class ManagerWsGlobalMessageCounterData(TypedDict, total=True):
    type: Literal["counter"]
    counter: CounterInstanceType
    idx: int


ManagerWsGlobalMessageCounter = ManagerWsGlobalMessageGeneric[
    ManagerWsGlobalMessageCounterData
]

ManagerWsGlobalMessage = ManagerWsGlobalMessageStop | ManagerWsGlobalMessageCounter


class ManagerWsCounterMessageGeneric[D](TypedDict, total=True):
    type: Literal["counter"]
    data: D


class CounterUpdateOptions(TypedDict, total=False):
    incr: NumberLike  # = 1
    force: bool  # = False


class CounterUpdateTypeCounter(TypedDict, total=True):
    type: Literal["counter"]
    options: CounterUpdateOptions


class CounterUpdateTypeStatusBar(TypedDict, total=True):
    type: Literal["status_bar"]
    options: StatusBarInterfaceUpdateOptions


CounterUpdateType = CounterUpdateTypeCounter | CounterUpdateTypeStatusBar


class ManagerWsCounterMessageUpdateData(TypedDict, total=True):
    type: Literal["update"]
    idx: int
    options: CounterUpdateType


ManagerWsCounterMessageUpdate = ManagerWsCounterMessageGeneric[
    ManagerWsCounterMessageUpdateData
]


class CounterMessageCloseOptions(TypedDict, total=True):
    clear: bool


class ManagerWsCounterMessageCloseData(TypedDict, total=True):
    type: Literal["close"]
    idx: int
    options: CounterMessageCloseOptions


ManagerWsCounterMessageClose = ManagerWsCounterMessageGeneric[
    ManagerWsCounterMessageCloseData
]

ManagerWsCounterMessage = ManagerWsCounterMessageUpdate | ManagerWsCounterMessageClose


class ManagerWsChoiceMessageGeneric[D](TypedDict, total=True):
    type: Literal["choice"]
    data: D


class ManagerWsChoiceMessageAskQuestionChoiceDataSeparator(TypedDict, total=True):
    tag: Literal["separator"]


type ChoiceColorTypeValues = Literal["fg", "bg"]


type ChoiceColorValueValues = Literal["ansiblue", "ansigreen"]


class ChoiceColorDict(
    TypedDict,
    total=True,
):
    type: ChoiceColorTypeValues
    color: ChoiceColorValueValues


def choice_color_to_serializable_data(
    choice_color: ChoiceColor,
) -> ChoiceColorDict:
    return {"type": choice_color.type.value, "color": choice_color.color.value}


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle(
    TypedDict,
    total=True,
):
    color: Optional[ChoiceColorDict]
    content: str


def choice_title_to_serializable_data(
    title: ChoiceTitle,
) -> ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle:
    return {
        "content": title.content,
        "color": (
            None
            if title.color is None
            else choice_color_to_serializable_data(title.color)
        ),
    }


type SelectedTypeValues = Literal["open", "no_language", " copy", "more", "unknown"]


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual(
    TypedDict,
    total=True,
):
    type: Literal["manual"]
    selected: SelectedTypeValues


class LanguageTypedDict(
    TypedDict,
    total=True,
):
    short: str
    long: str


def language_to_serializable_data(language: Language) -> LanguageTypedDict:
    return {"short": language.short, "long": language.long}


def deserialize_language(language: LanguageTypedDict) -> Language:
    return Language(short=language["short"], long=language["long"])


class PredictionBestDict(
    TypedDict,
    total=True,
):
    accuracy: float
    language: LanguageTypedDict


def prediction_best_to_serializable_data(data: PredictionBest) -> PredictionBestDict:
    return {
        "accuracy": data.accuracy,
        "language": language_to_serializable_data(data.language),
    }


def deserialize_prediction_best(data: PredictionBestDict) -> PredictionBest:
    return PredictionBest(
        accuracy=data["accuracy"],
        language=deserialize_language(data["language"]),
    )


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest(
    TypedDict,
    total=True,
):
    type: Literal["prediction_best"]
    value: PredictionBestDict


ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult = (
    ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual
    | ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest
)


def select_result_to_serializable_data(
    result: SelectResult,
) -> ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult:
    if isinstance(result, ManualSelectResult):
        manual: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual = {
            "type": "manual",
            "selected": result.selected.value,
        }
        return manual
    if isinstance(result, PredictionBestSelectResult):
        value = prediction_best_to_serializable_data(result.value)

        choice: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest = {
            "type": "prediction_best",
            "value": value,
        }
        return choice
    assert_never(result)


def deserialize_select_result(
    result: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult,
) -> SelectResult:
    match result["type"]:
        case "manual":
            manual: ManualSelectResult = ManualSelectResult(
                select_result_type="manual",
                selected=SelectedType(result["selected"]),
            )
            return manual
        case "prediction_best":
            value = deserialize_prediction_best(result["value"])
            best: PredictionBestSelectResult = PredictionBestSelectResult(
                select_result_type="prediction_best",
                value=value,
            )
            return best
        case _:
            assert_never(result)


class ManagerWsChoiceMessageAskQuestionChoiceDataChoice(TypedDict, total=True):
    tag: Literal["choice"]
    title: list[ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle]
    value: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult


ManagerWsChoiceMessageAskQuestionChoiceData = (
    ManagerWsChoiceMessageAskQuestionChoiceDataSeparator
    | ManagerWsChoiceMessageAskQuestionChoiceDataChoice
)


def choice_to_serializable_data(
    data: "WSChoice",
) -> ManagerWsChoiceMessageAskQuestionChoiceData:
    if isinstance(data, WSChoiceSeparator):
        separator: ManagerWsChoiceMessageAskQuestionChoiceDataSeparator = {
            "tag": "separator",
        }
        return separator
    if isinstance(data, WSChoiceChoice):
        impl = data.impl

        title = [choice_title_to_serializable_data(segement) for segement in impl.title]
        value = select_result_to_serializable_data(impl.value)

        choice: ManagerWsChoiceMessageAskQuestionChoiceDataChoice = {
            "tag": "choice",
            "title": title,
            "value": value,
        }
        return choice
    assert_never(data)


class ManagerWsChoiceMessageAskQuestionData(TypedDict, total=True):
    type: Literal["ask_question"]
    message: str
    choices: list[ManagerWsChoiceMessageAskQuestionChoiceData]
    default: ManagerWsChoiceMessageAskQuestionChoiceData
    id: str


ManagerWsChoiceMessageAskQuestion = ManagerWsChoiceMessageGeneric[
    ManagerWsChoiceMessageAskQuestionData
]


class ManagerWsChoiceMessageQuestionReplyReceivedData(TypedDict, total=True):
    type: Literal["reply_received"]
    id: str


ManagerWsChoiceMessageQuestionReplyReceived = ManagerWsChoiceMessageGeneric[
    ManagerWsChoiceMessageQuestionReplyReceivedData
]


ManagerWsChoiceMessage = (
    ManagerWsChoiceMessageAskQuestion | ManagerWsChoiceMessageQuestionReplyReceived
)

OutgoingWsData = (
    ManagerWsGlobalMessage | ManagerWsCounterMessage | ManagerWsChoiceMessage
)


class ScannerStatusBar(StatusBarInterface):
    __ref: "ScannerManager"
    __idx: int

    def __init__(self: Self, ref: "ScannerManager", idx: int) -> None:
        super().__init__()
        self.__ref = ref
        self.__idx = idx

    async def __update_impl(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        data: ManagerWsCounterMessageUpdate = {
            "type": "counter",
            "data": {
                "type": "update",
                "idx": self.__idx,
                "options": {"type": "status_bar", "options": fields},
            },
        }
        await self.__ref.send_data(data)

    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        return asyncio.run(self.__update_impl(**fields))


class ScannerCounter(CounterInterface):
    __ref: "ScannerManager"
    __idx: int

    def __init__(self: Self, ref: "ScannerManager", idx: int) -> None:
        super().__init__()
        self.__ref = ref
        self.__idx = idx

    async def __update_impl(
        self: Self,
        incr: NumberLike = 1,
        force: bool = False,
    ) -> None:
        data: ManagerWsCounterMessageUpdate = {
            "type": "counter",
            "data": {
                "type": "update",
                "idx": self.__idx,
                "options": {
                    "type": "counter",
                    "options": {
                        "force": force,
                        "incr": number_like_convert_to_serializable(incr),
                    },
                },
            },
        }
        await self.__ref.send_data(data)

    @override
    def update(self: Self, incr: NumberLike = 1, force: bool = False) -> None:
        return asyncio.run(self.__update_impl(incr=incr, force=force))

    async def __close_impl(self: Self, clear: bool = False) -> None:
        data: ManagerWsCounterMessageClose = {
            "type": "counter",
            "data": {
                "type": "close",
                "idx": self.__idx,
                "options": {"clear": clear},
            },
        }
        await self.__ref.send_data(data)

    @override
    def close(self: Self, clear: bool = False) -> None:
        return asyncio.run(self.__close_impl(clear=clear))


class EmptyContextManager(AbstractContextManager[None]):
    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def __enter__(self: Self) -> None:
        pass

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        return False


@dataclass
class WSChoiceChoiceImpl:
    title: list[ChoiceTitle]
    value: SelectResult


class WSChoiceChoice(ChoiceInterface):
    __impl: WSChoiceChoiceImpl

    def __init__(self: Self, impl: WSChoiceChoiceImpl) -> None:
        super().__init__()
        self.__impl = impl

    @property
    def impl(self: Self) -> WSChoiceChoiceImpl:
        return self.__impl


class WSChoiceSeparator(ChoiceInterface):
    def __init__(self: Self) -> None:
        super().__init__()


WSChoice = WSChoiceChoice | WSChoiceSeparator


@dataclass
class ReplyData:
    type: str
    finished: bool
    event: asyncio.Event
    data: None | Any


class ManagerCtx(AbstractContextManager[ScanManager]):
    __manager: ScanManager
    __remove_fn: Callable[[], None]

    def __init__(
        self: Self, manager: ScanManager, remove_fn: Callable[[], None]
    ) -> None:
        super().__init__()
        self.__manager = manager
        self.__remove_fn = remove_fn

    @override
    def __enter__(self: Self) -> ScanManager:
        return self.__manager

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        self.__remove_fn()
        return False


class ScannerManager(ManagerInterface, ChoiceManagerInterface):
    __instances: list[ScanManager]
    __counters: list[CounterType]
    __loop: asyncio.AbstractEventLoop
    __reply_ids: dict[uuid.UUID, ReplyData]

    def __init__(self: Self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.__instances = []
        self.__counters = []
        self.__loop = loop
        self.__reply_ids = {}

    def ctx(self: Self, websocket: WebSocket) -> ManagerCtx:
        manager = ScanManager(self, websocket)
        self.__instances.append(manager)

        def remove_fn() -> None:
            self.__instances.remove(manager)

        return ManagerCtx(manager=manager, remove_fn=remove_fn)

    async def send_data(self: Self, data: OutgoingWsData) -> None:
        futures: list[Coroutine[Any, Any, None]] = [
            instance.send_data(data) for instance in self.__instances
        ]

        await asyncio.gather(*futures)

    def send_data_sync(self: Self, data: OutgoingWsData) -> None:
        future = asyncio.run_coroutine_threadsafe(
            coro=self.send_data(data),
            loop=self.__loop,
        )
        return future.result()

    def __add_counter(
        self: Self,
        instance: CounterInstanceType,
    ) -> int:
        idx = len(self.__counters)
        self.__counters.append(instance["type"])
        instance_serializable: CounterInstanceType

        if instance["type"] == "counter":
            serializable_options1: CounterOptions = {**instance["options"]}
            number_like_keys: list[Literal["count", "total"]] = [
                "count",
                "total",
            ]
            for number_like_key in number_like_keys:
                if serializable_options1.get(number_like_key) is not None:
                    serializable_options1[number_like_key] = (
                        number_like_convert_to_serializable(
                            serializable_options1[number_like_key],
                        )
                    )
            instance_serializable = {
                "type": "counter",
                "options": serializable_options1,
            }
        else:
            serializable_options2: StatusBarGetOptions = {**instance["options"]}

            instance_serializable = {
                "type": "status_bar",
                "options": serializable_options2,
            }
        data: ManagerWsGlobalMessageCounter = {
            "type": "global",
            "data": {"type": "counter", "counter": instance_serializable, "idx": idx},
        }
        self.send_data_sync(data)
        return idx

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        options: CounterTypeStatusBar = {"type": "status_bar", "options": kwargs}
        idx: int = self.__add_counter(options)
        return ScannerStatusBar(self, idx)

    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        options: CounterTypeCounter = {"type": "counter", "options": kwargs}
        idx: int = self.__add_counter(options)
        return ScannerCounter(self, idx)

    @override
    def stop(
        self: Self,
    ) -> None:
        data: ManagerWsGlobalMessageStop = {"type": "global", "data": {"type": "stop"}}
        self.send_data_sync(data)

    @override
    def get_choice(
        self: Self,
        title: list[ChoiceTitle],
        value: SelectResult,
    ) -> ChoiceInterface:
        return WSChoiceChoice(WSChoiceChoiceImpl(title=title, value=value))

    @override
    def get_separator(
        self: Self,
    ) -> ChoiceInterface:
        return WSChoiceSeparator()

    @override
    def picker_ctx(
        self: Self,
    ) -> AbstractContextManager[None]:
        return EmptyContextManager()

    @staticmethod
    def __get_underlying_choice(choice: ChoiceInterface) -> WSChoice:
        if isinstance(choice, WSChoiceChoice) or isinstance(choice, WSChoiceSeparator):
            return choice

        msg = "Implementation Error: used wrong choices with wrong choices manager!"
        raise RuntimeError(msg)

    @override
    def ask_question(
        self: Self,
        message: str,
        choices: list[ChoiceInterface],
        default: ChoiceInterface,
    ) -> Optional[SelectResult]:

        choices_impl = [
            choice_to_serializable_data(ScannerManager.__get_underlying_choice(choice))
            for choice in choices
        ]
        default_impl = choice_to_serializable_data(
            ScannerManager.__get_underlying_choice(default),
        )

        uid: uuid.UUID = uuid.uuid4()

        data: ManagerWsChoiceMessageAskQuestion = {
            "type": "choice",
            "data": {
                "type": "ask_question",
                "message": message,
                "choices": choices_impl,
                "default": default_impl,
                "id": str(uid),
            },
        }

        event = asyncio.Event()

        reply_data = ReplyData(
            type="ask_question",
            finished=False,
            event=event,
            data=None,
        )

        self.__reply_ids[uid] = reply_data

        async def wait_for_uuid_finish(r_data: ReplyData) -> Optional[SelectResult]:
            await r_data.event.wait()
            data = r_data.data

            del self.__reply_ids[uid]

            return data

        self.send_data_sync(data)

        thread_loop = asyncio.get_running_loop()

        future = asyncio.run_coroutine_threadsafe(
            coro=wait_for_uuid_finish(reply_data),
            loop=thread_loop,
        )
        return future.result()

    def process_ask_question_reply(
        self: Self,
        result: Optional[SelectResult],
        id: uuid.UUID,
    ) -> Optional[str]:
        reply_data = self.__reply_ids.get(id, None)
        if reply_data is None:
            return "Error: reply not present or already answered!"

        if reply_data.type != "ask_question":
            return "Error: wrong reply type!"

        if reply_data.finished or reply_data.event.is_set():
            return "Error: reply already finished!"

        reply_data.data = result
        reply_data.finished = True
        reply_data.event.set()

        return None


class ScanStartQuery(pydantic.BaseModel):
    model_config = {"extra": "forbid"}

    filter: Optional[list[ConfigFilterItem] | ConfigFilterItem] = None


def get_config_filters(
    filter_inp: Optional[list[ConfigFilterItem] | ConfigFilterItem],
) -> Optional[ConfigFilter]:
    if filter_inp is None:
        return None

    if isinstance(filter_inp, list):
        return filter_inp

    res: ConfigFilter = [filter_inp]
    return res


def register_routes(app: FastAPI, backend_ref: BackendRef) -> None:

    async def retreive_backend() -> "Backend":
        return await backend_ref.wait_for_ready()

    @app.get("/shutdown/")
    async def shutdown(
        backend: Annotated[Backend, Depends(retreive_backend)],
        background_tasks: BackgroundTasks,
    ) -> Response:

        # if we would await backend.shutdown() , we would deadlock here
        async def shutdown_ignore_result() -> None:
            backend.schedule_shutdown()

        background_tasks.add_task(shutdown_ignore_result)
        return Response(status_code=200, content="Server shutting down")

    @app.websocket("/scan/managers/ws/")
    async def scan_manager_ws(
        websocket: WebSocket,
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> None:
        await websocket.accept()
        with backend.scanner.manager_ctx(websocket) as manager:
            await manager.process()

    @app.get("/scan/start")
    async def scan_start(
        backend: Annotated[Backend, Depends(retreive_backend)],
        background_tasks: BackgroundTasks,
        start_query: Annotated[ScanStartQuery, Query()],
    ) -> Response:
        cfg_filter: Optional[ConfigFilter] = get_config_filters(start_query.filter)

        def run_in_background(fn: Callable[[], Coroutine[Any, Any, Any]]) -> None:
            print("add background task")
            background_tasks.add_task(fn)

        result: Optional[str] = backend.scanner.start(
            cfg_filter=cfg_filter,
            run_in_background=run_in_background,
        )

        if result is not None:
            return JSONResponse(status_code=422, content={"error": result})

        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/scan/status")
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


class ScannerStateIdle(TypedDict):
    type: Literal["idle"]


class ScannerStateRunning(TypedDict):
    type: Literal["running"]
    configs: list[FinalConfig]


type SummaryTuple = tuple[LanguageDict, MetadataDict, ScanSummaryDetailed]


class ScannerStateFinished(TypedDict):
    type: Literal["finished"]
    result: list[SummaryTuple]


class ScannerStateError(TypedDict):
    type: Literal["error"]
    error: str | BaseException


ScannerState = (
    ScannerStateIdle | ScannerStateRunning | ScannerStateFinished | ScannerStateError
)


class ThreadSafeAcquired[A]:
    __get_impl: Callable[[], A]
    __set_impl: Callable[[A], None]

    def __init__(self: Self, get_fn: Callable[[], A], set_fn: Callable[[A], None]):
        self.__get_impl = get_fn
        self.__set_impl = set_fn

    def set(self: Self, data: A) -> None:
        self.__set_impl(data)

    def get(self: Self) -> A:
        return self.__get_impl()

    def modify(self: Self, fn: Callable[[A], A]) -> None:
        self.__set_impl(fn(self.__get_impl()))


class ThreadSafe[A](AbstractContextManager[ThreadSafeAcquired[A]]):
    __data: A
    __mutex: threading.Lock

    def __init__(self: Self, data: A) -> None:
        super().__init__()
        self.__mutex = threading.Lock()
        self.__data = data

    def get_data(self: Self) -> A:
        self.__mutex.acquire()
        data = self.__data
        self.__mutex.release()
        return data

    def set_data(self: Self, data: A) -> None:
        self.__mutex.acquire()
        self.__data = data
        self.__mutex.release()

    def modify_data(self: Self, fn: Callable[[A], A]) -> None:
        self.__mutex.acquire()
        self.__data = fn(self.__data)
        self.__mutex.release()

    def ctx(self: Self) -> AbstractContextManager[ThreadSafeAcquired[A]]:
        return self

    @override
    def __enter__(self: Self) -> ThreadSafeAcquired[A]:
        self.__mutex.acquire()

        def set_fn(d: A) -> None:
            self.__data = d

        return ThreadSafeAcquired(lambda: self.__data, set_fn)

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        self.__mutex.release()
        return False


@dataclass
class ThreadState:
    thread: threading.Thread
    event: asyncio.Event


@dataclass
class ScannerThreadState:
    state: ScannerState
    thread: Optional[ThreadState]


def run_in_thread(
    self: "BackendScanner",
    configs: list[FinalConfig],
    event: asyncio.Event,
) -> None:
    asyncio.run(self.start_run_async(configs))
    event.set()


class BackendScanner:
    __manager: ScannerManager
    __all_configs: list[FinalConfig]

    __state: ThreadSafe[ScannerThreadState]

    def __init__(
        self: Self,
        configs: list[FinalConfig],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.__all_configs = configs
        self.__manager = ScannerManager(loop=loop)
        self.__state = ThreadSafe[ScannerThreadState](
            ScannerThreadState(state={"type": "idle"}, thread=None),
        )

    def manager_ctx(self: Self, websocket: WebSocket) -> ManagerCtx:
        return self.__manager.ctx(websocket)

    async def __launch_scanner_in_background(
        self: Self,
        config: FinalConfig,
        name_parser: NameParser,
        all_content_type: AnyType,
        config_paramaters: Optional[tuple[int, int]],
        manager: ScannerManager,
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

        language_picker: LanguagePicker = get_picker_from_config(
            config.picker,
            manager,
        )

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
        self: Self,
        configs: list[FinalConfig],
        manager: ScannerManager,
    ) -> list[SummaryTuple]:
        result: list[SummaryTuple] = []

        # TODO: set current configs and configs to process, support arguments
        for index, config in enumerate(configs):
            name_parser = CustomNameParser(season_special_names=config.parser.special)

            config_paramaters: Optional[tuple[int, int]] = (
                None if len(configs) == 1 else (index, len(configs))
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

    async def start_run_async(
        self: Self,
        configs: list[FinalConfig],
    ) -> None:
        try:
            result: list[SummaryTuple] = await self.__start_coroutine(
                configs=configs,
                manager=self.__manager,
            )

            self.__state.modify_data(
                lambda d: ScannerThreadState(
                    state={"type": "finished", "result": result},
                    thread=d.thread,
                ),
            )
        except BaseException as err:
            print("RUN ASYNC err: ", err)
            self.__state.modify_data(
                lambda d: ScannerThreadState(
                    state={"type": "error", "error": err},
                    thread=d.thread,
                ),
            )

    def __start_impl(
        self: Self,
        configs: list[FinalConfig],
        run_in_background: Callable[[Callable[[], Coroutine[Any, Any, Any]]], None],
    ) -> Optional[str]:

        with self.__state.ctx() as ctx:
            state = ctx.get()
            if state.state["type"] == "running" or state.thread is not None:
                return "Scanner is already running"

            event = asyncio.Event()

            thread = threading.Thread(
                target=run_in_thread,
                args=(self, configs, event),
            )

            new_state: ScannerThreadState = ScannerThreadState(
                state={"type": "running", "configs": configs},
                thread=ThreadState(thread=thread, event=event),
            )

            async def start_and_wait_for_thread() -> None:
                thread.start()
                await event.wait()

                with self.__state.ctx() as ctx:
                    state = ctx.get()
                    # cleanup the thread
                    if state.thread is not None:
                        await state.thread.event.wait()
                        ctx.modify(
                            lambda d: ScannerThreadState(state=d.state, thread=None),
                        )

            run_in_background(start_and_wait_for_thread)

            ctx.set(new_state)

        return None

    def start(
        self: Self,
        cfg_filter: Optional[ConfigFilter],
        run_in_background: Callable[[Callable[[], Coroutine[Any, Any, Any]]], None],
    ) -> Optional[str]:
        if cfg_filter is None:
            return self.__start_impl(
                configs=self.__all_configs,
                run_in_background=run_in_background,
            )
        try:
            configs = filter_configs(configs=self.__all_configs, cfg_filter=cfg_filter)
            return self.__start_impl(
                configs=configs,
                run_in_background=run_in_background,
            )
        except RuntimeError as err:
            raise HTTPException(status_code=400, detail=str(err)) from None

    def status(self: Self) -> dict[str, Any]:
        state = self.__state.get_data()
        match state.state["type"]:
            case "error":
                return {"state": "error", "error": str(state.state["error"])}
            case _:
                return cast(dict[str, Any], state.state)


class Backend:
    __options: BackendOptions
    __server: uvicorn.Server
    __ready: asyncio.Event
    __scanner: BackendScanner

    def __init__(
        self: Self,
        options: BackendOptions,
        configs: list[FinalConfig],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.__options = options
        self.__scanner = BackendScanner(configs=configs, loop=loop)
        self.__ready = asyncio.Event()

        app = FastAPI(dependencies=[Depends(self.ready)], strict_content_type=True)
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

    def schedule_shutdown(self: Self) -> None:
        if not self.__server:
            return

        self.__server.should_exit = True

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


async def start_all(options: BackendOptions, configs: list[FinalConfig]) -> int:
    loop = asyncio.get_running_loop()

    backend = Backend(options=options, configs=configs, loop=loop)

    await backend.run()
    return 0


def launch_api(options: BackendOptions, configs: list[FinalConfig]) -> int:
    return asyncio.run(start_all(options, configs))
