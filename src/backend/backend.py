import asyncio
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager
from dataclasses import dataclass
from logging import Formatter, Handler, Logger, LogRecord
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    Self,
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
from content.summary import LanguageDict, MetadataDict, MetadataSubDict, Summary
from helper.base import (
    AnyType,
    parse_contents,
)
from helper.classifier import Classifier, Model, voxlingua107_ecapa_model
from helper.config import ConfigFilter, ConfigFilterItem, FinalConfig, filter_configs
from helper.devices import DeviceManager
from helper.log import get_logger
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
    async def process_data(self: Self, data: Any) -> Optional[ProcessResult]: ...

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


DEFAULT_MODEL_CONFIG: pydantic.ConfigDict = pydantic.ConfigDict(
    extra="forbid",
    strict=True,
)


class ManagerWsChoiceMessageAskQuestionReply(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["reply"] = "reply"
    reply: Literal["ask_question"]
    result: Optional["ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult"]
    id: str


IncomingWsData = ManagerWsChoiceMessageAskQuestionReply


ProcessResultTyped = Result["OutgoingWsData", str]


class WsSingleManager(WebsocketHandler):
    __parent_ref: "WsManager"

    def __init__(
        self: Self,
        parent_ref: "WsManager",
        websocket: WebSocket,
    ) -> None:
        super().__init__(websocket=websocket)
        self.__parent_ref = parent_ref

    async def __process_data(
        self: Self,
        data: IncomingWsData,
    ) -> Optional[ProcessResultTyped]:
        match data.type:
            case "reply":
                match data.reply:
                    case "ask_question":
                        res_data: Optional[SelectResult] = (
                            None
                            if data.result is None
                            else deserialize_select_result(data.result)
                        )
                        unique_id: uuid.UUID = uuid.UUID(data.id)
                        result: Optional[str] = (
                            self.__parent_ref.process_ask_question_reply(
                                result=res_data,
                                unique_id=unique_id,
                            )
                        )
                        if result is None:
                            response = ManagerWsChoiceMessage(
                                data=ManagerWsChoiceMessageQuestionReplyReceivedData(
                                    id=data.id,
                                ),
                            )
                            return ProcessResultTyped.ok(response)

                        return ProcessResultTyped.err(result)
                    case _:
                        assert_never(data.reply)
            case _:
                assert_never(data.type)

    @override
    async def process_data(self: Self, data: Any) -> Optional[ProcessResult]:
        typed_data: IncomingWsData = IncomingWsData.model_validate(
            obj=data,
            strict=True,
            extra="forbid",
        )
        return await self.__process_data(typed_data)


class ManagerWsGlobalMessageStopData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["stop"] = "stop"


class CounterTypeCounter(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["counter"] = "counter"
    options: CounterOptions


class CounterTypeStatusBar(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["status_bar"] = "status_bar"
    options: StatusBarGetOptions


CounterType = Literal["counter", "status_bar"]

CounterInstanceType = Annotated[
    CounterTypeCounter | CounterTypeStatusBar,
    pydantic.Discriminator(discriminator="type"),
]


class ManagerWsGlobalMessageCounterData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["counter"] = "counter"
    counter: CounterInstanceType
    idx: int


class ManagerWsGlobalCounterMessage(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["global_counter"] = "global_counter"
    data: Annotated[
        ManagerWsGlobalMessageCounterData | ManagerWsGlobalMessageStopData,
        pydantic.Discriminator(discriminator="type"),
    ]


class CounterUpdateOptions(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    incr: NumberLike  # = 1
    force: bool  # = False


class CounterUpdateTypeCounter(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["counter"] = "counter"
    options: CounterUpdateOptions


class CounterUpdateTypeStatusBar(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["status_bar"] = "status_bar"
    options: StatusBarInterfaceUpdateOptions


class ManagerWsCounterMessageUpdateData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["update"] = "update"
    idx: int
    options: Annotated[
        CounterUpdateTypeCounter | CounterUpdateTypeStatusBar,
        pydantic.Discriminator(discriminator="type"),
    ]


class CounterMessageCloseOptions(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    clear: bool


class ManagerWsCounterMessageCloseData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG
    type: Literal["close"] = "close"
    idx: int
    options: CounterMessageCloseOptions


class ManagerWsCounterMessage(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["counter"] = "counter"
    data: Annotated[
        ManagerWsCounterMessageUpdateData | ManagerWsCounterMessageCloseData,
        pydantic.Discriminator(discriminator="type"),
    ]


class ManagerWsChoiceMessageAskQuestionChoiceDataSeparator(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    tag: Literal["separator"] = "separator"


type ChoiceColorTypeValues = Literal["fg", "bg"]


type ChoiceColorValueValues = Literal["ansiblue", "ansigreen"]


class ChoiceColorSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: ChoiceColorTypeValues
    color: ChoiceColorValueValues


def choice_color_to_serializable_data(
    choice_color: ChoiceColor,
) -> ChoiceColorSerializable:
    return ChoiceColorSerializable(
        type=choice_color.type.value,
        color=choice_color.color.value,
    )


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    color: Optional[ChoiceColorSerializable]
    content: str


def choice_title_to_serializable_data(
    title: ChoiceTitle,
) -> ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle:
    return ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle(
        content=title.content,
        color=(
            None
            if title.color is None
            else choice_color_to_serializable_data(title.color)
        ),
    )


type SelectedTypeValues = Literal["open", "no_language", " copy", "more", "unknown"]


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual(
    pydantic.BaseModel,
):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["manual"] = "manual"
    selected: SelectedTypeValues


class LanguageSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    short: str
    long: str


def language_to_serializable_data(language: Language) -> LanguageSerializable:
    return LanguageSerializable(short=language.short, long=language.long)


def deserialize_language(language: LanguageSerializable) -> Language:
    return Language(short=language.short, long=language.long)


class PredictionBestSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    accuracy: float
    language: LanguageSerializable


def prediction_best_to_serializable_data(
    data: PredictionBest,
) -> PredictionBestSerializable:
    return PredictionBestSerializable(
        accuracy=data.accuracy,
        language=language_to_serializable_data(data.language),
    )


def deserialize_prediction_best(data: PredictionBestSerializable) -> PredictionBest:
    return PredictionBest(
        accuracy=data.accuracy,
        language=deserialize_language(data.language),
    )


class ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest(
    pydantic.BaseModel,
):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["prediction_best"] = "prediction_best"
    value: PredictionBestSerializable


ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult = Annotated[
    (
        ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual
        | ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest
    ),
    pydantic.Discriminator(discriminator="type"),
]


def select_result_to_serializable_data(
    result: SelectResult,
) -> ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult:
    if isinstance(result, ManualSelectResult):
        manual: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual = (
            ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultManual(
                selected=result.selected.value,
            )
        )
        return manual
    if isinstance(result, PredictionBestSelectResult):
        value = prediction_best_to_serializable_data(result.value)

        choice: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest = (
            ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResultBest(
                value=value,
            )
        )
        return choice
    assert_never(result)


def deserialize_select_result(
    result: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult,
) -> SelectResult:
    match result.type:
        case "manual":
            manual: ManualSelectResult = ManualSelectResult(
                select_result_type="manual",
                selected=SelectedType(value=result.selected),
            )
            return manual
        case "prediction_best":
            value = deserialize_prediction_best(data=result.value)
            best: PredictionBestSelectResult = PredictionBestSelectResult(
                select_result_type="prediction_best",
                value=value,
            )
            return best
        case _:
            assert_never(result)


class ManagerWsChoiceMessageAskQuestionChoiceDataChoice(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    tag: Literal["choice"] = "choice"
    title: list[ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectTitle]
    value: ManagerWsChoiceMessageAskQuestionChoiceDataChoiceSelectResult


ManagerWsChoiceMessageAskQuestionChoiceData = Annotated[
    (
        ManagerWsChoiceMessageAskQuestionChoiceDataSeparator
        | ManagerWsChoiceMessageAskQuestionChoiceDataChoice
    ),
    pydantic.Discriminator(discriminator="tag"),
]


def choice_to_serializable_data(
    data: "WSChoice",
) -> ManagerWsChoiceMessageAskQuestionChoiceData:
    if isinstance(data, WSChoiceSeparator):
        separator: ManagerWsChoiceMessageAskQuestionChoiceDataSeparator = (
            ManagerWsChoiceMessageAskQuestionChoiceDataSeparator()
        )
        return separator
    if isinstance(data, WSChoiceChoice):
        impl = data.impl

        title = [choice_title_to_serializable_data(segement) for segement in impl.title]
        value = select_result_to_serializable_data(impl.value)

        choice: ManagerWsChoiceMessageAskQuestionChoiceDataChoice = (
            ManagerWsChoiceMessageAskQuestionChoiceDataChoice(
                title=title,
                value=value,
            )
        )
        return choice
    assert_never(data)


class ManagerWsChoiceMessageAskQuestionData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["ask_question"] = "ask_question"
    message: str
    choices: list[ManagerWsChoiceMessageAskQuestionChoiceData]
    default: ManagerWsChoiceMessageAskQuestionChoiceData
    id: str


class ManagerWsChoiceMessageQuestionReplyReceivedData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["reply_received"] = "reply_received"
    id: str


class ManagerWsChoiceMessage(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["choice"] = "choice"
    data: Annotated[
        ManagerWsChoiceMessageAskQuestionData
        | ManagerWsChoiceMessageQuestionReplyReceivedData,
        pydantic.Discriminator(discriminator="type"),
    ]


type LogLevelStr = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


class ManagerWsLogMessageEventData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["event"] = "event"
    level: LogLevelStr
    message: str
    asctime: str
    module: str


class ManagerWsLogMessage(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["log"] = "log"
    data: Annotated[
        ManagerWsLogMessageEventData,
        pydantic.Discriminator(discriminator="type"),
    ]


type ScannersStateStr = Literal["idle", "running", "finished", "error"]


class ManagerWsScannerMessageStatusChangedData(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["status_changed"] = "status_changed"
    previous: ScannersStateStr
    new: ScannersStateStr


class ManagerWsScannerMessage(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["scanner"] = "scanner"
    data: Annotated[
        ManagerWsScannerMessageStatusChangedData,
        pydantic.Discriminator(discriminator="type"),
    ]


OutgoingWsData = Annotated[
    (
        ManagerWsGlobalCounterMessage
        | ManagerWsCounterMessage
        | ManagerWsChoiceMessage
        | ManagerWsLogMessage
        | ManagerWsScannerMessage
    ),
    pydantic.Discriminator(discriminator="type"),
]


class ScannerStatusBar(StatusBarInterface):
    __ref: "WsManager"
    __idx: int

    def __init__(self: Self, ref: "WsManager", idx: int) -> None:
        super().__init__()
        self.__ref = ref
        self.__idx = idx

    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        data: ManagerWsCounterMessage = ManagerWsCounterMessage(
            data=ManagerWsCounterMessageUpdateData(
                idx=self.__idx,
                options=CounterUpdateTypeStatusBar(options=fields),
            ),
        )
        self.__ref.send_data_sync(data)


class ScannerCounter(CounterInterface):
    __ref: "WsManager"
    __idx: int

    def __init__(self: Self, ref: "WsManager", idx: int) -> None:
        super().__init__()
        self.__ref = ref
        self.__idx = idx

    @override
    def update(self: Self, incr: NumberLike = 1, *, force: bool = False) -> None:
        data: ManagerWsCounterMessage = ManagerWsCounterMessage(
            data=ManagerWsCounterMessageUpdateData(
                type="update",
                idx=self.__idx,
                options=CounterUpdateTypeCounter(
                    options=CounterUpdateOptions(
                        force=force,
                        incr=number_like_convert_to_serializable(incr),
                    ),
                ),
            ),
        )
        self.__ref.send_data_sync(data)

    @override
    def close(self: Self, *, clear: bool = False) -> None:
        data: ManagerWsCounterMessage = ManagerWsCounterMessage(
            data=ManagerWsCounterMessageCloseData(
                idx=self.__idx,
                options=CounterMessageCloseOptions(clear=clear),
            ),
        )
        self.__ref.send_data_sync(data)


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
    data: Optional[Any]


class ManagerCtx(AbstractContextManager[WsSingleManager]):
    __manager: WsSingleManager
    __remove_fn: Callable[[], None]

    def __init__(
        self: Self,
        manager: WsSingleManager,
        remove_fn: Callable[[], None],
    ) -> None:
        super().__init__()
        self.__manager = manager
        self.__remove_fn = remove_fn

    @override
    def __enter__(self: Self) -> WsSingleManager:
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


class WsManager(ManagerInterface, ChoiceManagerInterface):
    __instances: list[WsSingleManager]
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
        manager = WsSingleManager(self, websocket)
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
        self.__counters.append(instance.type)
        instance_serializable: CounterInstanceType

        if instance.type == "counter":
            serializable_options1: CounterOptions = {**instance.options}
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
            instance_serializable = CounterTypeCounter(
                options=serializable_options1,
            )
        else:
            serializable_options2: StatusBarGetOptions = {**instance.options}

            instance_serializable = CounterTypeStatusBar(
                options=serializable_options2,
            )
        data: ManagerWsGlobalCounterMessage = ManagerWsGlobalCounterMessage(
            data=ManagerWsGlobalMessageCounterData(
                counter=instance_serializable,
                idx=idx,
            ),
        )
        self.send_data_sync(data)
        return idx

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        options: CounterTypeStatusBar = CounterTypeStatusBar(options=kwargs)
        idx: int = self.__add_counter(options)
        return ScannerStatusBar(self, idx)

    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        options: CounterTypeCounter = CounterTypeCounter(options=kwargs)
        idx: int = self.__add_counter(options)
        return ScannerCounter(self, idx)

    @override
    def stop(
        self: Self,
    ) -> None:
        data: ManagerWsGlobalCounterMessage = ManagerWsGlobalCounterMessage(
            data=ManagerWsGlobalMessageStopData(),
        )
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
        if isinstance(choice, (WSChoiceChoice, WSChoiceSeparator)):
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
            choice_to_serializable_data(WsManager.__get_underlying_choice(choice))
            for choice in choices
        ]
        default_impl = choice_to_serializable_data(
            WsManager.__get_underlying_choice(default),
        )

        uid: uuid.UUID = uuid.uuid4()

        data: ManagerWsChoiceMessage = ManagerWsChoiceMessage(
            data=ManagerWsChoiceMessageAskQuestionData(
                message=message,
                choices=choices_impl,
                default=default_impl,
                id=str(uid),
            ),
        )

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
        unique_id: uuid.UUID,
    ) -> Optional[str]:
        reply_data = self.__reply_ids.get(unique_id, None)
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
    model_config = DEFAULT_MODEL_CONFIG

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

    @app.websocket("/ws/")
    async def ws(
        websocket: WebSocket,
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> None:
        await websocket.accept()
        with backend.manager.ctx(websocket) as manager:
            await manager.process()

    @app.get("/scan/start")
    async def scan_start(
        backend: Annotated[Backend, Depends(retreive_backend)],
        background_tasks: BackgroundTasks,
        start_query: Annotated[ScanStartQuery, Query()],
    ) -> Response:
        cfg_filter: Optional[ConfigFilter] = get_config_filters(start_query.filter)

        def run_in_background(fn: Callable[[], Coroutine[Any, Any, Any]]) -> None:
            background_tasks.add_task(fn)

        result: Optional[str] = backend.scanner.start(
            cfg_filter=cfg_filter,
            run_in_background=run_in_background,
            backend=backend,
        )

        if result is not None:
            return JSONResponse(status_code=422, content={"error": result})

        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/scan/status")
    async def scan_status(
        backend: Annotated[Backend, Depends(retreive_backend)],
    ) -> Response:
        status: ScanStatusSerializable = backend.scanner.status()
        return JSONResponse(
            status_code=200,
            content=ScanStatusSerializableResult(status=status),
        )


@dataclass
class Address:
    host: str
    port: int

    def __str__(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendOptions:
    address: Address


class ScannerStateIdle(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["idle"] = "idle"


class ScannerStateRunning(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["running"] = "running"
    configs: list[FinalConfig]


type SummaryTuple = tuple[LanguageDict, MetadataDict, ScanSummaryDetailed]

# TODO: use the short string
type LongLanguageStr = str

type LanguageDictSerializable = dict[LongLanguageStr, int]


class MetadataSubDictSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    ok: int
    missing: int
    skipped: int


class MetadataDictSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    series: MetadataSubDictSerializable
    season: MetadataSubDictSerializable
    episode: MetadataSubDictSerializable


class ScanSummaryDetailedSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    success: LanguageDictSerializable
    failure: dict[str, int]


class SummaryTupleSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    language: LanguageDictSerializable
    metadata: MetadataDictSerializable
    details: ScanSummaryDetailedSerializable


def language_dict_to_serializable_data(
    language_dict: LanguageDict,
) -> LanguageDictSerializable:
    return {k.long: v for k, v in language_dict.items()}


def scan_summary_detailed_to_serializable_data(
    obj: ScanSummaryDetailed,
) -> ScanSummaryDetailedSerializable:
    return ScanSummaryDetailedSerializable(
        success=language_dict_to_serializable_data(obj.success),
        failure=obj.failure,
    )


def metadata_sub_dict_to_serializable_data(
    obj: MetadataSubDict,
) -> MetadataSubDictSerializable:
    return cast(MetadataSubDictSerializable, {k.value: v for k, v in obj.items()})


def metadata_dict_to_serializable_data(
    obj: MetadataDict,
) -> MetadataDictSerializable:
    return cast(
        MetadataDictSerializable,
        {k.value: metadata_sub_dict_to_serializable_data(v) for k, v in obj.items()},
    )


def summary_tuple_to_serializable_data(
    summary: SummaryTuple,
) -> SummaryTupleSerializable:
    return SummaryTupleSerializable(
        language=language_dict_to_serializable_data(summary[0]),
        metadata=metadata_dict_to_serializable_data(summary[1]),
        details=scan_summary_detailed_to_serializable_data(summary[2]),
    )


@dataclass
class ScannerStateFinished:
    type: Literal["finished"]
    result: list[SummaryTuple]


class ScannerStateFinishedSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["finished"] = "finished"
    result: list[SummaryTupleSerializable]


@dataclass
class ScannerStateError:
    type: Literal["error"]
    error: str | BaseException


class ScannerStateErrorSerializable(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    type: Literal["error"] = "error"
    error: str


ScannerState = (
    ScannerStateIdle | ScannerStateRunning | ScannerStateFinished | ScannerStateError
)


ScanStatusSerializable = Annotated[
    (
        ScannerStateIdle
        | ScannerStateRunning
        | ScannerStateFinishedSerializable
        | ScannerStateErrorSerializable
    ),
    pydantic.Discriminator(discriminator="type"),
]


def scanner_state_to_serializable_data(
    state: ScannerState,
) -> ScanStatusSerializable:
    match state.type:
        case "error":
            return ScannerStateErrorSerializable(error=str(state.error))
        case "finished":
            return ScannerStateFinishedSerializable(
                result=[
                    summary_tuple_to_serializable_data(item) for item in state.result
                ],
            )
        case _:
            return state


class ScanStatusSerializableResult(pydantic.BaseModel):
    model_config = DEFAULT_MODEL_CONFIG

    status: ScanStatusSerializable


class ThreadSafeAcquired[A]:
    __get_impl: Callable[[], A]
    __set_impl: Callable[[A], None]

    def __init__(
        self: Self,
        get_fn: Callable[[], A],
        set_fn: Callable[[A], None],
    ) -> None:
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


type ThreadId = int


class ThreadHandler(Handler):
    __send: Callable[[ManagerWsLogMessageEventData], None]
    __thread_id: ThreadId

    def __init__(
        self: Self,
        send: Callable[[ManagerWsLogMessageEventData], None],
        thread_id: ThreadId,
    ) -> None:
        super().__init__()
        self.__send = send
        self.__thread_id = thread_id

    def emit(self, record: LogRecord) -> None:
        if threading.current_thread().native_id != self.__thread_id:
            return
        try:

            msg = self.format(record)

            message: ManagerWsLogMessageEventData = ManagerWsLogMessageEventData(
                level=cast(LogLevelStr, record.levelname),
                message=msg,
                asctime=record.asctime,
                module=record.module,
            )
            self.__send(message)

        except RecursionError:
            raise
        except Exception:  # noqa: BLE001
            self.handleError(record)


class ThreadLoggerCtx(AbstractContextManager[Logger]):
    __handlers: Optional[list[Handler]]
    __send: Callable[[ManagerWsLogMessage], None]
    __thread_id: ThreadId

    def __init__(
        self: Self,
        send: Callable[[ManagerWsLogMessage], None],
        thread_id: ThreadId,
    ) -> None:
        super().__init__()
        self.__handlers = None
        self.__send = send
        self.__thread_id = thread_id

    @override
    def __enter__(self: Self) -> Logger:
        logger = get_logger()
        self.__handlers = logger.handlers

        logger.handlers = []

        def send(data: ManagerWsLogMessageEventData) -> None:
            message: ManagerWsLogMessage = ManagerWsLogMessage(data=data)
            self.__send(message)

        thread_handler = ThreadHandler(send=send, thread_id=self.__thread_id)

        formatter = Formatter(
            fmt="%(message)s",
            style="%",
            validate=True,
        )

        thread_handler.setFormatter(formatter)

        logger.addHandler(hdlr=thread_handler)

        return logger

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        if self.__handlers is not None:
            logger = get_logger()
            logger.handlers = self.__handlers
            self.__handlers = None

        return False


def run_in_thread(
    self: "BackendScanner",
    configs: list[FinalConfig],
    event: asyncio.Event,
    backend: "Backend",
) -> None:

    with backend.thread_logger():
        asyncio.run(self.start_run_async(configs=configs, backend=backend))

    event.set()


class BackendScanner:
    __all_configs: list[FinalConfig]

    __state: ThreadSafe[ScannerThreadState]

    def __init__(
        self: Self,
        configs: list[FinalConfig],
    ) -> None:
        self.__all_configs = configs
        self.__state = ThreadSafe[ScannerThreadState](
            ScannerThreadState(state=ScannerStateIdle(), thread=None),
        )

    async def __launch_scanner_in_background(
        self: Self,
        config: FinalConfig,
        name_parser: NameParser,
        all_content_type: AnyType,
        config_paramaters: Optional[tuple[int, int]],
        manager: WsManager,
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
        manager: WsManager,
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
        backend: "Backend",
    ) -> None:

        def on_status_change(previous: ScannersStateStr, new: ScannersStateStr) -> None:
            data: ManagerWsScannerMessage = ManagerWsScannerMessage(
                data=ManagerWsScannerMessageStatusChangedData(
                    previous=previous,
                    new=new,
                ),
            )
            backend.manager.send_data_sync(data)

        try:
            result: list[SummaryTuple] = await self.__start_coroutine(
                configs=configs,
                manager=backend.manager,
            )

            def mod(d: ScannerThreadState) -> ScannerThreadState:
                on_status_change(d.state.type, "finished")
                return ScannerThreadState(
                    state=ScannerStateFinished(type="finished", result=result),
                    thread=d.thread,
                )

            self.__state.modify_data(mod)
        except BaseException as err:  # noqa: BLE001

            # this indirection if for ruff, that doesn't recognize, that if i just define mod, err is captured and usable, mypy does recognize it :)
            def mod_helper(
                err: BaseException,
            ) -> Callable[[ScannerThreadState], ScannerThreadState]:

                def mod(d: ScannerThreadState) -> ScannerThreadState:
                    on_status_change(d.state.type, "error")
                    return ScannerThreadState(
                        state=ScannerStateError(type="error", error=err),
                        thread=d.thread,
                    )

                return mod

            self.__state.modify_data(mod_helper(err))

    def __start_impl(
        self: Self,
        configs: list[FinalConfig],
        run_in_background: Callable[[Callable[[], Coroutine[Any, Any, Any]]], None],
        backend: "Backend",
    ) -> Optional[str]:

        with self.__state.ctx() as ctx:
            state = ctx.get()
            if state.state.type == "running" or state.thread is not None:
                return "Scanner is already running"

            event = asyncio.Event()

            thread = threading.Thread(
                target=run_in_thread,
                args=(self, configs, event, backend.manager),
            )

            new_state: ScannerThreadState = ScannerThreadState(
                state=ScannerStateRunning(configs=configs),
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
                        # NOTE: now ManagerWsScannerMessageStatusChanged message needed, as no state is changed, only the internal thread
                        ctx.modify(
                            lambda d: ScannerThreadState(state=d.state, thread=None),
                        )

            run_in_background(start_and_wait_for_thread)

            ctx.set(new_state)
            data: ManagerWsScannerMessage = ManagerWsScannerMessage(
                data=ManagerWsScannerMessageStatusChangedData(
                    previous=state.state.type,
                    new=new_state.state.type,
                ),
            )
            backend.manager.send_data_sync(data)

        return None

    def start(
        self: Self,
        cfg_filter: Optional[ConfigFilter],
        run_in_background: Callable[[Callable[[], Coroutine[Any, Any, Any]]], None],
        backend: "Backend",
    ) -> Optional[str]:
        if cfg_filter is None:
            return self.__start_impl(
                configs=self.__all_configs,
                run_in_background=run_in_background,
                backend=backend,
            )
        try:
            configs = filter_configs(configs=self.__all_configs, cfg_filter=cfg_filter)
            return self.__start_impl(
                configs=configs,
                run_in_background=run_in_background,
                backend=backend,
            )
        except RuntimeError as err:
            raise HTTPException(status_code=400, detail=str(err)) from None

    # TODO: type correctly
    def status(self: Self) -> ScanStatusSerializable:
        state = self.__state.get_data()
        return scanner_state_to_serializable_data(state.state)


class Backend:
    __options: BackendOptions
    __server: uvicorn.Server
    __ready: asyncio.Event
    __scanner: BackendScanner
    __manager: WsManager

    def __init__(
        self: Self,
        options: BackendOptions,
        configs: list[FinalConfig],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.__options = options
        self.__manager = WsManager(loop=loop)
        self.__scanner = BackendScanner(configs=configs)
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

    @property
    def manager(self: Self) -> WsManager:
        return self.__manager

    def thread_logger(self: Self) -> ThreadLoggerCtx:

        def send_log_data(data: ManagerWsLogMessage) -> None:
            self.__manager.send_data_sync(data)

        thread_id: Optional[int] = threading.current_thread().native_id

        if thread_id is None:
            msg = "Thread Id is undefined, how could this happen?"
            raise RuntimeError(msg)

        return ThreadLoggerCtx(
            send=send_log_data,
            thread_id=thread_id,
        )

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


def suppress_logs() -> None:
    logger = get_logger()

    logger.handlers = []


def launch_api(options: BackendOptions, configs: list[FinalConfig]) -> int:
    suppress_logs()
    return asyncio.run(start_all(options, configs))
