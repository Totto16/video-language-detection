import json
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Literal,
    Optional,
    Self,
    TypedDict,
    assert_never,
    override,
)

from apischema import deserialize, serialize

from content.base_class import (
    CallbackData,
    Content,
    ContentCharacteristic,
    DefaultStatusBarInfo,
    Scanner,
    StatusBarInfo,
    StatusBarInfoRaw,
    process_folder,
)
from content.general import (
    Callback,
    ContentType,
    NameParser,
    ScannedFileType,
    StartAmount,
)
from content.language_picker import LanguagePicker
from content.metadata.metadata import HandlesType
from content.scan_helpers import normal_content_from_scan, numerated_content_from_scan
from helper.config import ConfigType, FinalConfig, ParsedTargetFile
from helper.constants import APP_NAME
from helper.decorator import decorate_class
from helper.error import ErrorMode
from helper.manager import (
    CounterInterface,
    ManagerInterface,
    ManagerJustify,
    StatusBarInterface,
)
from helper.translation import get_translator

_ = get_translator()

type AnyType = Any


def save_to_file(
    tgt: ParsedTargetFile,
    contents: list[Content],
    serialize_type: AnyType,
) -> None:
    if not tgt.file.parent.exists():
        tgt.file.parent.mkdir(parents=True)

    if tgt.type == "json":

        encoded_dict: dict[str, Any] = serialize(
            list[serialize_type],
            contents,
        )

        with tgt.file.open(
            mode="w",
        ) as file:
            suffix: str = tgt.file.suffix[1:]
            match suffix:
                case "json":
                    json.dump(encoded_dict, file, indent=4, ensure_ascii=False)
                case _:
                    msg = _("Data not saveable to '{suffix}' file!").format(
                        suffix=suffix,
                    )
                    raise RuntimeError(msg)

        return

    msg = f"file type '{tgt.type}' is not implemented yet"
    raise NotImplementedError(msg)


def load_from_file(
    tgt: ParsedTargetFile,
    serialize_type: AnyType,
) -> list[Content]:
    if tgt.type == "json":
        with tgt.file.open(mode="r") as file:
            suffix: str = tgt.file.suffix[1:]
            match suffix:
                case "json":
                    parsed_dict: dict[str, Any] = json.load(file)
                    json_loaded: list[Content] = deserialize(
                        list[serialize_type],
                        parsed_dict,
                    )
                    return json_loaded
                case _:
                    msg = _("Data not loadable from '{suffix}' file!").format(
                        suffix=suffix,
                    )
                    raise RuntimeError(msg)

    msg = f"file type '{tgt.type}' is not implemented yet"
    raise NotImplementedError(msg)


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    trailer_names: list[str]
    parse_error_is_exception: bool


@decorate_class(slots=True)
class AppStatusBar:
    __manager: ManagerInterface
    __status_bar: StatusBarInterface
    __stopped: bool
    __length: Optional[int]
    __info_kw: dict[str, str]

    def __init__(
        self: Self,
        configs: list[FinalConfig],
        manager: ManagerInterface,
    ) -> None:
        super().__init__()

        self.__manager = manager
        self.__stopped = False
        self.__length = None if len(configs) == 1 else len(configs)

        info_str: str = ""

        info_parts: list[tuple[str, str]] = [
            (f"info_{i}", _("<Starting>"))
            for i in range(2 if self.__length is None else 3)
        ]
        info_kw = dict(info_parts)
        info_str = "{fill}".join(f"{{{x[0]}}}" for x in info_parts) + "{fill}"

        info_kw["stage"] = _("Idle")

        self.__info_kw = info_kw

        self.__status_bar = self.__manager.status_bar(
            status_format=APP_NAME
            + "{fill}"
            + _("Stage")
            + ": {stage}{fill}"
            + info_str
            + "{elapsed}",
            color="bold_underline_bright_white_on_blue",
            justify=ManagerJustify.CENTER,
            autorefresh=True,
            min_delta=0.5,
            additional_args=self.__info_kw,
        )

    def __update_info(self: Self) -> None:
        self.__status_bar.update(additional_args=self.__info_kw)

    @property(fget=None).setter
    def stage(self: Self, stage: str) -> None:
        self.__info_kw["stage"] = stage
        self.__update_info()

    def __unset_config(self: Self) -> None:
        info_parts: list[tuple[str, str]] = [
            (f"info_{i}", _("<Not set>"))
            for i in range(2 if self.__length is None else 3)
        ]

        for key, value in info_parts:
            self.__info_kw[key] = value

        self.__update_info()

    def __set_config(self: Self, index: int, config: FinalConfig) -> None:
        config_paramaters: Optional[tuple[int, int]] = (
            None if self.__length is None else (index, self.__length)
        )

        general_info: list[str] = [
            x
            for x in [
                _("Config: '{config_name}'").format(
                    config_name=config.config_name,
                ),
                (
                    None
                    if config_paramaters is None
                    else _("Config progress: {start} / {end}").format(
                        start=config_paramaters[0] + 1,
                        end=config_paramaters[1],
                    )
                ),
                _("Config type: '{config_type}'").format(
                    config_type=config.config_type.value,
                ),
            ]
            if x is not None
        ]

        expected_length = 2 if self.__length is None else 3

        if len(general_info) != expected_length:
            msg = f"Invalid config set:  {len(general_info)}  != {expected_length}"
            raise RuntimeError(msg)

        info_parts: list[tuple[str, str]] = [
            (f"info_{i}", general_info[i]) for i in range(len(general_info))
        ]

        for key, value in info_parts:
            self.__info_kw[key] = value

        self.__update_info()

    def config(
        self: Self,
        index: int,
        config: FinalConfig,
    ) -> AbstractContextManager[None]:
        def enter_cb() -> None:
            self.__set_config(index, config)

        def exit_cb() -> None:
            self.__unset_config()

        @decorate_class(slots=True)
        class ConfigContextWrapper(AbstractContextManager[None]):

            def __init__(self: Self) -> None:
                super().__init__()

            @override
            def __enter__(self: Self) -> None:
                enter_cb()

            @override
            def __exit__(
                self: Self,
                exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                exit_cb()
                return False

        return ConfigContextWrapper()

    def stop(self: Self) -> None:
        if not self.__stopped:
            self.__stopped = True
            self.stage = _("End")
            self.__manager.stop()

    def __del__(self: Self) -> None:
        with suppress(BaseException):
            self.stop()


def app_status_bar_manager(
    configs: list[FinalConfig],
    manager: ManagerInterface,
) -> AbstractContextManager[AppStatusBar]:
    @decorate_class(slots=True)
    class AppStatusBarContextManager(AbstractContextManager[AppStatusBar]):
        __status_bar: Optional[AppStatusBar]

        def __init__(self: Self) -> None:
            super().__init__()

            self.__status_bar = None

        @override
        def __enter__(self: Self) -> AppStatusBar:
            self.__status_bar = AppStatusBar(configs, manager)

            return self.__status_bar

        @override
        def __exit__(
            self: Self,
            _exc_type: Optional[type[BaseException]],
            _exc_val: Optional[BaseException],
            _exc_tb: Optional[TracebackType],
        ) -> Literal[False]:  # actually bool
            if self.__status_bar is not None:
                self.__status_bar.stop()
                self.__status_bar = None

            return False

    return AppStatusBarContextManager()


@decorate_class(slots=True)
class StatusBarManager:
    __progress_bars: dict[str, CounterInterface]
    __manager: ManagerInterface

    def __init__(
        self: Self,
        manager: ManagerInterface,
    ) -> None:
        super().__init__()

        self.__manager = manager
        self.__progress_bars = {}

    @property
    def manager(self: Self) -> ManagerInterface:
        return self.__manager

    def start(
        self: Self,
        amount: StartAmount,
        name: str,
        info: StatusBarInfo,
    ) -> None:
        value: StatusBarInfoRaw

        match info:
            case tuple():
                value = info
            case ContentType.collection:
                value = ("blue", _("series"))
            case ContentType.series:
                value = ("cyan", _("seasons"))
            case ContentType.season:
                value = ("green", _("episodes"))
            case ContentType.episode:
                value = ("yellow", _("tasks"))
            case _:
                value = ("purple", _("folders"))

        color, unit = value

        self.__progress_bars[name] = self.__manager.counter(
            total=amount.processing,
            desc=name,
            unit=unit,
            leave=False,
            color=color,
        )
        self.__progress_bars[name].update(0, force=True)

    def progress(
        self: Self,
        name: str,
        *,
        amount: int,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            msg = _("No Progressbar, on progress callback")
            raise RuntimeError(msg)

        self.__progress_bars[name].update(amount)

    def finish(
        self: Self,
        name: str,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            msg = _("No Progressbar, on progress finish")
            raise RuntimeError(msg)

        self.__progress_bars[name].close(clear=True)
        del self.__progress_bars[name]


@decorate_class(slots=True)
class ContentCallback(Callback[Content, ContentCharacteristic, CallbackData]):
    __options: ContentOptions
    __name_parser: NameParser
    __scanner: Scanner
    __status_bar_manager: StatusBarManager

    __language_picker: LanguagePicker
    __error_mode: ErrorMode

    def __init__(
        self: Self,
        options: ContentOptions,
        name_parser: NameParser,
        scanner: Scanner,
        language_picker: LanguagePicker,
        manager: ManagerInterface,
        error_mode: ErrorMode,
    ) -> None:
        super().__init__()

        self.__options = options
        self.__name_parser = name_parser
        self.__scanner = scanner

        self.__status_bar_manager = StatusBarManager(manager)

        self.__language_picker = language_picker
        self.__error_mode = error_mode

    @override
    def get_saved(self: Self) -> CallbackData:
        return CallbackData(
            manager=self.__status_bar_manager.manager,
            scanner=self.__scanner,
            language_picker=self.__language_picker,
            error_mode=self.__error_mode,
        )

    @override
    def ignore(
        self: Self,
        file_path: Path,
        file_type: ScannedFileType,
        parent_folders: list[str],
    ) -> bool:
        name: str = file_path.name
        if file_type == ScannedFileType.folder:
            if name.startswith("."):
                return True

            if name in self.__options["ignore_files"]:
                return True
        else:
            extension: str = file_path.suffix[1:]
            if extension not in self.__options["video_formats"]:
                return True

        return False

    @override
    def start(
        self: Self,
        amount: StartAmount,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        info, _i = characteristic.as_tuple()

        self.__status_bar_manager.start(amount, name, info)

    @override
    def progress(
        self: Self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
        *,
        amount: int,
    ) -> None:
        self.__status_bar_manager.progress(name, amount=amount)

    @override
    def finish(
        self: Self,
        name: str,
        parent_folders: list[str],
        deleted: int,
        characteristic: ContentCharacteristic,
    ) -> None:
        self.__status_bar_manager.finish(name)

    @property
    def name_parser(self: Self) -> NameParser:
        return self.__name_parser

    @property
    def options(self: Self) -> ContentOptions:
        return self.__options


@decorate_class(slots=True)
class NormalContentCallback(ContentCallback):
    @override
    def process(
        self: Self,
        file_path: Path,
        file_type: ScannedFileType,
        handles: HandlesType,
        parent_folders: list[str],
        *,
        trailer_names: list[str],
        rescan: Optional[Content] = None,
    ) -> Optional[Content]:
        manager, _scanner, _language_picker, _error_mode = self.get_saved().as_tuple()
        if rescan is None:
            content: Optional[Content] = normal_content_from_scan(
                file_path,
                file_type,
                parent_folders=parent_folders,
                name_parser=self.name_parser,
                trailer_names=trailer_names,
                manager=manager,
            )
            if content is None:
                if self.options["parse_error_is_exception"]:
                    msg = _("Parse Error: Couldn't parse content from '{file}'").format(
                        file=file_path,
                    )
                    raise RuntimeError(msg)

                return None

            content.scan(
                callback=self,
                handles=handles,
                parent_folders=parent_folders,
                trailer_names=trailer_names,
            )

            return content

        rescan.scan(
            callback=self,
            handles=handles,
            parent_folders=parent_folders,
            rescan=True,
            trailer_names=trailer_names,
        )

        return None


@decorate_class(slots=True)
class NumeratedContentCallback(ContentCallback):
    @override
    def process(
        self: Self,
        file_path: Path,
        file_type: ScannedFileType,
        handles: HandlesType,
        parent_folders: list[str],
        *,
        trailer_names: list[str],
        rescan: Optional[Content] = None,
    ) -> Optional[Content]:
        manager, _scanner, _language_picker, _error_mode = self.get_saved().as_tuple()
        if rescan is None:
            content: Optional[Content] = numerated_content_from_scan(
                file_path,
                file_type,
                parent_folders=parent_folders,
                name_parser=self.name_parser,
                trailer_names=trailer_names,
                manager=manager,
            )
            if content is None:
                if self.options["parse_error_is_exception"]:
                    msg = _("Parse Error: Couldn't parse content from '{file}'").format(
                        file=file_path,
                    )
                    raise RuntimeError(msg)

                return None

            content.scan(
                callback=self,
                handles=handles,
                parent_folders=parent_folders,
                trailer_names=trailer_names,
            )

            return content

        rescan.scan(
            callback=self,
            handles=handles,
            parent_folders=parent_folders,
            rescan=True,
            trailer_names=trailer_names,
        )

        return None


@decorate_class(slots=True)
class SymlinkedContentCallback(ContentCallback):
    pass


def parse_contents(
    root_folder: Path,
    options: ContentOptions,
    save_file: ParsedTargetFile,
    name_parser: NameParser,
    scanner: Scanner,
    language_picker: LanguagePicker,
    all_content_type: AnyType,
    config_type: ConfigType,
    manager: ManagerInterface,
    error_mode: ErrorMode,
    *,
    check: bool,
) -> list[Content]:

    callback: ContentCallback

    match config_type:
        case ConfigType.normal:
            callback = NormalContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                manager=manager,
                error_mode=error_mode,
            )
        case ConfigType.numerated:
            callback = NumeratedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                manager=manager,
                error_mode=error_mode,
            )
            # TODO
            return []
        case ConfigType.symlinked:
            callback = SymlinkedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                manager=manager,
                error_mode=error_mode,
            )
            # TODO
            return []
        case _:
            assert_never(config_type)

    if not save_file.file.exists():
        contents: list[Content] = process_folder(
            root_folder,
            callback=callback,
            handles=[],
            parent_folders=[],
            trailer_names=options["trailer_names"],
            parent_type=DefaultStatusBarInfo(),
        )

        save_to_file(
            tgt=save_file,
            contents=contents,
            serialize_type=all_content_type,
        )

        return contents

    contents = load_from_file(
        tgt=save_file,
        serialize_type=all_content_type,
    )

    if not check:
        return contents

    new_contents: list[Content] = process_folder(
        root_folder,
        callback=callback,
        handles=[],
        rescan=contents,
        parent_folders=[],
        trailer_names=options["trailer_names"],
        parent_type=DefaultStatusBarInfo(),
    )

    save_to_file(
        tgt=save_file,
        contents=new_contents,
        serialize_type=all_content_type,
    )

    return new_contents
