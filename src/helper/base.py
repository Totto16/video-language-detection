import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    assert_never,
    cast,
    override,
)

import enlighten
from apischema import deserialize, serialize

from config import ConfigType
from content.base_class import (
    CallbackTuple,
    Content,
    ContentCharacteristic,
    Scanner,
    process_folder,
)
from content.general import (
    Callback,
    ContentType,
    NameParser,
    ScannedFileType,
)
from content.language_picker import LanguagePicker
from content.metadata.metadata import HandlesType
from content.scan_helpers import normal_content_from_scan, numerated_content_from_scan
from helper.constants import APP_NAME
from helper.translation import get_translator

_ = get_translator()

type AnyType = Any


def save_to_file(
    file_path: Path,
    contents: list[Content],
    serialize_type: AnyType,
) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    with file_path.open(
        mode="w",
    ) as file:
        suffix: str = file_path.suffix[1:]
        match suffix:
            case "json":
                encoded_dict: dict[str, Any] = serialize(
                    list[serialize_type],
                    contents,
                )
                json.dump(encoded_dict, file, indent=4, ensure_ascii=False)
            case _:
                msg = _("Data not saveable to '{suffix}' file!").format(suffix=suffix)
                raise RuntimeError(msg)


def load_from_file(
    file_path: Path,
    serialize_type: AnyType,
) -> list[Content]:
    with file_path.open(mode="r") as file:
        suffix: str = file_path.suffix[1:]
        match suffix:
            case "json":
                parsed_dict: dict[str, Any] = json.load(file)
                json_loaded: list[Content] = deserialize(
                    list[serialize_type],
                    parsed_dict,
                )
                return json_loaded
            case _:
                msg = _("Data not loadable from '{suffix}' file!").format(suffix=suffix)
                raise RuntimeError(msg)


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    trailer_names: list[str]
    parse_error_is_exception: bool


ManagerJustify = enlighten.Justify


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar
class StatusBarOptions(TypedDict, total=False):
    color: str
    justify: enlighten.Justify
    min_delta: float  # = 0.1
    status_format: str


type AdditionalArgs = dict[str, Any]


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.status_bar
class StatusBarGetOptions(StatusBarOptions, total=False):
    autorefresh: bool
    additional_args: AdditionalArgs


class SupportsFloat(Protocol):
    def __float__(self) -> float: ...


# actual type int, but implementation and python allows classes, which support int() or float()
type NumberLike = int | float | SupportsFloat


def number_like_convert_to_serializable(number_like: NumberLike) -> float:
    return float(number_like)


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter
class CounterOptions(TypedDict, total=False):
    bar_format: str
    count: NumberLike  # = 0,
    color: str
    desc: str
    leave: bool  # = True
    total: NumberLike
    unit: str


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
class StatusBarInterfaceUpdateOptions(TypedDict, total=False):
    force: bool
    additional_args: AdditionalArgs


class StatusBarInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @abstractmethod
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None: ...


# NOTE: only StatusBarInterface supports AdditionalArgs atm!


class CounterInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter.update
    @abstractmethod
    def update(self: Self, incr: NumberLike = 1, force: bool = False) -> None: ...

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter.close
    @abstractmethod
    def close(self: Self, clear: bool = False) -> None: ...


class ManagerInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.status_bar
    @abstractmethod
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface: ...

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.counter
    @abstractmethod
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface: ...

    @abstractmethod
    def stop(
        self: Self,
    ) -> None: ...


class TuiStatusBar(StatusBarInterface):
    __impl: enlighten.StatusBar

    def __init__(self: Self, impl: enlighten.StatusBar) -> None:
        super().__init__()
        self.__impl = impl

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        modified_fields: StatusBarInterfaceUpdateOptions = {**fields}

        if modified_fields.get("additional_args") is not None:
            additional_args: AdditionalArgs = modified_fields["additional_args"]
            del modified_fields["additional_args"]
            for key, value in additional_args.items():
                if modified_fields.get(key) is not None:
                    msg = f"Trying to overwrite normal option key '{key}' in enlighten Manager implementation"
                    raise RuntimeError(msg)

                cast(AdditionalArgs, modified_fields)[key] = value

        return self.__impl.update(fields=modified_fields)


class TuiCounter(CounterInterface):
    __impl: enlighten.Counter

    def __init__(self: Self, impl: enlighten.Counter) -> None:
        super().__init__()
        self.__impl = impl

    @override
    def update(self: Self, incr: NumberLike = 1, force: bool = False) -> None:
        return self.__impl.update(incr=incr, force=force)

    @override
    def close(self: Self, clear: bool = False) -> None:
        return self.__impl.close(clear=clear)


class TuiManager(ManagerInterface):
    __impl: enlighten.Manager

    def __init__(self: Self) -> None:
        super().__init__()
        manager = enlighten.get_manager()
        if not isinstance(manager, enlighten.Manager):
            msg = _("UNREACHABLE (not runnable in notebooks)")
            raise TypeError(msg)

        self.__impl = manager

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        modified_kwargs: StatusBarGetOptions = {**kwargs}

        if modified_kwargs.get("additional_args") is not None:
            additional_args: AdditionalArgs = modified_kwargs["additional_args"]
            del modified_kwargs["additional_args"]
            for key, value in additional_args.items():
                if modified_kwargs.get(key) is not None:
                    msg = f"Trying to overwrite normal option key '{key}' in enlighten Manager implementation"
                    raise RuntimeError(msg)

                cast(AdditionalArgs, modified_kwargs)[key] = value

        status_bar = self.__impl.status_bar(
            kwargs=modified_kwargs,
        )
        return TuiStatusBar(impl=status_bar)

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.counter
    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        counter = self.__impl.counter(
            position=None,
            kwargs=kwargs,
        )
        return TuiCounter(impl=counter)

    def stop(
        self: Self,
    ) -> None:
        return self.__impl.stop()


class ContentCallback(Callback[Content, ContentCharacteristic, CallbackTuple]):
    __options: ContentOptions
    __name_parser: NameParser
    __scanner: Scanner
    __progress_bars: dict[str, CounterInterface]
    __manager: ManagerInterface
    __status_bar: StatusBarInterface
    __language_picker: LanguagePicker

    def __init__(
        self: Self,
        options: ContentOptions,
        name_parser: NameParser,
        scanner: Scanner,
        language_picker: LanguagePicker,
        general_info: list[str],
        manager: ManagerInterface,
    ) -> None:
        super().__init__()

        self.__options = options
        self.__name_parser = name_parser
        self.__scanner = scanner
        self.__progress_bars = {}
        self.__manager = manager

        info_str: str = ""
        info_kw: dict[str, str] = {}

        info_kw["stage"] = _("Scanning")

        if len(general_info) > 0:
            info_parts: list[tuple[str, str]] = [
                (f"info_{i}", general_info[i]) for i in range(len(general_info))
            ]
            info_kw = dict(info_parts)
            info_str = "{fill}".join(f"{{{x[0]}}}" for x in info_parts) + "{fill}"

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
            additional_args=info_kw,
        )
        self.__language_picker = language_picker

    @override
    def get_saved(self: Self) -> CallbackTuple:
        return (self.__manager, self.__scanner, self.__language_picker)

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
        amount: tuple[int, int, int],
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        content_type, _i = characteristic

        value: tuple[str, str]

        match content_type:
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

        _1, processing, _2 = amount

        self.__progress_bars[name] = self.__manager.counter(
            total=processing,
            desc=name,
            unit=unit,
            leave=False,
            color=color,
        )
        self.__progress_bars[name].update(0, force=True)

    @override
    def progress(
        self: Self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
        *,
        amount: int = 1,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            msg = _("No Progressbar, on progress callback")
            raise RuntimeError(msg)

        self.__progress_bars[name].update(amount)

    @override
    def finish(
        self: Self,
        name: str,
        parent_folders: list[str],
        deleted: int,
        characteristic: ContentCharacteristic,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            msg = _("No Progressbar, on progress finish")
            raise RuntimeError(msg)

        self.__progress_bars[name].close(clear=True)
        del self.__progress_bars[name]

    def __del__(self: Self) -> None:
        self.__status_bar.update(additional_args={"stage": _("finished")})
        self.__manager.stop()

    @property
    def name_parser(self: Self) -> NameParser:
        return self.__name_parser

    @property
    def options(self: Self) -> ContentOptions:
        return self.__options


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
        if rescan is None:
            content: Optional[Content] = normal_content_from_scan(
                file_path,
                file_type,
                parent_folders=parent_folders,
                name_parser=self.name_parser,
                trailer_names=trailer_names,
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
        if rescan is None:
            content: Optional[Content] = numerated_content_from_scan(
                file_path,
                file_type,
                parent_folders=parent_folders,
                name_parser=self.name_parser,
                trailer_names=trailer_names,
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


class SymlinkedContentCallback(ContentCallback):
    pass


def parse_contents(
    root_folder: Path,
    options: ContentOptions,
    save_file: Path,
    name_parser: NameParser,
    scanner: Scanner,
    language_picker: LanguagePicker,
    all_content_type: AnyType,
    general_info: list[str],
    config_type: ConfigType,
    manager: ManagerInterface,
) -> list[Content]:

    callback: ContentCallback

    match config_type:
        case ConfigType.normal:
            callback = NormalContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                general_info=general_info,
                manager=manager,
            )
        case ConfigType.numerated:
            callback = NumeratedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                general_info=general_info,
                manager=manager,
            )
            # TODO
            return []
        case ConfigType.symlinked:
            callback = SymlinkedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                general_info=general_info,
                manager=manager,
            )
            # TODO
            return []
        case _:
            assert_never(config_type)

    if not save_file.exists():
        contents: list[Content] = process_folder(
            root_folder,
            callback=callback,
            handles=[],
            parent_folders=[],
            trailer_names=options["trailer_names"],
        )

        save_to_file(
            file_path=save_file,
            contents=contents,
            serialize_type=all_content_type,
        )

        return contents

    contents = load_from_file(
        file_path=save_file,
        serialize_type=all_content_type,
    )
    new_contents: list[Content] = process_folder(
        root_folder,
        callback=callback,
        handles=[],
        rescan=contents,
        parent_folders=[],
        trailer_names=options["trailer_names"],
    )

    save_to_file(
        file_path=save_file,
        contents=new_contents,
        serialize_type=all_content_type,
    )

    return new_contents
