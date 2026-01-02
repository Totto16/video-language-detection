import json
from pathlib import Path
from typing import Any, Optional, Self, TypedDict, assert_never, override

from apischema import deserialize, serialize
from enlighten import Justify, Manager, get_manager

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


class ContentCallback(Callback[Content, ContentCharacteristic, CallbackTuple]):
    __options: ContentOptions
    __name_parser: NameParser
    __scanner: Scanner
    __progress_bars: dict[str, Any]
    __manager: Manager
    __status_bar: Any
    __language_picker: LanguagePicker

    def __init__(
        self: Self,
        options: ContentOptions,
        name_parser: NameParser,
        scanner: Scanner,
        language_picker: LanguagePicker,
        general_info: list[str],
    ) -> None:
        super().__init__()

        self.__options = options
        self.__name_parser = name_parser
        self.__scanner = scanner
        self.__progress_bars = {}
        manager = get_manager()
        if not isinstance(manager, Manager):
            msg = _("UNREACHABLE (not runnable in notebooks)")
            raise TypeError(msg)

        self.__manager = manager

        info_str: str = ""
        info_kw: dict[str, str] = {}

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
            justify=Justify.CENTER,
            stage=_("Scanning"),
            autorefresh=True,
            min_delta=0.5,
            **info_kw,
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
        self.__status_bar.update(stage=_("finished"))
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
            )
        case ConfigType.numerated:
            callback = NumeratedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                general_info=general_info,
            )
        case ConfigType.symlinked:
            callback = SymlinkedContentCallback(
                options=options,
                name_parser=name_parser,
                scanner=scanner,
                language_picker=language_picker,
                general_info=general_info,
            )
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
