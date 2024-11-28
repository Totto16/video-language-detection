import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional, Self, TypedDict, override

from apischema import deserialize, schema, serialize
from enlighten import Justify, Manager, get_manager

from config import Config
from content.base_class import (
    CallbackTuple,
    Content,
    ContentCharacteristic,
    Scanner,
    process_folder,
)
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import (
    Callback,
    ContentType,
    EmitType,
    NameParser,
    ScannedFileType,
    get_schema,
)
from content.metadata.metadata import HandlesType
from content.scan_helpers import content_from_scan
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from helper.constants import APP_NAME
from helper.translation import get_translator

if TYPE_CHECKING:
    from collections.abc import Mapping


_ = get_translator()


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    parse_error_is_exception: bool


class ContentCallback(Callback[Content, ContentCharacteristic, CallbackTuple]):
    __options: ContentOptions
    __name_parser: NameParser
    __scanner: Scanner
    __progress_bars: dict[str, Any]
    __manager: Manager
    __status_bar: Any

    def __init__(
        self: Self,
        options: ContentOptions,
        name_parser: NameParser,
        scanner: Scanner,
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
        self.__status_bar = self.__manager.status_bar(
            status_format=APP_NAME + "{fill}" + _("Stage") + ": {stage}{fill}{elapsed}",
            color="bold_underline_bright_white_on_blue",
            justify=Justify.CENTER,
            stage=_("Scanning"),
            autorefresh=True,
            min_delta=0.5,
        )

    @override
    def get_saved(self: Self) -> CallbackTuple:
        return (self.__manager, self.__scanner)

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
    def process(
        self: Self,
        file_path: Path,
        file_type: ScannedFileType,
        handles: HandlesType,
        parent_folders: list[str],
        *,
        rescan: Optional[Content] = None,
    ) -> Optional[Content]:
        if rescan is None:
            content: Optional[Content] = content_from_scan(
                file_path,
                file_type,
                parent_folders,
                name_parser=self.__name_parser,
            )
            if content is None:
                if self.__options["parse_error_is_exception"]:
                    msg = _("Parse Error: Couldn't parse content from '{file}'").format(
                        file=file_path,
                    )
                    raise RuntimeError(msg)

                return None

            content.scan(
                callback=self,
                handles=handles,
                parent_folders=parent_folders,
            )

            return content

        rescan.scan(
            callback=self,
            handles=handles,
            parent_folders=parent_folders,
            rescan=True,
        )

        return None

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


# from: https://wyfo.github.io/apischema/0.18/json_schema/
# schema extra can be callable to modify the schema in place
def to_one_of(schema: dict[str, Any]) -> None:
    if "anyOf" in schema:
        schema["oneOf"] = schema.pop("anyOf")


OneOf = schema(extra=to_one_of)


AllContent = Annotated[
    EpisodeContent | SeasonContent | SeriesContent | CollectionContent,
    OneOf,
]


def load_from_file(file_path: Path) -> list[Content]:
    with file_path.open(mode="r") as file:
        suffix: str = file_path.suffix[1:]
        match suffix:
            case "json":
                parsed_dict: dict[str, Any] = json.load(file)
                json_loaded: list[Content] = deserialize(
                    list[AllContent],
                    parsed_dict,
                )
                return json_loaded
            case _:
                msg = _("Data not loadable from '{suffix}' file!").format(suffix=suffix)
                raise RuntimeError(msg)


def save_to_file(file_path: Path, contents: list[Content]) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    with file_path.open(
        mode="w",
    ) as file:
        suffix: str = file_path.suffix[1:]
        match suffix:
            case "json":
                encoded_dict: dict[str, Any] = serialize(
                    list[AllContent],
                    contents,
                )
                json.dump(encoded_dict, file, indent=4, ensure_ascii=False)
            case _:
                msg = _("Data not saveable to '{suffix}' file!").format(suffix=suffix)
                raise RuntimeError(msg)


def parse_contents(
    root_folder: Path,
    options: ContentOptions,
    save_file: Path,
    name_parser: NameParser,
    scanner: Scanner,
) -> list[Content]:
    callback = ContentCallback(options, name_parser, scanner)

    if not save_file.exists():
        contents: list[Content] = process_folder(
            root_folder,
            callback=callback,
            handles=[],
            parent_folders=[],
        )

        save_to_file(save_file, contents)

        return contents

    contents = load_from_file(save_file)
    new_contents: list[Content] = process_folder(
        root_folder,
        callback=callback,
        handles=[],
        rescan=contents,
        parent_folders=[],
    )

    save_to_file(save_file, new_contents)

    return new_contents


def generate_schema(
    file_path: Path,
    any_type: Any,
    *,
    emit_type: Optional[EmitType] = None,
) -> None:
    result: Mapping[str, Any] = get_schema(
        any_type,
        additional_properties=False,
        all_refs=True,
        emit_type=emit_type,
    )

    if not file_path.parent.exists():
        Path(file_path).parent.mkdir(parents=True)

    with file_path.open(mode="w") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def generate_schemas(folder: Path) -> None:
    generate_schema(folder / "content_list_schema.json", list[AllContent])
    generate_schema(folder / "config_schema.json", Config, emit_type="deserialize")
