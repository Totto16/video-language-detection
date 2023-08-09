#!/usr/bin/env python3


import json
from os import makedirs
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional, Self, TypedDict

from apischema import deserialize, schema, serialize
from apischema.json_schema import (
    deserialization_schema,
    serialization_schema,
)
from classifier import Classifier
from content.base_class import Content, ContentCharacteristic, process_folder
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import Callback, ContentType, NameParser, ScannedFileType
from content.scan_helpers import content_from_scan
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from enlighten import Justify, Manager, get_manager
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Mapping


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    parse_error_is_exception: bool


class ContentCallback(Callback[Content, ContentCharacteristic, Manager]):
    __options: ContentOptions
    __classifier: Classifier
    __progress_bars: dict[str, Any]
    __manager: Manager
    __status_bar: Any

    def __init__(self: Self, options: ContentOptions, classifier: Classifier) -> None:
        super().__init__()

        self.__options = options
        self.__classifier = classifier
        self.__progress_bars = {}
        manager = get_manager()
        if not isinstance(manager, Manager):
            raise RuntimeError("UNREACHABLE (not runnable in notebooks)")

        self.__manager = manager
        self.__status_bar = self.__manager.status_bar(
            status_format="Video Language Detector{fill}Stage: {stage}{fill}{elapsed}",
            color="bold_underline_bright_white_on_blue",
            justify=Justify.CENTER,
            stage="Scanning",
            autorefresh=True,
            min_delta=0.5,
        )

    @override
    def get_saved(self: Self) -> Manager:
        return self.__manager

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
        parent_folders: list[str],
        name_parser: NameParser,
        *,
        rescan: Optional[Content] = None,
    ) -> Optional[Content]:
        if rescan is None:
            content: Optional[Content] = content_from_scan(
                file_path,
                file_type,
                parent_folders,
                name_parser=name_parser,
            )
            if content is None:
                if self.__options["parse_error_is_exception"]:
                    raise RuntimeError(
                        f"Parse Error: Couldn't parse content from '{file_path}'",
                    )

                return None

            content.scan(
                callback=self,
                parent_folders=parent_folders,
                classifier=self.__classifier,
                name_parser=name_parser,
            )

            return content

        rescan.scan(
            callback=self,
            parent_folders=parent_folders,
            classifier=self.__classifier,
            name_parser=name_parser,
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
        content_type, _ = characteristic

        value: tuple[str, str]

        match content_type:
            case ContentType.collection:
                value = ("blue", "series")
            case ContentType.series:
                value = ("cyan", "seasons")
            case ContentType.season:
                value = ("green", "episodes")
            case ContentType.episode:
                value = ("yellow", "tasks")
            case _:
                value = ("purple", "folders")

        color, unit = value

        _, processing, _ = amount

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
            raise RuntimeError("No Progressbar, on progress callback")

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
            raise RuntimeError("No Progressbar, on progress finish")

        self.__progress_bars[name].close(clear=True)
        del self.__progress_bars[name]

    def __del__(self: Self) -> None:
        self.__status_bar.update(stage="finished")
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
    with open(file_path) as file:
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
                raise RuntimeError(f"Not loadable from '{suffix}' file!")


def save_to_file(file_path: Path, contents: list[Content]) -> None:
    if not file_path.parent.exists():
        makedirs(file_path.parent)

    with open(
        file_path,
        "w",
    ) as file:
        suffix: str = file_path.suffix[1:]
        match suffix:
            case "json":
                encoded_dict: dict[str, Any] = serialize(
                    list[AllContent],
                    contents,
                )
                json_content: str = json.dumps(encoded_dict, indent=4)
                file.write(json_content)
            case _:
                raise RuntimeError(f"Not loadable from '{suffix}' file!")


def parse_contents(
    root_folder: Path,
    options: ContentOptions,
    save_file: Path,
    name_parser: NameParser,
) -> list[Content]:
    classifier = Classifier()
    callback = ContentCallback(options, classifier)

    if not save_file.exists():
        contents: list[Content] = process_folder(
            root_folder,
            callback=callback,
            name_parser=name_parser,
            parent_folders=[],
        )

        save_to_file(save_file, contents)

        return contents

    contents = load_from_file(save_file)
    new_contents: list[Content] = process_folder(
        root_folder,
        callback=callback,
        name_parser=name_parser,
        rescan=contents,
        parent_folders=[],
    )

    save_to_file(save_file, new_contents)

    return new_contents


def generate_json_schema(file_path: Path, any_type: Any) -> None:
    result: Mapping[str, Any] = deserialization_schema(
        any_type,
        additional_properties=False,
        all_refs=True,
    )

    result2 = serialization_schema(any_type, additional_properties=False, all_refs=True)

    if result != result2:
        raise RuntimeError("Deserialization and Serialization scheme mismatch")

    if not file_path.parent.exists():
        makedirs(file_path.parent)

    with open(file_path, "w") as file:
        json_content: str = json.dumps(result, indent=4)
        file.write(json_content)
