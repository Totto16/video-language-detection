#!/usr/bin/env python3

from json import JSONEncoder
from os import listdir
from pathlib import Path
from typing import (
    Any,
    Optional,
    Self,
    TypedDict,
)

from classifier import Classifier
from enlighten import Manager

from content.general import (
    Callback,
    ContentType,
    MissingOverrideError,
    NameParser,
    ScannedFile,
    ScannedFileType,
    Summary,
)

ContentCharacteristic = tuple[Optional[ContentType], ScannedFileType]


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile


class Content:
    __type: ContentType
    __scanned_file: ScannedFile

    def __init__(self: Self, _type: ContentType, scanned_file: ScannedFile) -> None:
        self.__type = _type
        self.__scanned_file = scanned_file

    def summary(self: Self, _detailed: bool = False) -> Summary:
        raise MissingOverrideError

    @property
    def type(self: Self) -> ContentType:  # noqa: A003
        return self.__type

    @property
    def description(self: Self) -> Any:
        raise MissingOverrideError

    @property
    def scanned_file(self: Self) -> ScannedFile:
        return self.__scanned_file

    def generate_checksum(self: Self, manager: Manager) -> None:
        self.__scanned_file.generate_checksum(manager)

    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = {
            "scanned_file": encode(self.__scanned_file),
            "type": encode(self.__type),
        }
        return as_dict

    def scan(
        self: Self,
        callback: Callback["Content", ContentCharacteristic, Manager],  # noqa: ARG002
        name_parser: NameParser,  # noqa: ARG002
        *,
        parent_folders: list[str],  # noqa: ARG002
        classifier: Classifier,  # noqa: ARG002
        rescan: bool = False,  # noqa: ARG002
    ) -> None:
        raise MissingOverrideError

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


def process_folder(
    directory: Path,
    callback: Callback[Content, ContentCharacteristic, Manager],
    name_parser: NameParser,
    *,
    parent_folders: list[str],
    parent_type: Optional[ContentType] = None,
    rescan: Optional[list[Content]] = None,
) -> list[Content]:
    temp: list[tuple[Path, ScannedFileType, list[str]]] = []
    ignored: int = 0
    sorted_files: list[str] = sorted(listdir(directory))
    for file in sorted_files:
        file_path: Path = Path(directory) / file

        file_type: ScannedFileType = ScannedFileType.from_path(file_path)
        should_ignore: bool = callback.ignore(file_path, file_type, parent_folders)
        if should_ignore:
            ignored += 1
            continue

        temp.append((file_path, file_type, parent_folders))

    value: ContentCharacteristic = (parent_type, ScannedFileType.folder)

    #  total, processing, ignored
    amount: tuple[int, int, int] = (len(temp) + ignored, len(temp), ignored)

    callback.start(amount, directory.name, parent_folders, value)

    if rescan is None:
        results: list[Content] = []
        for file_path, file_type, parent_folders in temp:
            result: Optional[Content] = callback.process(
                file_path,
                file_type,
                parent_folders,
                name_parser=name_parser,
            )
            value = (
                result.type if result is not None else None,
                ScannedFileType.from_path(file_path),
            )
            callback.progress(directory.name, parent_folders, value)
            if result is not None:
                results.append(result)

        value = (parent_type, ScannedFileType.folder)
        callback.finish(directory.name, parent_folders, value)

        return results

    already_scanned_file_paths: list[Path] = [
        content.scanned_file.path for content in rescan
    ]

    for file_path, file_type, parent_folders in temp:
        is_rescan = (
            None
            if file_path not in already_scanned_file_paths
            else (rescan[already_scanned_file_paths.index(file_path)])
        )

        result = callback.process(
            file_path,
            file_type,
            parent_folders,
            name_parser=name_parser,
            rescan=is_rescan,
        )

        value = (
            result.type if result is not None else None,
            ScannedFileType.from_path(file_path),
        )

        callback.progress(directory.name, parent_folders, value)
        if result is not None and is_rescan is None:
            rescan.append(result)

    value = (parent_type, ScannedFileType.folder)
    callback.finish(directory.name, parent_folders, value)

    return rescan