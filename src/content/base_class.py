#!/usr/bin/env python3

from dataclasses import dataclass, field
from os import listdir
from pathlib import Path
from typing import Any, Optional, Self, TypedDict

from apischema import alias
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
    safe_index,
)

ContentCharacteristic = tuple[Optional[ContentType], ScannedFileType]


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile




@dataclass(slots=True, repr=True)
class Content:
    __type: ContentType = field(metadata=alias("type"))
    __scanned_file: ScannedFile = field(metadata=alias("scanned_file"))


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
        callback.finish(directory.name, parent_folders, 0, value)

        return results

    already_scanned_file_paths: list[Path] = [
        content.scanned_file.path for content in rescan
    ]
    scanned_file_registry: list[bool] = [False for _ in rescan]

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

        if is_rescan is not None:
            idx = already_scanned_file_paths.index(file_path)
            scanned_file_registry[idx] = True

    deleted: int = 0
    for path, was_found in zip(
        already_scanned_file_paths,
        scanned_file_registry,
        strict=True,
    ):
        if was_found:
            continue

        index: Optional[int] = safe_index(
            [content.scanned_file.path for content in rescan],
            path,
        )
        if index is None:
            raise RuntimeError(f"Path to delete wasn't founds: {path}")

        del rescan[index]
        deleted += 1

    value = (parent_type, ScannedFileType.folder)
    callback.finish(directory.name, parent_folders, deleted, value)

    return rescan
