from dataclasses import dataclass, field
from logging import Logger
from os import listdir
from pathlib import Path
from typing import Any, Optional, Self, TypedDict

from apischema import alias
from enlighten import Manager

from classifier import Classifier, FileMetadataError, Language, WAVFile
from content.general import (
    Callback,
    ContentType,
    MissingOverrideError,
    ScannedFile,
    ScannedFileType,
    Summary,
    safe_index,
)
from content.metadata.metadata import HandlesType, MetadataHandle
from content.metadata.scanner import MetadataScanner
from content.shared import ScanType
from helper.log import get_logger

logger: Logger = get_logger()

ContentCharacteristic = tuple[Optional[ContentType], ScannedFileType]


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile


class LanguageScanner:
    __classifier: Classifier

    def __init__(
        self: Self,
        classifier: Classifier,
    ) -> None:
        self.__classifier = classifier

    def get_language(
        self: Self,
        scanned_file: ScannedFile,
        *,
        manager: Optional[Manager] = None,
    ) -> Language:
        try:
            wav_file = WAVFile(scanned_file.path)

            best, _ = self.__classifier.predict(
                wav_file,
                scanned_file.path,
                manager,
            )
        except FileMetadataError:
            logger.exception("Get Language")
            return Language.unknown()
        else:
            # python is funky xD, leaking variables as desired pattern xD
            return best.language


class Scanner:
    __language_scanner: LanguageScanner
    __metadata_scanner: MetadataScanner

    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
    ) -> None:
        self.__language_scanner = language_scanner
        self.__metadata_scanner = metadata_scanner

    def should_scan_language(
        self: Self,
        scan_type: ScanType,  # noqa: ARG002
    ) -> bool:
        raise MissingOverrideError

    def should_scan_metadata(
        self: Self,
        scan_type: ScanType,  # noqa: ARG002
        metadata: Optional[MetadataHandle],  # noqa: ARG002
    ) -> bool:
        raise MissingOverrideError

    @property
    def language_scanner(self: Self) -> LanguageScanner:
        return self.__language_scanner

    @property
    def metadata_scanner(self: Self) -> MetadataScanner:
        return self.__metadata_scanner


CallbackTuple = tuple[Manager, Scanner]


@dataclass(slots=True, repr=True)
class Content:
    __type: ContentType = field(metadata=alias("type"))
    __scanned_file: ScannedFile = field(metadata=alias("scanned_file"))
    _metadata: Optional[MetadataHandle] = field(
        metadata=alias("metadata"),
    )

    def summary(self: Self, *, detailed: bool = False) -> Summary:  # noqa: ARG002
        raise MissingOverrideError

    @property
    def type(self: Self) -> ContentType:
        return self.__type

    @property
    def description(self: Self) -> Any:
        raise MissingOverrideError

    @property
    def scanned_file(self: Self) -> ScannedFile:
        return self.__scanned_file

    @property
    def metadata(self: Self) -> Optional[MetadataHandle]:
        return self._metadata

    def _get_new_handles(self: Self, old_handles: HandlesType) -> HandlesType:
        new_handles: HandlesType = None
        if self.metadata is not None and old_handles is not None:
            # Note: this is important, so that it's a copy
            new_handles = list(old_handles)
            new_handles.append(self.metadata)

        return new_handles

    def generate_checksum(self: Self, manager: Manager) -> None:
        self.__scanned_file.generate_checksum(manager)

    def scan(
        self: Self,
        callback: Callback[  # noqa: ARG002
            "Content",
            ContentCharacteristic,
            CallbackTuple,
        ],
        *,
        handles: HandlesType,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        rescan: bool = False,  # noqa: ARG002
    ) -> None:
        raise MissingOverrideError


def process_folder(
    directory: Path,
    callback: Callback[Content, ContentCharacteristic, CallbackTuple],
    *,
    handles: HandlesType,
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
        for file_path, file_type, parent_folders_temp in temp:
            result: Optional[Content] = callback.process(
                file_path,
                file_type,
                handles,
                parent_folders_temp,
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

    for file_path, file_type, parent_folders_temp in temp:
        is_rescan = (
            None
            if file_path not in already_scanned_file_paths
            else (rescan[already_scanned_file_paths.index(file_path)])
        )

        result = callback.process(
            file_path,
            file_type,
            handles,
            parent_folders_temp,
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
            msg = f"Path to delete wasn't founds: {path}"
            raise RuntimeError(msg)

        del rescan[index]
        deleted += 1

    value = (parent_type, ScannedFileType.folder)
    callback.finish(directory.name, parent_folders, deleted, value)

    return rescan
