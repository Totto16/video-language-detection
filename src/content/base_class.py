from dataclasses import dataclass, field
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Self, TypedDict, override

from apischema import alias
from enlighten import Manager

from classifier import Classifier, FileMetadataError, PredictionFailReason, WAVFile
from content.general import (
    Callback,
    ContentType,
    MissingOverrideError,
    ScannedFile,
    ScannedFileType,
    safe_index,
)
from content.language import Language
from content.language_picker import LanguagePicker
from content.metadata.metadata import (
    HandlesType,
    InternalMetadataType,
    MetadataHandle,
    SkipHandle,
)
from content.metadata.scanner import MetadataScanner
from content.prediction import PredictionBest
from content.shared import ScanType
from content.summary import Summary
from helper.log import get_logger

logger: Logger = get_logger()

type ContentCharacteristic = tuple[Optional[ContentType], ScannedFileType]


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile


class SummaryResult:
    __file: ScannedFile
    __value: bool

    def __init__(self: Self, file: ScannedFile, *, value: bool) -> None:
        self.__file = file
        self.__value = value

    def get_reason_as_str(self: Self) -> str:
        raise MissingOverrideError

    @property
    def file(self: Self) -> ScannedFile:
        return self.__file

    @property
    def value(self: Self) -> bool:
        return self.__value


type ScanSummary = dict[bool, int]


class SuccessSummaryManager:
    __results: list[SummaryResult]

    def __init__(self: Self) -> None:
        self.__results = []

    def add(self: Self, result: SummaryResult) -> None:
        self.__results.append(result)

    def get_summary(self: Self) -> ScanSummary:
        summary: ScanSummary = {True: 0, False: 0}
        for result in self.__results:
            summary[result.value] = summary[result.value] + 1

        return summary


class FailReason(Enum):
    exception = " exception"
    scan_failure = "scan_failure"


class FailedFor(SummaryResult):
    __reason: FailReason

    def __init__(self: Self, reason: FailReason, file: ScannedFile) -> None:
        super().__init__(file, value=False)
        self.__reason = reason

    @override
    def get_reason_as_str(self: Self) -> str:
        return f"Scan failed with reason: {self.__reason!s}"

    @property
    def reason(self: Self) -> FailReason:
        return self.__reason


class FailedForWithLanguage(FailedFor):
    __best: Optional[PredictionBest]
    __reason: PredictionFailReason

    def __init__(
        self: Self,
        best: Optional[PredictionBest],
        reason: PredictionFailReason,
        file: ScannedFile,
    ) -> None:
        super().__init__(FailReason.scan_failure, file)
        self.__best = best
        self.__reason = reason

    @override
    def get_reason_as_str(self: Self) -> str:
        if self.__best is None:
            return f"Scan failed with language reason: {self.__reason!s}"

        return f"Scan failed with language reason: {self.__reason!s} and the best language was {self.__best.language} with {self.__best.accuracy:.2%}"

    @property
    def best(self: Self) -> Optional[PredictionBest]:
        return self.__best


class SuccessFor(SummaryResult):
    __success_rate: float

    def __init__(self: Self, success_rate: float, file: ScannedFile) -> None:
        super().__init__(file, value=True)
        self.__success_rate = success_rate

    @override
    def get_reason_as_str(self: Self) -> str:
        return "Scan was successful"

    @property
    def success_rat(self: Self) -> float:
        return self.__success_rate


class LanguageScanner:
    __classifier: Classifier
    __summary_manager: SuccessSummaryManager

    def __init__(
        self: Self,
        classifier: Classifier,
    ) -> None:
        self.__classifier = classifier
        self.__summary_manager = SuccessSummaryManager()

    def get_language(
        self: Self,
        scanned_file: ScannedFile,
        language_picker: LanguagePicker,
        *,
        manager: Optional[Manager] = None,
    ) -> Language:
        try:
            wav_file = WAVFile(scanned_file.path)

            prediction_result = self.__classifier.predict(
                wav_file,
                scanned_file.path,
                language_picker,
                manager,
            )

            if isinstance(prediction_result, PredictionBest):
                self.__summary_manager.add(
                    SuccessFor(prediction_result.accuracy, scanned_file),
                )
                return prediction_result.language

            self.__summary_manager.add(
                FailedForWithLanguage(
                    prediction_result.best,
                    prediction_result.reason,
                    scanned_file,
                ),
            )
            return Language.unknown()
        except FileMetadataError:
            logger.exception("Get Language")
            self.__summary_manager.add(FailedFor(FailReason.exception, scanned_file))
            return Language.unknown()

    @property
    def summary_manager(self: Self) -> SuccessSummaryManager:
        return self.__summary_manager


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
        metadata: InternalMetadataType,  # noqa: ARG002
    ) -> bool:
        raise MissingOverrideError

    @property
    def language_scanner(self: Self) -> LanguageScanner:
        return self.__language_scanner

    @property
    def metadata_scanner(self: Self) -> MetadataScanner:
        return self.__metadata_scanner


type CallbackTuple = tuple[Manager, Scanner, LanguagePicker]


@dataclass(slots=True, repr=True)
class Content:
    __type: ContentType = field(metadata=alias("type"))
    __scanned_file: ScannedFile = field(metadata=alias("scanned_file"))
    _metadata: InternalMetadataType = field(
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
    def metadata(self: Self) -> InternalMetadataType:
        return self._metadata

    def _get_new_handles(self: Self, old_handles: HandlesType) -> HandlesType:
        if self.metadata is None or old_handles is None:
            return None

        if isinstance(self.metadata, SkipHandle) or isinstance(old_handles, SkipHandle):
            return SkipHandle()

        # Note: this is important, so that it's a copy
        new_handles: list[MetadataHandle] = list(old_handles)
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
    sorted_files: list[Path] = sorted(Path.iterdir(directory))
    for file in sorted_files:
        file_path: Path = directory / file

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
