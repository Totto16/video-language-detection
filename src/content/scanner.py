from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Literal,
    Optional,
    Self,
    TypedDict,
    assert_never,
    override,
)

from content.base_class import LanguageScanner, Scanner
from content.metadata.metadata import InternalMetadataType
from content.metadata.scanner import MetadataScanner
from content.shared import ScanKind, ScanType
from helper.apischema import Deprecated, OneOf
from helper.decorator import decorate_class


@decorate_class(slots=True)
class StaticScanner(Scanner):
    __value: bool

    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
        *,
        value: bool,
    ) -> None:
        super().__init__(language_scanner, metadata_scanner)
        self.__value = value

    @override
    def should_scan_language(self: Self, scan_type: ScanType) -> bool:
        return self.__value

    @override
    def should_scan_metadata(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool:
        if not self.__value:
            return False

        return self.metadata_scanner.can_scan() and self.metadata_scanner.should_scan(
            scan_type,
            metadata,
        )


@decorate_class(slots=True)
class FullScanner(StaticScanner):
    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
    ) -> None:
        super().__init__(language_scanner, metadata_scanner, value=True)


@decorate_class(slots=True)
class NoScanner(StaticScanner):
    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
    ) -> None:
        super().__init__(language_scanner, metadata_scanner, value=False)


class ScannerTypes(Enum):
    only_metadata = "only_metadata"
    only_language = "only_language"
    both = "both"
    none = "none"


class AdvancedScannerPositionDict(TypedDict, total=False):
    language: int
    metadata: int


@dataclass(slots=True, repr=True)
class AdvancedScannerPositionData:
    language: int
    metadata: int

    def at(self: Self, key: Literal["metadata", "language"] | ScanKind) -> int:
        return (
            self.language if key in ("language", ScanKind.language) else self.metadata
        )

    def set(
        self: Self,
        key: Literal["metadata", "language"] | ScanKind,
        value: int,
    ) -> None:
        if key in ("language", ScanKind.language):
            self.language = value
        else:
            self.metadata = value


def to_advanced_scanner_postion_total(
    value: int | AdvancedScannerPositionDict | AdvancedScannerPositionData,
    defaults: AdvancedScannerPositionData,
) -> AdvancedScannerPositionData:
    if isinstance(value, int):
        return AdvancedScannerPositionData(
            language=value,
            metadata=value,
        )

    if isinstance(value, AdvancedScannerPositionData):
        return value

    language: int = value.get(
        "language",
        defaults.language,
    )

    metadata: int = value.get(
        "metadata",
        defaults.metadata,
    )

    return AdvancedScannerPositionData(
        language=language,
        metadata=metadata,
    )


SimplePosition = Annotated[int, Deprecated]

Position = Annotated[SimplePosition | AdvancedScannerPositionDict, OneOf]


class ConfigScannerDict(TypedDict, total=False):
    start_position: Position
    scan_amount: Position
    allow_abort: bool
    types: ScannerTypes


@dataclass(slots=True, repr=True)
class ConfigScannerData:
    start_position: AdvancedScannerPositionData
    scan_amount: AdvancedScannerPositionData
    allow_abort: bool
    types: ScannerTypes
    # TODO: print progress option


@decorate_class(slots=True)
class ConfigScanner(Scanner):
    __start_position: AdvancedScannerPositionData
    __scan_amount: AdvancedScannerPositionData
    __allow_abort: bool
    __types: ScannerTypes
    # state
    __current_position: AdvancedScannerPositionData
    __is_aborted: bool

    @property
    def __defaults(self: Self) -> ConfigScannerData:
        return ConfigScannerData(
            start_position=AdvancedScannerPositionData(language=0, metadata=0),
            scan_amount=AdvancedScannerPositionData(language=100, metadata=100),
            allow_abort=True,
            types=ScannerTypes.both,
        )

    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
        *,
        config: Optional[ConfigScannerDict] = None,
    ) -> None:
        super().__init__(language_scanner, metadata_scanner)

        loaded_dict: Optional[ConfigScannerDict] = config
        if loaded_dict is not None:
            start_position: (
                int | AdvancedScannerPositionData | AdvancedScannerPositionDict
            ) = loaded_dict.get(
                "start_position",
                self.__defaults.start_position,
            )

            self.__start_position = to_advanced_scanner_postion_total(
                start_position,
                self.__defaults.start_position,
            )

            scan_amount: (
                int | AdvancedScannerPositionData | AdvancedScannerPositionDict
            ) = loaded_dict.get(
                "scan_amount",
                self.__defaults.scan_amount,
            )

            self.__scan_amount = to_advanced_scanner_postion_total(
                scan_amount,
                self.__defaults.scan_amount,
            )

            self.__allow_abort = loaded_dict.get(
                "allow_abort",
                self.__defaults.allow_abort,
            )
            self.__types = loaded_dict.get(
                "types",
                self.__defaults.types,
            )
        else:
            self.__start_position = self.__defaults.start_position
            self.__scan_amount = self.__defaults.scan_amount
            self.__allow_abort = self.__defaults.allow_abort
            self.__types = self.__defaults.types

        self.__current_position = AdvancedScannerPositionData(language=0, metadata=0)
        self.__is_aborted = False

    def __is_type(self: Self, scan_kind: ScanKind) -> bool:
        if self.__types == ScannerTypes.none:
            return False

        if self.__types == ScannerTypes.both:
            return True

        if self.__types == ScannerTypes.only_language:
            return scan_kind == ScanKind.language

        return scan_kind == ScanKind.metadata

    def __should_scan_kind(self: Self, scan_kind: ScanKind) -> bool:
        if not self.__is_type(scan_kind):
            return False

        if self.__is_aborted and self.__allow_abort:
            return False

        result = False

        if (
            self.__start_position.at(scan_kind) <= self.__current_position.at(scan_kind)
        ) and (
            self.__current_position.at(scan_kind) - self.__start_position.at(scan_kind)
            < self.__scan_amount.at(scan_kind)
        ):
            result = True

        self.__current_position.set(
            scan_kind, self.__current_position.at(scan_kind) + 1,
        )

        return result

    @override
    def should_scan_language(
        self: Self,
        scan_type: ScanType,
    ) -> bool:
        return self.__should_scan_kind(ScanKind.language)

    @override
    def should_scan_metadata(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool:
        should_scan_kind = self.__should_scan_kind(ScanKind.metadata)

        if not should_scan_kind:
            return False

        return self.metadata_scanner.can_scan() and self.metadata_scanner.should_scan(
            scan_type,
            metadata,
        )

    @property
    def allow_abort(self: Self) -> bool:
        return self.__allow_abort

    def abort(self: Self) -> bool:
        if self.__allow_abort:
            self.__is_aborted = True

        return self.__is_aborted


@dataclass(slots=True, repr=True)
class FullScannerConfig:
    scanner_type: Literal["full"]


@dataclass(slots=True, repr=True)
class NoScannerConfig:
    scanner_type: Literal["nothing"]


@dataclass(slots=True, repr=True)
class ConfigScannerConfig:
    scanner_type: Literal["config"]
    config: Annotated[Optional[ConfigScannerDict], OneOf]


ScannerConfig = Annotated[
    FullScannerConfig | NoScannerConfig | ConfigScannerConfig,
    OneOf,
]


def get_scanner_from_config(
    config: ScannerConfig,
    language_scanner: LanguageScanner,
    metadata_scanner: MetadataScanner,
) -> Scanner:
    match config.scanner_type:
        case "config":
            return ConfigScanner(
                language_scanner=language_scanner,
                metadata_scanner=metadata_scanner,
                config=config.config,
            )
        case "full":
            return FullScanner(
                language_scanner=language_scanner,
                metadata_scanner=metadata_scanner,
            )
        case "nothing":
            return NoScanner(
                language_scanner=language_scanner,
                metadata_scanner=metadata_scanner,
            )
        case _:
            assert_never(config.scanner_type)


# TODO add time based scanner
# TODO optionally ask user for input each x steps as option for other scanners
