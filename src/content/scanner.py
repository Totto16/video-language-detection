from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Optional, Self, TypedDict, cast, override

from content.base_class import LanguageScanner, Scanner
from content.general import OneOf
from content.metadata.metadata import InternalMetadataType
from content.metadata.scanner import MetadataScanner
from content.shared import ScanKind, ScanType


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


class FullScanner(StaticScanner):
    def __init__(
        self: Self,
        language_scanner: LanguageScanner,
        metadata_scanner: MetadataScanner,
    ) -> None:
        super().__init__(language_scanner, metadata_scanner, value=True)


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


# TODO: is there a better way?
class ConfigScannerDict(TypedDict, total=False):
    start_position: int
    scan_amount: int
    allow_abort: bool
    types: ScannerTypes


class ConfigScannerDictTotal(TypedDict, total=True):
    start_position: int
    scan_amount: int
    allow_abort: bool
    types: ScannerTypes
    # TODO: print progress option


class ConfigScanner(Scanner):
    __start_position: tuple[int, int]
    __scan_amount: tuple[int, int]
    __allow_abort: bool
    __types: ScannerTypes
    # state
    __current_position: tuple[int, int]
    __is_aborted: bool

    @property
    def __defaults(self: Self) -> ConfigScannerDictTotal:
        return {
            "start_position": 0,
            "scan_amount": 100,
            "allow_abort": True,
            "types": ScannerTypes.both,
        }

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
            start_position: int = loaded_dict.get(
                "start_position",
                self.__defaults["start_position"],
            )
            self.__start_position = (start_position, start_position)

            scan_amount: int = loaded_dict.get(
                "scan_amount",
                self.__defaults["scan_amount"],
            )
            self.__scan_amount = (scan_amount, scan_amount)

            self.__allow_abort = loaded_dict.get(
                "allow_abort",
                self.__defaults["allow_abort"],
            )
            self.__types = loaded_dict.get(
                "types",
                self.__defaults["types"],
            )
        else:
            start_position = self.__defaults["start_position"]
            self.__start_position = (start_position, start_position)

            scan_amount = self.__defaults["scan_amount"]
            self.__scan_amount = (scan_amount, scan_amount)

            self.__allow_abort = self.__defaults["allow_abort"]
            self.__types = self.__defaults["types"]

        self.__current_position = (0, 0)
        self.__is_aborted = False

    def __get_index_for(self: Self, scan_kind: ScanKind) -> int:
        return 0 if scan_kind == ScanKind.language else 1

    def __is_type(self: Self, scan_kind: ScanKind) -> bool:
        if self.__types == ScannerTypes.both:
            return True

        if self.__types == ScannerTypes.only_language:
            return scan_kind == ScanKind.language

        return scan_kind == ScanKind.metadata

    def __should_scan_kind(self: Self, scan_kind: ScanKind) -> bool:
        if not self.__is_type(scan_kind):
            return False

        ## TODO: set somewhere, e.g. in gui
        if self.__is_aborted and self.__allow_abort:
            return False

        index = self.__get_index_for(scan_kind)

        result = False

        if (self.__start_position[index] <= self.__current_position[index]) and (
            self.__current_position[index] - self.__start_position[index]
            < self.__scan_amount[index]
        ):
            result = True

        pos1, pos2 = self.__current_position

        add1 = 1 if index == 0 else 0
        add2 = 1 if index == 1 else 0

        self.__current_position = (pos1 + add1, pos2 + add2)

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


@dataclass
class FullScannerConfig:
    scanner_type: Literal["full"]


@dataclass
class NoScannerConfig:
    scanner_type: Literal["nothing"]


@dataclass
class ConfigScannerConfig:
    scanner_type: Literal["config"]
    config: Optional[ConfigScannerDict]


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
                config=cast(ConfigScannerConfig, config).config,
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


# TODO add time based scanner
# TODO optionally ask user for input each x steps as option for other scanners
