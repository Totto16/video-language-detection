from dataclasses import dataclass
from typing import Literal, Optional, Self, TypedDict, cast, override

from content.base_class import LanguageScanner, Scanner
from content.general import EpisodeDescription
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
    def should_scan(
        self: Self,
        description: EpisodeDescription,
        scan_type: ScanType,
        scan_kind: ScanKind,
    ) -> bool:
        return self.__value


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


# TODO: is there a better way?
class ConfigScannerDict(TypedDict, total=False):
    start_position: int
    scan_amount: int
    allow_abort: bool


class ConfigScannerDictTotal(TypedDict, total=True):
    start_position: int
    scan_amount: int
    allow_abort: bool
    # TODO: print progress option


class ConfigScanner(Scanner):
    __start_position: int
    __scan_amount: int
    __allow_abort: bool
    # state
    __current_position: int
    __is_aborted: bool

    @property
    def __defaults(self: Self) -> ConfigScannerDictTotal:
        return {"start_position": 0, "scan_amount": 100, "allow_abort": True}

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
            self.__start_position = loaded_dict.get(
                "start_position",
                self.__defaults["start_position"],
            )
            self.__scan_amount = loaded_dict.get(
                "scan_amount",
                self.__defaults["scan_amount"],
            )
            self.__allow_abort = loaded_dict.get(
                "allow_abort",
                self.__defaults["allow_abort"],
            )
        else:
            self.__start_position = self.__defaults["start_position"]
            self.__scan_amount = self.__defaults["scan_amount"]
            self.__allow_abort = self.__defaults["allow_abort"]

        self.__current_position = 0
        self.__is_aborted = False

    @override
    def should_scan(
        self: Self,
        description: EpisodeDescription,
        scan_type: ScanType,
        scan_kind: ScanKind,
    ) -> bool:
        ## TODO: set somewhere, e.g. in gui
        if self.__is_aborted:
            return False

        if (self.__start_position >= self.__current_position) and (
            self.__start_position < (self.__current_position + self.__scan_amount)
        ):
            self.__current_position += 1
            return True

        self.__current_position += 1
        return False


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


ScannerConfig = FullScannerConfig | NoScannerConfig | ConfigScannerConfig


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
