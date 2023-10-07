from dataclasses import dataclass
from typing import Literal, Optional, Self, TypedDict, cast, override

from classifier import Classifier
from content.base_class import LanguageScanner, ScanType
from content.general import EpisodeDescription


class StaticLanguageScanner(LanguageScanner):
    __value: bool

    def __init__(
        self: Self,
        classifier: Classifier,
        *,
        value: bool,
    ) -> None:
        super().__init__(classifier)
        self.__value = value

    @override
    def should_scan(
        self: Self,
        description: EpisodeDescription,
        scan_type: ScanType,
    ) -> bool:
        return self.__value


class FullLanguageScanner(StaticLanguageScanner):
    def __init__(
        self: Self,
        classifier: Classifier,
    ) -> None:
        super().__init__(classifier, value=True)


class NoLanguageScanner(StaticLanguageScanner):
    def __init__(
        self: Self,
        classifier: Classifier,
    ) -> None:
        super().__init__(classifier, value=False)


# TODO: is there a better way?
class ConfigScannerDict(TypedDict, total=False):
    start_position: int
    scan_amount: int


class ConfigScannerDictTotal(TypedDict, total=True):
    start_position: int
    scan_amount: int
    # TODO: print progress option


class ConfigLanguageScanner(LanguageScanner):
    __start_position: int
    __scan_amount: int
    __current_position: int

    @property
    def __defaults(self: Self) -> ConfigScannerDictTotal:
        return {"start_position": 0, "scan_amount": 100}

    def __init__(
        self: Self,
        classifier: Classifier,
        *,
        config: Optional[ConfigScannerDict] = None,
    ) -> None:
        super().__init__(classifier)
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
        else:
            self.__start_position = self.__defaults["start_position"]
            self.__scan_amount = self.__defaults["scan_amount"]

        self.__current_position = 0

    @override
    def should_scan(
        self: Self,
        description: EpisodeDescription,
        scan_type: ScanType,
    ) -> bool:
        if (self.__start_position >= self.__current_position) and (
            self.__start_position < (self.__current_position + self.__scan_amount)
        ):
            self.__current_position += 1
            return True

        self.__current_position += 1
        return False


@dataclass
class FullLanguageScannerConfig:
    scanner_type: Literal["full"]


@dataclass
class NoLanguageScannerConfig:
    scanner_type: Literal["nothing"]


@dataclass
class ConfigScannerConfig:
    scanner_type: Literal["config"]
    config: Optional[ConfigScannerDict]


ScannerConfig = (
    FullLanguageScannerConfig | NoLanguageScannerConfig | ConfigScannerConfig
)


def get_scanner_from_config(
    config: ScannerConfig,
    classifier: Classifier,
) -> LanguageScanner:
    match config.scanner_type:
        case "config":
            return ConfigLanguageScanner(
                classifier=classifier,
                config=cast(ConfigScannerConfig, config).config,
            )
        case "full":
            return FullLanguageScanner(classifier=classifier)
        case "nothing":
            return NoLanguageScanner(classifier=classifier)


# TODO add time based scanner
# TODO optionally ask user for input each x steps as option for other scanners
