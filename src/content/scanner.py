import json
from configparser import ConfigParser
from pathlib import Path
from typing import Optional, Self, TypedDict

from classifier import Classifier
from helper.timestamp import parse_int_safely
from typing_extensions import override

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
        super().__init__(classifier, value=True)


# TODO: is there a better way?
class PartialScannerDict(TypedDict, total=False):
    start_position: int
    scan_amount: int


class PartialScannerDictTotal(TypedDict, total=True):
    start_position: int
    scan_amount: int
    #TODO: print progress option


INI_SETTINGS_SECTION_KEY = "settings"


class PartialLanguageScanner(LanguageScanner):
    __start_position: int
    __scan_amount: int
    __current_position: int

    @property
    def __defaults(self: Self) -> PartialScannerDictTotal:
        return {"start_position": 0, "scan_amount": 100}

    def __init__(
        self: Self,
        classifier: Classifier,
        *,
        config_file: Path = Path("./config.ini"),
    ) -> None:
        super().__init__(classifier)
        loaded_dict: Optional[PartialScannerDict] = None

        if config_file.exists():
            with config_file.open(mode="r") as file:
                suffix: str = config_file.suffix[1:]
                match suffix:
                    case "json":
                        loaded_dict = json.load(file)
                    case "ini":
                        config = ConfigParser()
                        config.read(config_file)
                        if INI_SETTINGS_SECTION_KEY in config:
                            temp_dict = dict(config.items(INI_SETTINGS_SECTION_KEY))

                            loaded_dict = {}
                            if temp_dict.get("start_position", None) is not None:
                                int_result = parse_int_safely(
                                    temp_dict["start_position"],
                                )
                                if int_result is not None:
                                    loaded_dict["start_position"] = int_result
                            if temp_dict.get("scan_amount", None) is not None:
                                int_result = parse_int_safely(
                                    temp_dict["scan_amount"],
                                )
                                if int_result is not None:
                                    loaded_dict["scan_amount"] = int_result

                    case _:
                        msg = f"Config not loadable from '{suffix}' file!"
                        raise RuntimeError(msg)

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


# TODO add time based scanner
# TODO optionally ask user for input each x steps as option for other scanners
