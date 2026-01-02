import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Self,
    TypedDict,
    assert_never,
    override,
)

import pyperclip
from questionary import Choice, Question, Separator, select
from questionary.prompts.common import FormattedText

from content.general import MissingOverrideError
from content.language import Language
from content.prediction import Prediction, PredictionBest
from helper.apischema import OneOf
from helper.log import get_logger
from helper.terminal import Terminal


class LanguagePicker:
    def __init__(
        self: Self,
    ) -> None:
        pass

    def pick_language(
        self: Self,
        path: Path,  # noqa: ARG002
        prediction: Prediction,  # noqa: ARG002
    ) -> Optional[Language]:
        raise MissingOverrideError


class NoLanguagePicker(LanguagePicker):
    def __init__(
        self: Self,
    ) -> None:
        super().__init__()

    @override
    def pick_language(
        self: Self,
        path: Path,
        prediction: Prediction,
    ) -> Optional[Language]:
        return None


# TODO: is there a better way?
class InteractiveLanguagePickerDict(TypedDict, total=False):
    entries_to_show: int
    show_full_list: bool
    play_sound: bool


class InteractiveLanguagePickerDictTotal(TypedDict, total=True):
    entries_to_show: int
    show_full_list: bool
    play_sound: bool


def resolve_interactive_config(
    config: Optional[InteractiveLanguagePickerDict],
) -> InteractiveLanguagePickerDictTotal:
    defaults: InteractiveLanguagePickerDictTotal = {
        "entries_to_show": 10,
        "show_full_list": False,
        "play_sound": True,
    }

    loaded_dict: Optional[InteractiveLanguagePickerDict] = config
    result: InteractiveLanguagePickerDictTotal = defaults

    if loaded_dict is not None:
        result["entries_to_show"] = loaded_dict.get(
            "entries_to_show",
            defaults["entries_to_show"],
        )

        result["show_full_list"] = loaded_dict.get(
            "show_full_list",
            defaults["show_full_list"],
        )

        result["play_sound"] = loaded_dict.get(
            "play_sound",
            defaults["play_sound"],
        )
    else:
        result["entries_to_show"] = defaults["entries_to_show"]
        result["show_full_list"] = defaults["show_full_list"]
        result["play_sound"] = defaults["play_sound"]

    return result


@dataclass()
class PredictionBestSelectResult:
    select_result_type: Literal["prediction_best"]
    value: PredictionBest


class SelectedType(Enum):
    open = "open"
    no_language = "no_language"
    copy = " copy"
    more = "more"
    unknown = "unknown"


@dataclass()
class ManualSelectResult:
    select_result_type: Literal["manual"]
    selected: SelectedType


type SelectResult = PredictionBestSelectResult | ManualSelectResult


def construct_choice(title: FormattedText, value: SelectResult) -> Choice:
    return Choice(title=title, value=value)


def ask_question(question: Question) -> Optional[SelectResult] | Any:
    return question.ask()


def open_file(path: Path) -> None:
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path], check=True)  # noqa: S603, S607
    elif platform.system() == "Windows":
        subprocess.run(  # noqa: S602
            ["cmd", "/c", "start", "", path],  # noqa: S607
            shell=True,
            check=True,
        )
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", path], check=True)  # noqa: S603, S607
    else:
        msg = "Unsupported operating system"
        raise OSError(msg)


def copy_to_clipboard(path: Path) -> None:
    pyperclip.copy(str(path.absolute()))


def play_notification_sound() -> None:
    print("\a")  # noqa: T201


logger: Logger = get_logger()


INCREASE_STEP_FOR_SELECTOR: int = 5


class InteractiveLanguagePicker(LanguagePicker):
    __config: InteractiveLanguagePickerDictTotal

    def __init__(self: Self, *, config: InteractiveLanguagePickerDictTotal) -> None:
        super().__init__()
        self.__config = config

    def __get_manual_choices(
        self: Self,
        path: Path,
    ) -> list[Choice]:
        result: list[Choice] = []

        result.append(
            construct_choice(
                title=[
                    ("fg:ansiblue", "[open]"),
                    ("fg:ansigreen", " '"),
                    ("", f"{path}"),
                    ("fg:ansigreen", "'"),
                ],
                value=ManualSelectResult("manual", SelectedType.open),
            ),
        )

        result.append(
            construct_choice(
                title=[
                    ("fg:ansiblue", "[copy path]"),
                    ("fg:ansigreen", " '"),
                    ("", f"{path}"),
                    ("fg:ansigreen", "'"),
                ],
                value=ManualSelectResult("manual", SelectedType.copy),
            ),
        )

        result.append(
            construct_choice(
                title=[("fg:ansiblue", "[more]")],
                value=ManualSelectResult("manual", SelectedType.more),
            ),
        )

        result.append(
            construct_choice(
                title=[("fg:ansiblue", "[unknown language]")],
                value=ManualSelectResult("manual", SelectedType.unknown),
            ),
        )

        result.append(
            construct_choice(
                title=[("fg:ansiblue", "[no language]")],
                value=ManualSelectResult("manual", SelectedType.no_language),
            ),
        )

        return result

    def __get_prediction_choices(
        self: Self,
        best_list: list[PredictionBest],
        length_to_use: int,
    ) -> list[Choice]:

        def format_choice(index: int, value: PredictionBest) -> Choice:
            title: FormattedText = [
                ("fg:ansiblue", f"[{index}]"),
                ("", " "),
                ("", f"{value.language}"),
                ("", " - "),
                ("fg:ansigreen", f"{value.accuracy:.2%}"),
            ]

            return construct_choice(
                title=title,
                value=PredictionBestSelectResult("prediction_best", value),
            )

        result: list[Choice] = [
            format_choice(i, best)
            for i, best in enumerate(best_list)
            if i < length_to_use
        ]

        return result

    def __get_choices(
        self: Self,
        path: Path,
        best_list: list[PredictionBest],
        length_to_use: int,
    ) -> list[Choice]:

        result: list[Choice] = []

        prediction_choices = self.__get_prediction_choices(best_list, length_to_use)
        result.extend(prediction_choices)

        result.append(Separator())

        manual_choices = self.__get_manual_choices(path)
        result.extend(manual_choices)

        return result

    @override
    def pick_language(
        self: Self,
        path: Path,
        prediction: Prediction,
    ) -> Optional[Language]:
        with Terminal.clear_block(clear_on_entry=False):
            if self.__config["play_sound"]:
                play_notification_sound()

            best_list: list[PredictionBest] = prediction.get_best_list()
            length_to_use: int = (
                len(best_list)
                if self.__config["show_full_list"]
                else self.__config["entries_to_show"]
            )

            while True:
                choices = self.__get_choices(
                    path,
                    best_list,
                    length_to_use,
                )

                question = select(
                    "Select the desired option:",
                    choices=choices,
                    default=choices[0],
                )

                result = ask_question(question)
                if result is None:
                    continue

                if not isinstance(
                    result,
                    PredictionBestSelectResult,
                ) and not isinstance(result, ManualSelectResult):
                    continue

                match result.select_result_type:
                    case "manual":
                        manual_value = result
                        match manual_value.selected:
                            case SelectedType.open:
                                try:
                                    open_file(path)
                                except RuntimeError as err:
                                    msg: str = f"Couldn't open file '{path}':\n{err}"
                                    logger.warning(msg)
                                # fall trough and run the loop again
                            case SelectedType.copy:
                                try:
                                    copy_to_clipboard(path)
                                except RuntimeError as err:
                                    cb_err_msg: str = (
                                        f"Couldn't copy file path to clipboard: '{path}':\n{err}"
                                    )
                                    logger.warning(cb_err_msg)
                                # fall trough and run the loop again
                            case SelectedType.no_language:
                                return Language.no_language()
                            case SelectedType.unknown:
                                return Language.get_default()
                            case SelectedType.more:
                                length_to_use = min(
                                    len(best_list),
                                    length_to_use + INCREASE_STEP_FOR_SELECTOR,
                                )
                                # fall trough and run the loop again
                            case _:
                                assert_never(manual_value.selected)
                    case "prediction_best":
                        prediction_value = result
                        return prediction_value.value.language

                    case _:
                        assert_never(result.select_result_type)


@dataclass
class NoLanguagePickerConfig:
    picker_type: Literal["none"]


@dataclass
class InteractiveLanguagePickerConfig:
    picker_type: Literal["interactive"]
    config: Annotated[Optional[InteractiveLanguagePickerDict], OneOf]


LanguagePickerConfig = Annotated[
    NoLanguagePickerConfig | InteractiveLanguagePickerConfig,
    OneOf,
]


def get_picker_from_config(
    config: LanguagePickerConfig,
) -> LanguagePicker:
    match config.picker_type:
        case "none":
            return NoLanguagePicker()
        case "interactive":
            resolved_config = resolve_interactive_config(
                config.config,
            )
            return InteractiveLanguagePicker(config=resolved_config)
