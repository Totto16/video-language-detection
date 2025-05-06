import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Annotated, Literal, Optional, Self, TypedDict, cast, override

from questionary import Choice, Question, Separator, select
from questionary.prompts.common import FormattedText

from content.general import (
    MissingOverrideError,
    OneOf,
)
from content.language import Language
from content.prediction import Prediction, PredictionBest
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


@dataclass()
class PredictionBestSelectResult:
    select_result_type: Literal["prediction_best"]
    value: PredictionBest


class SelectedType(Enum):
    open = "open"
    no_language = "no_language"


@dataclass()
class ManualSelectResult:
    select_result_type: Literal["manual"]
    selected: SelectedType


SelectResult = PredictionBestSelectResult | ManualSelectResult


def construct_choice(title: FormattedText, value: SelectResult) -> Choice:
    return Choice(title=title, value=value)


def ask_question(question: Question) -> Optional[SelectResult]:
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


def play_notification_sound() -> None:
    print("\a")  # noqa: T201


logger: Logger = get_logger()


class InteractiveLanguagePicker(LanguagePicker):
    __entries_to_show: int
    __show_full_list: bool
    __play_sound: bool

    def __init__(
        self: Self,
        *,
        config: Optional[InteractiveLanguagePickerDict] = None,
    ) -> None:
        super().__init__()

        loaded_dict: Optional[InteractiveLanguagePickerDict] = config
        if loaded_dict is not None:
            self.__entries_to_show = loaded_dict.get(
                "entries_to_show",
                self.__defaults["entries_to_show"],
            )

            self.__show_full_list = loaded_dict.get(
                "show_full_list",
                self.__defaults["show_full_list"],
            )

            self.__play_sound = loaded_dict.get(
                "play_sound",
                self.__defaults["play_sound"],
            )
        else:
            self.__entries_to_show = self.__defaults["entries_to_show"]
            self.__show_full_list = self.__defaults["show_full_list"]
            self.__play_sound = self.__defaults["play_sound"]

    @property
    def __defaults(self: Self) -> InteractiveLanguagePickerDictTotal:
        return {
            "entries_to_show": 10,
            "show_full_list": False,
            "play_sound": True,
        }

    def __get_choices(
        self: Self,
        path: Path,
        prediction: Prediction,
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

        best_list = prediction.get_best_list()

        items_count: int = (
            len(best_list) if self.__show_full_list else self.__entries_to_show
        )

        result: list[Choice] = [
            format_choice(i, best)
            for i, best in enumerate(best_list)
            if i < items_count
        ]

        result.append(Separator())

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
                title=[("fg:ansiblue", "[no language]")],
                value=ManualSelectResult("manual", SelectedType.no_language),
            ),
        )

        return result

    @override
    def pick_language(
        self: Self,
        path: Path,
        prediction: Prediction,
    ) -> Optional[Language]:
        with Terminal.clear_block():

            choices = self.__get_choices(path, prediction)

            question = select(
                "Select the desired option:",
                choices=choices,
                default=choices[0],
            )

            if self.__play_sound:
                play_notification_sound()

            while True:
                result = ask_question(question)
                if result is None:
                    continue

                match result.select_result_type:
                    case "manual":
                        manual_value = cast(ManualSelectResult, result)
                        match manual_value.selected:
                            case SelectedType.open:
                                try:
                                    open_file(path)
                                except RuntimeError as err:
                                    msg: str = f"Couldn't open file '{path}':\n{err}"
                                    logger.warning(msg)
                                # fall trough and run the loop again
                            case SelectedType.no_language:
                                return None
                    case "prediction_best":
                        prediction_value = cast(PredictionBestSelectResult, result)
                        return prediction_value.value.language


@dataclass
class NoLanguagePickerConfig:
    picker_type: Literal["none"]


@dataclass
class InteractiveLanguagePickerConfig:
    picker_type: Literal["interactive"]
    config: Optional[InteractiveLanguagePickerDict]


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
            return InteractiveLanguagePicker(
                config=cast(InteractiveLanguagePickerConfig, config).config,
            )
