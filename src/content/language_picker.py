import platform
import subprocess
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
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
import questionary
import questionary.prompts
import questionary.prompts.common

from content.language import Language
from helper.apischema import OneOf
from helper.decorator import decorate_class
from helper.log import get_logger
from helper.terminal import ClearContextManager, Terminal

if TYPE_CHECKING:
    from content.prediction import Prediction, PredictionBest


@decorate_class(slots=True)
class LanguagePicker(ABC):
    def __init__(
        self: Self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def pick_language(
        self: Self,
        path: Path,
        prediction: Prediction,
    ) -> Optional[Language]: ...


@decorate_class(slots=True)
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


class InteractiveLanguagePickerDict(TypedDict, total=False):
    entries_to_show: int
    show_full_list: bool
    play_sound: bool


@dataclass(slots=True, repr=True)
class InteractiveLanguagePickerData:
    entries_to_show: int
    show_full_list: bool
    play_sound: bool


def resolve_interactive_config(
    config: Optional[InteractiveLanguagePickerDict],
) -> InteractiveLanguagePickerData:
    defaults: InteractiveLanguagePickerData = InteractiveLanguagePickerData(
        entries_to_show=10,
        show_full_list=False,
        play_sound=True,
    )

    loaded_dict: Optional[InteractiveLanguagePickerDict] = config
    result: InteractiveLanguagePickerData = defaults

    if loaded_dict is not None:
        result.entries_to_show = loaded_dict.get(
            "entries_to_show",
            defaults.entries_to_show,
        )

        result.show_full_list = loaded_dict.get(
            "show_full_list",
            defaults.show_full_list,
        )

        result.play_sound = loaded_dict.get(
            "play_sound",
            defaults.play_sound,
        )
    else:
        result.entries_to_show = defaults.entries_to_show
        result.show_full_list = defaults.show_full_list
        result.play_sound = defaults.play_sound

    return result


@dataclass(slots=True, repr=True)
class PredictionBestSelectResult:
    select_result_type: Literal["prediction_best"]
    value: PredictionBest


class SelectedType(Enum):
    open = "open"
    no_language = "no_language"
    copy = " copy"
    more = "more"
    unknown = "unknown"


@dataclass(slots=True, repr=True)
class ManualSelectResult:
    select_result_type: Literal["manual"]
    selected: SelectedType


type SelectResult = PredictionBestSelectResult | ManualSelectResult


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


class ChoiceColorType(Enum):
    fg = "fg"
    bg = "bg"


class ChoiceColorValue(Enum):
    blue = "ansiblue"
    green = "ansigreen"


@dataclass(slots=True, repr=True)
class ChoiceColor:
    type: ChoiceColorType
    color: ChoiceColorValue


@dataclass(slots=True, repr=True)
class ChoiceTitle:
    color: Optional[ChoiceColor]
    content: str


@decorate_class(slots=True)
class ChoiceInterface:
    def __init__(self: Self) -> None:
        super().__init__()


@decorate_class(slots=True)
class ChoiceManagerInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    @abstractmethod
    def get_choice(
        self: Self,
        title: list[ChoiceTitle],
        value: SelectResult,
    ) -> ChoiceInterface: ...

    @abstractmethod
    def get_separator(
        self: Self,
    ) -> ChoiceInterface: ...

    @abstractmethod
    def picker_ctx(
        self: Self,
    ) -> AbstractContextManager[None]: ...

    @abstractmethod
    def ask_question(
        self: Self,
        message: str,
        choices: list[ChoiceInterface],
        default: ChoiceInterface,
    ) -> Optional[SelectResult]: ...


@decorate_class(slots=True)
class TUIChoice(ChoiceInterface):
    __impl: questionary.Choice

    def __init__(self: Self, impl: questionary.Choice) -> None:
        super().__init__()
        self.__impl = impl

    @property
    def impl(self: Self) -> questionary.Choice:
        return self.__impl


@decorate_class(slots=True)
class TuiContextWrapper(AbstractContextManager[None]):
    __underlying: ClearContextManager

    def __init__(self: Self, underlying: ClearContextManager) -> None:
        super().__init__()
        self.__underlying = underlying

    @override
    def __enter__(self: Self) -> None:
        self.__underlying.__enter__()

    @override
    def __exit__(
        self: Self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        self.__underlying.__exit__(
            exc_type,
            exc_val,
            exc_tb,
        )
        return False


@decorate_class(slots=True)
class TUIChoiceManager(ChoiceManagerInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    @staticmethod
    def __get_formatted_color(color: Optional[ChoiceColor]) -> str:
        if color is None:
            return ""

        return f"{color.type.value}:{color.color.value}"

    @staticmethod
    def __get_formatted_title(
        title: list[ChoiceTitle],
    ) -> questionary.prompts.common.FormattedText:
        return [
            (TUIChoiceManager.__get_formatted_color(segment.color), segment.content)
            for segment in title
        ]

    @override
    def get_choice(
        self: Self,
        title: list[ChoiceTitle],
        value: SelectResult,
    ) -> ChoiceInterface:
        title_formatted = TUIChoiceManager.__get_formatted_title(title)
        choice = questionary.Choice(title=title_formatted, value=value)
        return TUIChoice(choice)

    @override
    def get_separator(
        self: Self,
    ) -> ChoiceInterface:
        return TUIChoice(questionary.Separator())

    @override
    def picker_ctx(
        self: Self,
    ) -> AbstractContextManager[None]:
        return TuiContextWrapper(Terminal.clear_block(clear_on_entry=False))

    @staticmethod
    def __get_underlying_choice(choice: ChoiceInterface) -> questionary.Choice:
        if isinstance(choice, TUIChoice):
            return choice.impl

        msg = "Implementation Error: used wrong choices with wrong choices manager!"
        raise RuntimeError(msg)

    @override
    def ask_question(
        self: Self,
        message: str,
        choices: list[ChoiceInterface],
        default: ChoiceInterface,
    ) -> Optional[SelectResult]:
        choices_impl = [
            TUIChoiceManager.__get_underlying_choice(choice) for choice in choices
        ]
        default_impl = TUIChoiceManager.__get_underlying_choice(default)
        question = questionary.select(
            message,
            choices=choices_impl,
            default=default_impl,
        )
        result: Optional[SelectResult] | Any = question.ask()

        if not isinstance(
            result,
            PredictionBestSelectResult,
        ) and not isinstance(result, ManualSelectResult):
            return None

        return result


@decorate_class(slots=True)
class InteractiveLanguagePicker(LanguagePicker):
    __config: InteractiveLanguagePickerData
    __manager: ChoiceManagerInterface

    def __init__(
        self: Self,
        *,
        config: InteractiveLanguagePickerData,
        manager: ChoiceManagerInterface,
    ) -> None:
        super().__init__()
        self.__config = config
        self.__manager = manager

    def __get_manual_choices(
        self: Self,
        path: Path,
    ) -> list[ChoiceInterface]:
        result: list[ChoiceInterface] = []

        result.append(
            self.__manager.get_choice(
                title=[
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.blue,
                        ),
                        content="[open]",
                    ),
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.green,
                        ),
                        content=" '",
                    ),
                    ChoiceTitle(color=None, content=f"{path}"),
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.green,
                        ),
                        content="'",
                    ),
                ],
                value=ManualSelectResult("manual", SelectedType.open),
            ),
        )

        result.append(
            self.__manager.get_choice(
                title=[
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.blue,
                        ),
                        content="[copy path]",
                    ),
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.green,
                        ),
                        content=" '",
                    ),
                    ChoiceTitle(color=None, content=f"{path}"),
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.green,
                        ),
                        content="'",
                    ),
                ],
                value=ManualSelectResult("manual", SelectedType.copy),
            ),
        )

        result.append(
            self.__manager.get_choice(
                title=[
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.blue,
                        ),
                        content="[more]",
                    ),
                ],
                value=ManualSelectResult("manual", SelectedType.more),
            ),
        )

        result.append(
            self.__manager.get_choice(
                title=[
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.blue,
                        ),
                        content="[unknown language]",
                    ),
                ],
                value=ManualSelectResult("manual", SelectedType.unknown),
            ),
        )

        result.append(
            self.__manager.get_choice(
                title=[
                    ChoiceTitle(
                        color=ChoiceColor(
                            type=ChoiceColorType.fg,
                            color=ChoiceColorValue.blue,
                        ),
                        content="[no language]",
                    ),
                ],
                value=ManualSelectResult("manual", SelectedType.no_language),
            ),
        )

        return result

    def __get_prediction_choices(
        self: Self,
        best_list: list[PredictionBest],
        length_to_use: int,
    ) -> list[ChoiceInterface]:

        def format_choice(index: int, value: PredictionBest) -> ChoiceInterface:
            title: list[ChoiceTitle] = [
                ChoiceTitle(
                    color=ChoiceColor(
                        type=ChoiceColorType.fg,
                        color=ChoiceColorValue.blue,
                    ),
                    content=f"[{index}]",
                ),
                ChoiceTitle(color=None, content=" "),
                ChoiceTitle(color=None, content=f"{value.language}"),
                ChoiceTitle(color=None, content=" - "),
                ChoiceTitle(
                    color=ChoiceColor(
                        type=ChoiceColorType.fg,
                        color=ChoiceColorValue.green,
                    ),
                    content=f"{value.accuracy:.2%}",
                ),
            ]

            return self.__manager.get_choice(
                title=title,
                value=PredictionBestSelectResult("prediction_best", value),
            )

        result: list[ChoiceInterface] = [
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
    ) -> list[ChoiceInterface]:

        result: list[ChoiceInterface] = []

        prediction_choices = self.__get_prediction_choices(best_list, length_to_use)
        result.extend(prediction_choices)

        result.append(self.__manager.get_separator())

        manual_choices = self.__get_manual_choices(path)
        result.extend(manual_choices)

        return result

    @override
    def pick_language(
        self: Self,
        path: Path,
        prediction: Prediction,
    ) -> Optional[Language]:
        with self.__manager.picker_ctx():
            if self.__config.play_sound:
                play_notification_sound()

            best_list: list[PredictionBest] = prediction.get_best_list()
            length_to_use: int = (
                len(best_list)
                if self.__config.show_full_list
                else self.__config.entries_to_show
            )

            while True:
                choices = self.__get_choices(
                    path,
                    best_list,
                    length_to_use,
                )

                result = self.__manager.ask_question(
                    "Select the desired option:",
                    choices=choices,
                    default=choices[0],
                )

                if result is None:
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


@dataclass(slots=True, repr=True)
class NoLanguagePickerConfig:
    picker_type: Literal["none"]


@dataclass(slots=True, repr=True)
class InteractiveLanguagePickerConfig:
    picker_type: Literal["interactive"]
    config: Annotated[Optional[InteractiveLanguagePickerDict], OneOf]


LanguagePickerConfig = Annotated[
    NoLanguagePickerConfig | InteractiveLanguagePickerConfig,
    OneOf,
]


def get_picker_from_config(
    config: LanguagePickerConfig,
    choice_manager: ChoiceManagerInterface,
) -> LanguagePicker:
    match config.picker_type:
        case "none":
            return NoLanguagePicker()
        case "interactive":
            resolved_config = resolve_interactive_config(
                config.config,
            )
            return InteractiveLanguagePicker(
                config=resolved_config,
                manager=choice_manager,
            )
        case _:
            assert_never(config.picker_type)
