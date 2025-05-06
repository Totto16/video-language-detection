from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional, Self, override

from content.base_class import LanguagePicker
from content.general import OneOf
from content.language import Language
from content.prediction import Prediction


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


class InteractiveLanguagePicker(LanguagePicker):
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
        # TODO
        return None


@dataclass
class NoLanguagePickerConfig:
    picker_type: Literal["none"]


@dataclass
class InteractiveLanguagePickerConfig:
    picker_type: Literal["interactive"]


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
            return InteractiveLanguagePicker()
