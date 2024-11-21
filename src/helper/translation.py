import gettext
import locale
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, cast

SupportedLanguage = Literal["en", "de"]
SUPPORTED_LANGUAGES: list[SupportedLanguage] = ["de", "en"]
DEFAULT_LANGUAGE: SupportedLanguage = "en"


def get_current_language() -> SupportedLanguage:
    lang_str, _ = locale.getlocale()
    if lang_str is None:
        return DEFAULT_LANGUAGE

    [lang, *_] = lang_str.split("_")
    if lang in SUPPORTED_LANGUAGES:
        return cast(SupportedLanguage, lang)

    return DEFAULT_LANGUAGE


TranslationFunction = Callable[[str], str]


__global_translation: Optional[TranslationFunction] = None


def get_translator() -> TranslationFunction:
    global __global_translation  # noqa: PLW0603
    if __global_translation is None:
        translation = gettext.translation(
            "video_language_detect",
            localedir=(Path(__file__).parent.parent.parent / "locales"),
            languages=[get_current_language()],
        )
        translation.install()

        __global_translation = translation.gettext

    return __global_translation
