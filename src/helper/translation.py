import builtins
import gettext
import locale
from collections.abc import Callable
from pathlib import Path
from typing import Literal

type SupportedLanguage = Literal["en", "de"]
SUPPORTED_LANGUAGES: list[SupportedLanguage] = ["de", "en"]
DEFAULT_LANGUAGE: SupportedLanguage = "en"


def get_current_language() -> SupportedLanguage:
    lang_str, _ = locale.getlocale()
    if lang_str is None:
        return DEFAULT_LANGUAGE

    [lang, *_] = lang_str.split("_")
    if lang in SUPPORTED_LANGUAGES:
        return lang

    return DEFAULT_LANGUAGE


type TranslationFunction = Callable[[str], str]


TRANSLATION_DOMAIN = "video_language_detect"

TRANSLATION_DIR = Path(__file__).parent.parent.parent / "locales"


def get_translator() -> TranslationFunction:
    if "_" not in builtins.__dict__:
        translation = gettext.translation(
            TRANSLATION_DOMAIN,
            localedir=TRANSLATION_DIR,
            languages=[get_current_language()],
        )
        translation.install()

    return builtins.__dict__["_"]
