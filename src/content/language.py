import re
from dataclasses import dataclass
from typing import Annotated, NewType, Optional, Self

from annotated_types import Len, Predicate
from apischema import deserializer, schema, serializer

from helper.translation import get_translator

_ = get_translator()


def ExactLen(length: int) -> Len:  # noqa: N802
    return Len(min_length=length, max_length=length)


# see https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
# and: https://www.loc.gov/standards/iso639-2/php/code_list.php

# this should be ISO 639-2 codes (alpha-3 code)
Alpha3LanguageCode = NewType(
    "Alpha3LanguageCode",
    Annotated[str, ExactLen(3) | Predicate(str.islower)],
)

SHORT_LANGUAGE_STR_PATTERN = r"^([a-z]{2})$"


# this should be ISO 639-1 codes (alpha-2 code)
@schema(pattern=SHORT_LANGUAGE_STR_PATTERN)
class ShortLanguageStr:

    __PrivateStrImpl = NewType(
        "__PrivateStrImpl",
        Annotated[str, ExactLen(2) | Predicate(str.islower)],
    )

    __data: __PrivateStrImpl

    def __init__(self: Self, inp: str | __PrivateStrImpl) -> None:
        data = ShortLanguageStr.__from_raw_str_checked(inp)
        if data is None:
            msg = _("Couldn't get the Short Language String from str '{inp}'").format(
                inp=inp,
            )
            raise RuntimeError(msg)

        self.__data = data

    @staticmethod
    def __from_raw_str_checked(val: str) -> Optional[__PrivateStrImpl]:
        if len(val) != 2:
            return None

        if not val.islower():
            return None

        return ShortLanguageStr.__PrivateStrImpl(val)

    @staticmethod
    def from_str(inp: str) -> Optional["ShortLanguageStr"]:
        val = ShortLanguageStr.__from_raw_str_checked(inp)

        if val is None:
            return None

        return ShortLanguageStr(val)

    @staticmethod
    def from_str_unsafe(inp: str) -> "ShortLanguageStr":
        val: Optional[ShortLanguageStr] = ShortLanguageStr.from_str(inp)
        if val is None:
            msg = _("Couldn't get the Short Language String from str '{inp}'").format(
                inp=inp,
            )
            raise RuntimeError(msg)

        return val

    @serializer
    def serialize(self: Self) -> str:
        return self.__data

    @staticmethod
    def no_lang() -> "ShortLanguageStr":
        return ShortLanguageStr.from_str_unsafe("xx")

    @staticmethod
    def unknown_lang() -> "ShortLanguageStr":
        return ShortLanguageStr.from_str_unsafe("un")

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "ShortLanguageStr":
        match = re.match(SHORT_LANGUAGE_STR_PATTERN, inp)

        if match is None:
            msg = _(
                "Invalid pattern for ShortLanguageStr: got '{inp}', this didn't match the pattern '{pattern}'"  # noqa: COM812
            ).format(inp=inp, pattern=SHORT_LANGUAGE_STR_PATTERN)
            raise TypeError(msg)

        # backwards compatible, as before the enforcing of the two alpha rule, no language wasn't two alpha digits!
        if inp in ["no_lang", "xx"]:
            return ShortLanguageStr.no_lang()

        if inp == "un":
            return ShortLanguageStr.unknown_lang()

        return ShortLanguageStr.from_str_unsafe(inp)

    def __str__(self: Self) -> str:
        return self.__data

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(self.__data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, ShortLanguageStr):
            return self.__data == other.__data

        if isinstance(other, str):
            return self.__data == ShortLanguageStr.from_str(other)

        return False


LongLanguageStr = NewType("LongLanguageStr", str)


@dataclass
class Language:
    short: ShortLanguageStr
    long: LongLanguageStr

    @staticmethod
    def from_str(inp: str) -> Optional["Language"]:
        arr: list[str] = [a.strip() for a in inp.split(":")]
        if len(arr) != 2:
            return None

        short = ShortLanguageStr.from_str(arr[0])

        if short is None:
            return None

        return Language(short=short, long=LongLanguageStr(arr[1]))

    @staticmethod
    def from_str_unsafe(inp: str) -> "Language":
        lan: Optional[Language] = Language.from_str(inp)
        if lan is None:
            msg = _("Couldn't get the Language from str '{inp}'").format(inp=inp)
            raise RuntimeError(msg)

        return lan

    # this is for episodes, that have no real language, for some special episodes of some tv series
    @staticmethod
    def no_language() -> "Language":
        return Language(ShortLanguageStr.no_lang(), LongLanguageStr("No Language"))

    # TODO: get all languages somehow, from the classifier, that supports them all

    # note this is an implementation detail, that should not leak
    @staticmethod
    def __unknown() -> "Language":
        return Language(ShortLanguageStr.unknown_lang(), LongLanguageStr("Unknown"))

    @staticmethod
    def get_default() -> "Language":
        return Language.__unknown()

    @staticmethod
    def is_default_value(language: "Language") -> bool:
        return language == Language.__unknown()

    def __str__(self: Self) -> str:
        return self.long

    def __repr__(self: Self) -> str:
        return f"<Language short: {self.short!r} long: {self.long!r}>"

    def __hash__(self: Self) -> int:
        return hash((self.short, self.long))

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Language):
            return self.short == other.short and self.long == other.long

        return False
