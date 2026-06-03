import re
from dataclasses import dataclass
from typing import Annotated, NewType, Optional, Self

from annotated_types import Len, Predicate
from apischema import deserializer, schema, serializer

from helper.translation import get_translator

__all__: list[str] = ["ExactLen", "Language"]


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

    # this is used, so that init is only callable from the internal class, so that it only gets checked values!
    __PrivateSentinel = NewType("__PrivateSentinel", bool)

    def __init__(
        self: Self,
        data: __PrivateStrImpl,
        *,
        sentinel: __PrivateSentinel,
    ) -> None:
        self.__data = data

    @staticmethod
    def __from_raw_str_checked(
        val: str,
        *,
        valid_check: bool,
    ) -> Optional[__PrivateStrImpl]:
        if len(val) != 2:
            return None

        if not val.islower():
            return None

        if valid_check:
            raise NotImplementedError("TODO")

        return ShortLanguageStr.__PrivateStrImpl(val)

    @staticmethod
    def __from_str_impl(inp: str, *, valid_check: bool) -> Optional["ShortLanguageStr"]:
        val = ShortLanguageStr.__from_raw_str_checked(inp, valid_check=valid_check)

        if val is None:
            return None

        return ShortLanguageStr(
            data=val,
            sentinel=ShortLanguageStr.__PrivateSentinel(True),
        )

    @staticmethod
    def from_str(inp: str) -> Optional["ShortLanguageStr"]:
        return ShortLanguageStr.__from_str_impl(inp, valid_check=True)

    @staticmethod
    def __impl_from_str_unsafe(inp: str, *, valid_check: bool) -> "ShortLanguageStr":
        val: Optional[ShortLanguageStr] = ShortLanguageStr.__from_str_impl(
            inp, valid_check=True
        )
        if val is None:
            msg = _("Couldn't get the Short Language String from str '{inp}'").format(
                inp=inp,
            )
            raise RuntimeError(msg)

        return val

    @staticmethod
    def from_str_unsafe(inp: str) -> "ShortLanguageStr":
        return ShortLanguageStr.__impl_from_str_unsafe(inp, valid_check=True)

    @serializer
    def serialize(self: Self) -> str:
        return self.__data

    @staticmethod
    def no_lang() -> "ShortLanguageStr":
        return ShortLanguageStr.__impl_from_str_unsafe("xx", valid_check=False)

    @staticmethod
    def unknown_lang() -> "ShortLanguageStr":
        return ShortLanguageStr.__impl_from_str_unsafe("un", valid_check=False)

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


class Language:
    __short: ShortLanguageStr
    __long: LongLanguageStr

    # this is used, so that init is only callable from the internal class, so that it only gets checked values!
    __PrivateSentinel = NewType("__PrivateSentinel", bool)

    def __init__(
        self: Self,
        short: ShortLanguageStr,
        long: LongLanguageStr,
        *,
        sentinel: __PrivateSentinel,
    ) -> None:
        self.__short = short
        self.__long = long

    @property
    def short(self: Self) -> ShortLanguageStr:
        return self.__short

    @property
    def long(self: Self) -> LongLanguageStr:
        return self.__long

    @staticmethod
    def from_str(inp: str) -> Optional["Language"]:
        arr: list[str] = [a.strip() for a in inp.split(":")]
        if len(arr) != 2:
            return None

        return Language.from_values(short=arr[0], long=arr[1])

    @staticmethod
    def __from_values_impl(
        short: str,
        long: str,
        *,
        valid_check: bool,
    ) -> Optional["Language"]:
        short_val = ShortLanguageStr.from_str(short)

        if short_val is None:
            return None

        long_val: LongLanguageStr
        if valid_check:
            raise NotImplementedError("TODO")
        else:
            long_val = LongLanguageStr(long)

        return Language(
            short=short_val,
            long=long_val,
            sentinel=Language.__PrivateSentinel(True),
        )

    @staticmethod
    def from_values(
        short: str,
        long: str,
    ) -> Optional["Language"]:
        return Language.__from_values_impl(short=short, long=long, valid_check=True)

    @staticmethod
    def from_str_unsafe(inp: str) -> "Language":
        lan: Optional[Language] = Language.from_str(inp)
        if lan is None:
            msg = _("Couldn't get the Language from str: '{inp}'").format(inp=inp)
            raise RuntimeError(msg)

        return lan

    @staticmethod
    def from_values_unsafe(short: str, long: str) -> "Language":
        lan: Optional[Language] = Language.from_values(
            short,
            long,
        )
        if lan is None:
            msg = _("Couldn't get the Language from values: '{short}' '{long}'").format(
                short=short,
                long=long,
            )
            raise RuntimeError(msg)

        return lan

    # this is for episodes, that have no real language, for some special episodes of some tv series
    @staticmethod
    def no_language() -> "Language":
        return Language(
            ShortLanguageStr.no_lang(),
            LongLanguageStr("No Language"),
            sentinel=Language.__PrivateSentinel(True),
        )

    # TODO: get all languages somehow, from the classifier, that supports them all

    # note this is an implementation detail, that should not leak
    @staticmethod
    def __unknown() -> "Language":
        return Language(
            ShortLanguageStr.unknown_lang(),
            LongLanguageStr("Unknown"),
            sentinel=Language.__PrivateSentinel(True),
        )

    @staticmethod
    def get_default() -> "Language":
        return Language.__unknown()

    @staticmethod
    def is_default_value(language: "Language") -> bool:
        return language == Language.__unknown()

    def __str__(self: Self) -> str:
        return self.__long

    def __repr__(self: Self) -> str:
        return f"<Language short: {self.__short!r} long: {self.__long!r}>"

    def __hash__(self: Self) -> int:
        return hash((self.__short, self.__long))

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Language):
            return self.__short == other.__short and self.__long == other.__long

        return False
