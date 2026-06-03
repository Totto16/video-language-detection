from enum import StrEnum
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NewType, Optional, Self

from annotated_types import GroupedMetadata, Len, Predicate
from apischema import (
    deserializer,
    schema,
    serialize,
    serializer,
    type_name,
)

from content.iso_codes import valid_iso_languages_list
from helper.apischema import OneOf, use_schema_from
from helper.translation import get_translator

__all__: list[str] = ["ExactLen", "Language"]


_ = get_translator()


def ExactLen(length: int) -> Len:  # noqa: N802
    return Len(min_length=length, max_length=length)


# see https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
# and: https://www.loc.gov/standards/iso639-2/php/code_list.php


@dataclass
class AlphaLanguageCodeAnnnotation(GroupedMetadata):
    length: int

    def __iter__(self) -> Iterator[object]:
        yield Predicate(str.islower)

        yield ExactLen(self.length)


ALPHA_3_LANGUAGE_STR_PATTERN = r"^([a-z]{3})$"


# this should be ISO 639-2 codes (alpha-3 code)
@schema(pattern=ALPHA_3_LANGUAGE_STR_PATTERN)
class Alpha3LanguageStr:
    __PrivateStrImpl = NewType(
        "__PrivateStrImpl",
        Annotated[str, AlphaLanguageCodeAnnnotation(3)],
    )

    __data: __PrivateStrImpl

    # this is used, so that init is only callable from the internal class, so that it only gets checked values!
    __PrivateSentinel = NewType("__PrivateSentinel", bool)

    def __init__(
        self: Self,
        data: __PrivateStrImpl,
        *,
        sentinel: __PrivateSentinel,  # noqa: ARG002
    ) -> None:
        self.__data = data

    @staticmethod
    def __from_raw_str_checked(
        val: str,
        *,
        valid_check: bool,
    ) -> Optional[__PrivateStrImpl]:
        if len(val) != 3:
            return None

        if not val.islower():
            return None

        if valid_check:
            allowed_short_names = [entry[0] for entry in valid_iso_languages_list]
            if val not in allowed_short_names:
                msg = _(
                    "Short Language string is invalid according to ISO (3 alpha): '{short}'"  # noqa: COM812
                ).format(short=val)
                raise RuntimeError(msg)

        return Alpha3LanguageStr.__PrivateStrImpl(val)

    @staticmethod
    def __from_str_impl(
        inp: str,
        *,
        valid_check: bool,
    ) -> Optional["Alpha3LanguageStr"]:
        val = Alpha3LanguageStr.__from_raw_str_checked(inp, valid_check=valid_check)

        if val is None:
            return None

        return Alpha3LanguageStr(
            data=val,
            sentinel=Alpha3LanguageStr.__PrivateSentinel(True),  # noqa: FBT003
        )

    @staticmethod
    def from_str(inp: str) -> Optional["Alpha3LanguageStr"]:
        return Alpha3LanguageStr.__from_str_impl(inp, valid_check=True)

    @staticmethod
    def __impl_from_str_unsafe(inp: str, *, valid_check: bool) -> "Alpha3LanguageStr":
        val: Optional[Alpha3LanguageStr] = Alpha3LanguageStr.__from_str_impl(
            inp,
            valid_check=valid_check,
        )
        if val is None:
            msg = _("Couldn't get the Short Language String from str '{inp}'").format(
                inp=inp,
            )
            raise RuntimeError(msg)

        return val

    @staticmethod
    def from_str_unsafe(inp: str) -> "Alpha3LanguageStr":
        return Alpha3LanguageStr.__impl_from_str_unsafe(inp, valid_check=True)

    @serializer
    def serialize(self: Self) -> str:
        return self.__data

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "Alpha3LanguageStr":
        match = re.match(ALPHA_3_LANGUAGE_STR_PATTERN, inp)

        if match is None:
            msg = _(
                "Invalid pattern for Alpha3LanguageStr: got '{inp}', this didn't match the pattern '{pattern}'"  # noqa: COM812
            ).format(inp=inp, pattern=ALPHA_3_LANGUAGE_STR_PATTERN)
            raise TypeError(msg)

        return Alpha3LanguageStr.from_str_unsafe(inp)

    def __str__(self: Self) -> str:
        return self.__data

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(self.__data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Alpha3LanguageStr):
            return self.__data == other.__data

        if isinstance(other, str):
            return self.__data == Alpha3LanguageStr.from_str(other)

        return False


ALPHA_2_LANGUAGE_STR_PATTERN = r"^([a-z]{2})$"


# this should be ISO 639-1 codes (alpha-2 code)
@schema(pattern=ALPHA_2_LANGUAGE_STR_PATTERN)
class Alpha2LanguageStr:
    __PrivateStrImpl = NewType(
        "__PrivateStrImpl",
        Annotated[str, AlphaLanguageCodeAnnnotation(2)],
    )

    __data: __PrivateStrImpl

    # this is used, so that init is only callable from the internal class, so that it only gets checked values!
    __PrivateSentinel = NewType("__PrivateSentinel", bool)

    def __init__(
        self: Self,
        data: __PrivateStrImpl,
        *,
        sentinel: __PrivateSentinel,  # noqa: ARG002
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
            allowed_short_names = [entry[1] for entry in valid_iso_languages_list]
            if val not in allowed_short_names:
                msg = _(
                    "Short Language string is invalid according to ISO (2 alpha): '{short}'"  # noqa: COM812
                ).format(short=val)
                raise RuntimeError(msg)

        return Alpha2LanguageStr.__PrivateStrImpl(val)

    @staticmethod
    def __from_str_impl(
        inp: str,
        *,
        valid_check: bool,
    ) -> Optional["Alpha2LanguageStr"]:
        val = Alpha2LanguageStr.__from_raw_str_checked(inp, valid_check=valid_check)

        if val is None:
            return None

        return Alpha2LanguageStr(
            data=val,
            sentinel=Alpha2LanguageStr.__PrivateSentinel(True),  # noqa: FBT003
        )

    @staticmethod
    def from_str(inp: str) -> Optional["Alpha2LanguageStr"]:
        return Alpha2LanguageStr.__from_str_impl(inp, valid_check=True)

    @staticmethod
    def __impl_from_str_unsafe(inp: str, *, valid_check: bool) -> "Alpha2LanguageStr":
        val: Optional[Alpha2LanguageStr] = Alpha2LanguageStr.__from_str_impl(
            inp,
            valid_check=valid_check,
        )
        if val is None:
            msg = _("Couldn't get the Short Language String from str '{inp}'").format(
                inp=inp,
            )
            raise RuntimeError(msg)

        return val

    @staticmethod
    def from_str_unsafe(inp: str) -> "Alpha2LanguageStr":
        return Alpha2LanguageStr.__impl_from_str_unsafe(inp, valid_check=True)

    @serializer
    def serialize(self: Self) -> str:
        return self.__data

    @staticmethod
    def no_lang() -> "Alpha2LanguageStr":
        return Alpha2LanguageStr.__impl_from_str_unsafe("xx", valid_check=False)

    @staticmethod
    def unknown_lang() -> "Alpha2LanguageStr":
        return Alpha2LanguageStr.__impl_from_str_unsafe("un", valid_check=False)

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "Alpha2LanguageStr":
        # backwards compatible, as before the enforcing of the two alpha rule, no language wasn't two alpha digits!
        if inp in ["no_lang", "xx"]:
            return Alpha2LanguageStr.no_lang()

        match = re.match(ALPHA_2_LANGUAGE_STR_PATTERN, inp)

        if match is None:
            msg = _(
                "Invalid pattern for Alpha2LanguageStr: got '{inp}', this didn't match the pattern '{pattern}'"  # noqa: COM812
            ).format(inp=inp, pattern=ALPHA_2_LANGUAGE_STR_PATTERN)
            raise TypeError(msg)

        # explicit check for this

        if inp == "un":
            return Alpha2LanguageStr.unknown_lang()

        # some known errors and older codes form models, kept for backwards compatibility of older save data
        mappings: dict[str, str] = {"iw": "he", "jw": "jv"}
        if inp in mappings:
            return Alpha2LanguageStr.from_str_unsafe(mappings[inp])

        return Alpha2LanguageStr.from_str_unsafe(inp)

    def __str__(self: Self) -> str:
        return self.__data

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(self.__data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Alpha2LanguageStr):
            return self.__data == other.__data

        if isinstance(other, str):
            return self.__data == Alpha2LanguageStr.from_str(other)

        return False


LongLanguageStr = NewType("LongLanguageStr", str)


class NoLangDeprecatedType(StrEnum):
    no_lang = "no_lang"


@schema()
@type_name("LanguageImpl")
@dataclass
class LanguageSchema:
    short: Annotated[
        Alpha2LanguageStr | Alpha3LanguageStr | NoLangDeprecatedType,
        OneOf,
    ]
    long: LongLanguageStr


@use_schema_from(LanguageSchema)
class Language:
    __short: Alpha2LanguageStr | Alpha3LanguageStr
    __long: LongLanguageStr

    # this is used, so that init is only callable from the internal class, so that it only gets checked values!
    __PrivateSentinel = NewType("__PrivateSentinel", bool)

    def __init__(
        self: Self,
        short: Alpha2LanguageStr | Alpha3LanguageStr,
        long: LongLanguageStr,
        *,
        sentinel: Optional[__PrivateSentinel] = None,
    ) -> None:
        if sentinel is None:
            # NOTE: this is for apischema deserialization checks!
            myself = self.deserialize(LanguageSchema(short, long))
            self.__short = myself.__short  # noqa: SLF001
            self.__long = myself.__long  # noqa: SLF001
            return

        self.__short = short
        self.__long = long

    @property
    def short(self: Self) -> Alpha2LanguageStr | Alpha3LanguageStr:
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
    def __short_from_str(val: str) -> Optional[Alpha2LanguageStr | Alpha3LanguageStr]:
        if len(val) == 2:
            return Alpha2LanguageStr.from_str(val)

        if len(val) == 3:
            return Alpha3LanguageStr.from_str(val)

        return None

    @staticmethod
    def __from_values_impl(
        short: str,
        long: str,
        *,
        valid_check: bool,
    ) -> Optional["Language"]:
        short_val: Optional[Alpha2LanguageStr | Alpha3LanguageStr] = (
            Language.__short_from_str(short)
        )

        if short_val is None:
            return None

        if valid_check:
            allowed_long_names = [entry[2] for entry in valid_iso_languages_list]
            if long not in allowed_long_names:
                msg = _(
                    "Long Language string is invalid according to ISO: '{long}'"  # noqa: COM812
                ).format(long=long)
                raise RuntimeError(msg)

        return Language(
            short=short_val,
            long=LongLanguageStr(long),
            sentinel=Language.__PrivateSentinel(True),  # noqa: FBT003
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
            Alpha2LanguageStr.no_lang(),
            LongLanguageStr("No Language"),
            sentinel=Language.__PrivateSentinel(True),  # noqa: FBT003
        )

    # note this is an implementation detail, that should not leak
    @staticmethod
    def __unknown() -> "Language":
        return Language(
            Alpha2LanguageStr.unknown_lang(),
            LongLanguageStr("Unknown"),
            sentinel=Language.__PrivateSentinel(True),  # noqa: FBT003
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

    @serializer
    def serialize(self: Self) -> dict[str, Any]:
        serialized_dict: dict[str, Any] = serialize(
            LanguageSchema,
            LanguageSchema(short=self.__short, long=self.__long),
        )
        return serialized_dict

    # @deserializer
    @staticmethod
    def deserialize(language: LanguageSchema) -> "Language":
        if (
            language.short == "no_lang"  # noqa: PLR1714
            or language.short == NoLangDeprecatedType.no_lang
        ):
            return Language.no_language()

        return Language(
            language.short,
            language.long,
            sentinel=Language.__PrivateSentinel(True),  # noqa: FBT003
        )
