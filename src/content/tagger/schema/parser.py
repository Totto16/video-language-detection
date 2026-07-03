from abc import ABC, abstractmethod
from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
    TypeIs,
    assert_never,
    assert_type,
    override,
)
from xml.etree.ElementTree import XMLParser
from xml.etree.ElementTree import parse as parse_xml

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result
from helper.utils import parse_int_safely


@dataclass(slots=True, repr=True)
class EBMLSchemaRangeNot[A]:
    value: A

    def __str__(self: Self) -> str:
        return f"<EBMLSchemaRangeNot: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, EBMLSchemaRangeNot):
            return self.value == other.value

        return False


class EBMLSchemaRangeBound(Enum):
    Inclusive = "i"
    Exclusive = "e"


@dataclass(slots=True, repr=True)
class EBMLSchemaRangeElem[A: (int, float)]:
    value: A
    bound: EBMLSchemaRangeBound

    def check(self: Self, value: A, *, start: bool) -> bool:
        if start:
            if self.bound == EBMLSchemaRangeBound.Inclusive:
                return value >= self.value
            return value > self.value

        if self.bound == EBMLSchemaRangeBound.Inclusive:
            return value <= self.value
        return value < self.value

    def format(self: Self, *, start: bool) -> str:
        if start:
            if self.bound == EBMLSchemaRangeBound.Inclusive:
                return f"[{self.value}"
            return f"({self.value}"

        if self.bound == EBMLSchemaRangeBound.Inclusive:
            return f"{self.value}]"
        return f"{self.value})"


class EBMLSchemaRange[A: (int, float)]:
    __underlying: (
        A
        | EBMLSchemaRangeNot[A]
        | tuple[EBMLSchemaRangeElem[A], EBMLSchemaRangeElem[A]]
        | tuple[EBMLSchemaRangeElem[A], None]
        | tuple[None, EBMLSchemaRangeElem[A]]
    )

    def __init__(
        self: Self,
        value: (
            A
            | EBMLSchemaRangeNot[A]
            | tuple[EBMLSchemaRangeElem[A], EBMLSchemaRangeElem[A]]
            | tuple[EBMLSchemaRangeElem[A], None]
            | tuple[None, EBMLSchemaRangeElem[A]]
        ),
    ) -> None:
        self.__underlying = value

    def valid(self: Self, value: A) -> Result[None, str]:
        if isinstance(self.__underlying, (int, float)):
            # TODO. check if the value is the same type as A
            if value != self.__underlying:
                return Err(f"Value needs to be {self.__underlying} but was {value}")
        elif isinstance(self.__underlying, tuple):
            min_val, max_val = self.__underlying
            if min_val is not None and not min_val.check(value, start=True):
                return Err(
                    f"Value needs to be between {min_val.format(start=True)}, {max_val.format(start=False) if max_val is not None else "*)"} but was below it: {value}",
                )

            if max_val is not None and not max_val.check(value, start=False):
                return Err(
                    f"Value needs to be between {min_val.format(start=True)if min_val is not None else "(*"}, {max_val.format(start=False)} but was above it: {value}",
                )
        elif isinstance(self.__underlying, EBMLSchemaRangeNot):
            if value == self.__underlying:
                return Err(
                    f"Value needs to be anything but {self.__underlying} but was {value}",
                )
        else:
            assert_never(self.__underlying)

        return Ok(None)

    def __str__(self: Self) -> str:
        if isinstance(self.__underlying, (int, float)):
            return f"<EBMLSchemaRange exact value: {self.__underlying}>"

        if isinstance(self.__underlying, tuple):
            min_inclusive, max_exclusive = self.__underlying
            return f"<EBMLSchemaRange range: [{"*" if min_inclusive is None else min_inclusive}, {"*" if max_exclusive is None else max_exclusive}]>"

        if isinstance(self.__underlying, EBMLSchemaRangeNot):
            return f"<EBMLSchemaRange not value: {self.__underlying.value}>"

        assert_never(self.__underlying)

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(self.__underlying)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, EBMLSchemaRange):
            return self.__underlying == other.__underlying

        if isinstance(self.__underlying, (int, float)):
            if isinstance(other, (int, float)):
                return self.__underlying == other
            return False

        if isinstance(self.__underlying, tuple):
            if isinstance(other, tuple) and len(other) == 2:
                return self.__underlying == other
            return False
        if isinstance(self.__underlying, EBMLSchemaRangeNot):
            if isinstance(other, EBMLSchemaRangeNot):
                return self.__underlying == other
            return False

        assert_never(self.__underlying)


type EBMLOccurrences = EBMLSchemaRange[int]


def ebml_occurrences_from_values(
    min_occurrences: int,
    max_occurrences: Optional[int],
) -> EBMLOccurrences:
    if min_occurrences < 0:
        msg = f"min occurrences is negative: {min_occurrences}"
        raise RuntimeError(msg)

    if max_occurrences is None:
        return EBMLSchemaRange(
            (
                EBMLSchemaRangeElem(min_occurrences, EBMLSchemaRangeBound.Inclusive),
                None,
            ),
        )

    if max_occurrences < 0:
        msg = f"max occurrences is negative: {max_occurrences}"
        raise RuntimeError(msg)

    if min_occurrences == max_occurrences:
        return EBMLSchemaRange(min_occurrences)

    if min_occurrences > max_occurrences:
        msg = f"min occurrences is greater than max occurrences: {min_occurrences} > {max_occurrences}"
        raise RuntimeError(msg)

    return EBMLSchemaRange(
        (
            EBMLSchemaRangeElem(min_occurrences, EBMLSchemaRangeBound.Inclusive),
            EBMLSchemaRangeElem(max_occurrences, EBMLSchemaRangeBound.Inclusive),
        ),
    )


type EBMLVersions = EBMLSchemaRange[int]


def ebml_versions_from_values(
    min_ver: int,
    max_ver: Optional[int],
) -> EBMLVersions:
    if min_ver < 0:
        msg = f"min version is negative: {min_ver}"
        raise RuntimeError(msg)

    if max_ver is None:
        return EBMLSchemaRange(
            (
                EBMLSchemaRangeElem(min_ver, EBMLSchemaRangeBound.Inclusive),
                None,
            ),
        )

    if max_ver < 0:
        msg = f"max version is negative: {max_ver}"
        raise RuntimeError(msg)

    if min_ver == max_ver:
        return EBMLSchemaRange(min_ver)

    if min_ver > max_ver:
        msg = f"min version is greater than max version: {min_ver} > {max_ver}"
        raise RuntimeError(msg)

    return EBMLSchemaRange(
        (
            EBMLSchemaRangeElem(min_ver, EBMLSchemaRangeBound.Inclusive),
            EBMLSchemaRangeElem(max_ver, EBMLSchemaRangeBound.Inclusive),
        ),
    )


class EBMLElementType(Enum):
    SignedInteger = "si"
    UnsignedInteger = "ui"
    Float = "f"
    String = "s"
    UTF8 = "utf-8"
    Date = "d"
    Master = "m"
    Binary = "b"

    @staticmethod
    def from_str(inp: str) -> "EBMLElementType":
        match inp:
            case "integer":
                return EBMLElementType.SignedInteger
            case "uinteger":
                return EBMLElementType.UnsignedInteger
            case "float":
                return EBMLElementType.Float
            case "string":
                return EBMLElementType.String
            case "date":
                return EBMLElementType.Date
            case "utf-8":
                return EBMLElementType.UTF8
            case "master":
                return EBMLElementType.Master
            case "binary":
                return EBMLElementType.Binary
            case _:
                msg = f"Invalid EBMLElementType: {inp}"
                raise RuntimeError(msg)


@dataclass(slots=True, repr=True)
class DefaultRequired:
    pass


@dataclass(slots=True, repr=True)
class DefaultEmpty:
    pass


DefaultOptions = DefaultRequired | DefaultEmpty


@decorate_class(slots=True)
class EBMLAdvancedElementTypeAbstract[Type](ABC):
    @abstractmethod
    def validate(self: Self, value: Type) -> Result[None, str]: ...


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeInteger(EBMLAdvancedElementTypeAbstract[int]):
    type: Literal[EBMLElementType.SignedInteger, EBMLElementType.UnsignedInteger]
    default: int | DefaultOptions
    range: Optional[EBMLSchemaRange[int]]

    @override
    def validate(self: Self, value: int) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        range_valid = self.range.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeFloat(EBMLAdvancedElementTypeAbstract[float]):
    type: Literal[EBMLElementType.Float]
    default: float | DefaultOptions
    range: Optional[EBMLSchemaRange[float]]

    @override
    def validate(self: Self, value: float) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        range_valid = self.range.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


@decorate_class(slots=True)
class LengthRange:

    __range: EBMLSchemaRange[int]

    @property
    def range(self: Self) -> EBMLSchemaRange[int]:
        return self.__range

    def __init__(self: Self, value: EBMLSchemaRange[int]) -> None:
        self.__range = value

    def valid(self: Self, value: Sized) -> Result[None, str]:
        length = len(value)

        return self.__range.valid(length)

    def __str__(self: Self) -> str:
        return f"<LengthRange range: {self.__range!s}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(("LengthRange", self.__range))

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, LengthRange):
            return self.__range == other.range

        if isinstance(other, EBMLSchemaRange):
            return self.__range == other
        return False


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeString(EBMLAdvancedElementTypeAbstract[str]):
    type: Literal[EBMLElementType.String, EBMLElementType.UTF8]
    default: str | DefaultOptions
    length: Optional[LengthRange]

    @override
    def validate(self: Self, value: str) -> Result[None, str]:
        if self.length is None:
            return Ok(None)

        range_valid = self.length.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeDate(EBMLAdvancedElementTypeAbstract[datetime]):
    type: Literal[EBMLElementType.Date]
    default: datetime | DefaultOptions

    @override
    def validate(self: Self, value: datetime) -> Result[None, str]:
        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeMaster:
    type: Literal[EBMLElementType.Master]


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeBinary(EBMLAdvancedElementTypeAbstract[bytes]):
    type: Literal[EBMLElementType.Binary]
    default: bytes | DefaultOptions
    length: Optional[LengthRange]

    @override
    def validate(self: Self, value: bytes) -> Result[None, str]:
        if self.length is None:
            return Ok(None)

        range_valid = self.length.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


EBMLAdvancedElementType = (
    EBMLAdvancedElementTypeInteger
    | EBMLAdvancedElementTypeFloat
    | EBMLAdvancedElementTypeString
    | EBMLAdvancedElementTypeDate
    | EBMLAdvancedElementTypeMaster
    | EBMLAdvancedElementTypeBinary
)


@dataclass(slots=True, repr=True)
class EBMLElementIDParsed:
    value: int

    @staticmethod
    def from_checked(value: int) -> "EBMLElementIDParsed":
        # check if this would be a valid ID, alias it can be encoded as varint

        length = max(1, (value.bit_length() + 7) // 8)
        encoded_bytes = value.to_bytes(length=length, byteorder="big", signed=False)

        int_range = LengthRange(
            EBMLSchemaRange(
                (
                    EBMLSchemaRangeElem(1, EBMLSchemaRangeBound.Inclusive),
                    EBMLSchemaRangeElem(4, EBMLSchemaRangeBound.Inclusive),
                ),
            ),
        )

        is_valid = int_range.valid(encoded_bytes)

        if is_valid.err():
            msg = f"Invalid EBMLElementID: length has to be in range {int_range} but was: {len(encoded_bytes)}: {is_valid.as_err()}"
            raise RuntimeError(msg)

        first_byte = encoded_bytes[0]

        match len(encoded_bytes):
            case 1:
                val = first_byte & 0x80
                if val != 0x80:
                    msg = f"Invalid EBMLElementID: length not correctly encoded: {len(encoded_bytes)}: {hex(val)}"
                    raise RuntimeError(msg)
            case 2:
                val = first_byte & 0xC0
                if val != 0x40:
                    msg = f"Invalid EBMLElementID: length not correctly encoded: {len(encoded_bytes)}: {hex(val)}"
                    raise RuntimeError(msg)
            case 3:
                val = first_byte & 0xE0
                if val != 0x20:
                    msg = f"Invalid EBMLElementID: length not correctly encoded: {len(encoded_bytes)}: {hex(val)}"
                    raise RuntimeError(msg)
            case 4:
                val = first_byte & 0xF0
                if val != 0x10:
                    msg = f"Invalid EBMLElementID: length not correctly encoded: {len(encoded_bytes)}: {hex(val)}"
                    raise RuntimeError(msg)
            case _:
                msg = f"UNREACHABLE: LengthRange error: {int_range} {encoded_bytes!r} {value}"

        return EBMLElementIDParsed(value)

    def __str__(self: Self) -> str:
        return f"<EBMLElementIDParsed {hex(self.value)}>"

    def __repr__(self: Self) -> str:
        return repr(self.value)


@dataclass(slots=True, repr=True)
class EBMLElementDescriptionGeneric[A: (EBMLAdvancedElementType)]:
    name: str
    id: EBMLElementIDParsed
    occurrences: EBMLOccurrences
    type: A
    description: Optional[str]
    unknown_size_allowed: bool
    versions: EBMLVersions
    path: str
    recurring: bool
    recursive: bool


type EBMLElementDescription = EBMLElementDescriptionGeneric[EBMLAdvancedElementType]


@dataclass(slots=True, repr=True)
class DocType:
    type: str
    version: int

    def __str__(self: Self) -> str:
        return f"<DocType type: {self.type} version: {self.version}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
class EBMLSpec:
    __elements: list[EBMLElementDescription]
    doc_type: DocType

    def __init__(self: Self, doc_type: DocType) -> None:
        self.__elements = []
        self.doc_type = doc_type

    @property
    def elements(self: Self) -> list[EBMLElementDescription]:
        return self.__elements

    EBMLSpecByName = dict[str, EBMLElementDescription]

    def elements_by_name(self: Self) -> EBMLSpecByName:
        return {element.name: element for element in self.__elements}

    EBMLSpecById = dict[int, EBMLElementDescription]

    def elements_by_id(self: Self) -> EBMLSpecById:
        return {element.id.value: element for element in self.__elements}

    def append(self: Self, element: EBMLElementDescription) -> None:
        id_value = self.elements_by_id().get(element.id.value, None)
        if id_value is not None:
            msg = f"Duplicate ID: {id_value}: {element}"
            raise RuntimeError(msg)

        name_value = self.elements_by_name().get(element.name, None)
        if name_value is not None:
            msg = f"Duplicate name: {name_value}: {element}"
            raise RuntimeError(msg)

        self.__elements.append(element)

    def __str__(self: Self) -> str:
        return f"<EBMLSpec doc_type: {self.doc_type}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
class _MISSING:
    pass


def is_missing[A](value: A | type[_MISSING]) -> TypeIs[type[_MISSING]]:
    return isinstance(value, _MISSING)


def xml_required[A](dct: dict[str, A], key: str) -> A:
    value = dct.get(key, _MISSING)

    if is_missing(value):
        msg = f"Missing XML Require attribute '{key}': {dct}"
        raise TypeError(msg)

    assert_type(value, A)

    return value


def xml_int(value: str | int, base: int = 10) -> int:
    if isinstance(value, int):
        return value

    result = parse_int_safely(value, base)

    if result is None:
        msg = f"Invalid number: {value}"
        raise RuntimeError(msg)

    return result


def parse_bool_safely(inp: str) -> Optional[bool]:
    match inp:
        case "1":
            return True
        case "0":
            return False
        case "true":
            return True
        case "false":
            return False
        case _:
            return None


def xml_boolean(value: str | bool) -> bool:  # noqa: FBT001
    if isinstance(value, bool):
        return value

    result = parse_bool_safely(value)

    if result is None:
        msg = f"Invalid boolean: {value}"
        raise RuntimeError(msg)

    return result


def xml_int_optional(value: Optional[str | int], base: int = 10) -> Optional[int]:
    if value is None:
        return None

    return xml_int(value, base)


class RangeWrapper[A](ABC):

    @abstractmethod
    def parse(self: Self, value: str) -> Optional[A]: ...


def xml_any_range_result[A: (int, float)](  # noqa: PLR0915
    value: str,
    generic: RangeWrapper[A],
) -> Result[EBMLSchemaRange[A], str]:
    # spec: RFC 8794
    # chapter 11.1.6.6.1

    val = value.replace(" ", "")

    raw_num = generic.parse(val)

    if raw_num is not None:
        # Case 1: just a number
        return Ok(EBMLSchemaRange(raw_num))

    if val.startswith("not"):
        # Case 2: not <number>
        normal_val = val[len("not") :]
        raw_num = generic.parse(normal_val)

        if raw_num is None:
            return Err(f"Invalid number after not: '{normal_val}'")

        return Ok(EBMLSchemaRange(EBMLSchemaRangeNot(raw_num)))

    class Bound(Enum):
        LT = "<"
        LE = "<="
        GT = ">"
        GE = ">="

    def parse_bound(inp: str) -> Result[tuple[Bound, A], str]:
        def parse_bound_impl(
            bound: Bound,
            raw_inp: str,
        ) -> Result[tuple[Bound, A], str]:
            num_v = generic.parse(raw_inp)

            if num_v is None:
                return Err(f"Invalid bound number: '{raw_inp}'")

            return Ok((bound, num_v))

        if inp.startswith("<="):
            return parse_bound_impl(Bound.LE, inp[len("<=") :])
        if inp.startswith("<"):
            return parse_bound_impl(Bound.LT, inp[len("<") :])
        if inp.startswith(">="):
            return parse_bound_impl(Bound.GE, inp[len(">=") :])
        if inp.startswith(">"):
            return parse_bound_impl(Bound.GT, inp[len(">") :])

        return Err(f"Invalid bounded number: '{inp}'")

    if "," in val:
        # Case 3: new syntax <bound_num>, <bound_num>
        num1, num2 = val.split(",", 1)
        if "," in num2:
            return Err(f"Invalid syntax, only one ',' allowed: '{val}'")

        bound1_v = parse_bound(num1)

        if bound1_v.err():
            return Err(f"Invalid starting bound: {bound1_v.as_err()}")

        bound2_v = parse_bound(num2)

        if bound2_v.err():
            return Err(f"Invalid ending bound: {bound2_v.as_err()}")

        bound1_b, bound1_num = bound1_v.as_ok()

        new_range_start: EBMLSchemaRangeElem[A]

        match bound1_b:
            case Bound.GE:
                new_range_start = EBMLSchemaRangeElem(
                    bound1_num,
                    EBMLSchemaRangeBound.Inclusive,
                )
            case Bound.GT:
                new_range_start = EBMLSchemaRangeElem(
                    bound1_num,
                    EBMLSchemaRangeBound.Exclusive,
                )
            case _:
                return Err(
                    f"First boundary has to be the lower boundary: but was: {bound1_b.name}: '{val}'",
                )

        bound2_b, bound2_num = bound2_v.as_ok()

        new_range_end: EBMLSchemaRangeElem[A]

        match bound2_b:
            case Bound.LE:
                new_range_end = EBMLSchemaRangeElem(
                    bound2_num,
                    EBMLSchemaRangeBound.Inclusive,
                )
            case Bound.LT:
                new_range_end = EBMLSchemaRangeElem(
                    bound2_num,
                    EBMLSchemaRangeBound.Exclusive,
                )
            case _:
                return Err(
                    f"Second boundary has to be the upper boundary: but was: {bound2_b.name}: '{val}'",
                )

        new_range: tuple[EBMLSchemaRangeElem[A], EBMLSchemaRangeElem[A]] = (
            new_range_start,
            new_range_end,
        )
        return Ok(EBMLSchemaRange(new_range))

    if "-" in val:
        # Case 4: old syntax <num>-<num>
        num1, num2 = val.split("-", 1)
        if "-" in num2:
            return Err(f"Invalid syntax, only one '-' allowed: '{val}'")

        num1_v = generic.parse(num1)

        if num1_v is None:
            return Err(f"Invalid starting number: '{num1}'")

        num2_v = generic.parse(num2)

        if num2_v is None:
            return Err(f"Invalid ending number: '{num2}'")

        if num1_v > num2_v:
            return Err(
                f"Invalid range order: first number is bigger: {num1_v} > {num2_v}",
            )

        old_range: tuple[EBMLSchemaRangeElem[A], EBMLSchemaRangeElem[A]] = (
            EBMLSchemaRangeElem(num1_v, EBMLSchemaRangeBound.Inclusive),
            EBMLSchemaRangeElem(num2_v, EBMLSchemaRangeBound.Inclusive),
        )
        return Ok(EBMLSchemaRange(old_range))

    # Case 5: <bound_num>
    bound_v = parse_bound(val)

    if bound_v.err():
        return Err(f"Invalid bound: {bound_v.as_err()}")

    bound_b, bound_num = bound_v.as_ok()

    match bound_b:
        case Bound.LE:
            return Ok(
                EBMLSchemaRange(
                    (
                        None,
                        EBMLSchemaRangeElem(bound_num, EBMLSchemaRangeBound.Inclusive),
                    ),
                ),
            )
        case Bound.LT:
            return Ok(
                EBMLSchemaRange(
                    (
                        None,
                        EBMLSchemaRangeElem(bound_num, EBMLSchemaRangeBound.Exclusive),
                    ),
                ),
            )
        case Bound.GE:
            return Ok(
                EBMLSchemaRange(
                    (
                        EBMLSchemaRangeElem(bound_num, EBMLSchemaRangeBound.Inclusive),
                        None,
                    ),
                ),
            )
        case Bound.GT:
            return Ok(
                EBMLSchemaRange(
                    (
                        EBMLSchemaRangeElem(bound_num, EBMLSchemaRangeBound.Exclusive),
                        None,
                    ),
                ),
            )
        case _:
            assert_never(bound_b)


def xml_any_range_optional[A: (int, float)](
    value: Optional[str],
    generic: RangeWrapper[A],
) -> Optional[EBMLSchemaRange[A]]:
    if value is None:
        return None

    result = xml_any_range_result(value, generic)

    if result.err():
        msg = f"Invalid range: {result.as_err()}"
        raise RuntimeError(msg)

    return result.as_ok()


class WrapperInt(RangeWrapper[int]):

    @override
    def parse(self: Self, value: str) -> Optional[int]:
        return parse_int_safely(value)


def xml_int_range_optional(value: Optional[str]) -> Optional[EBMLSchemaRange[int]]:
    return xml_any_range_optional(value, WrapperInt())


def xml_parse_float_safely(inp: str) -> Optional[float]:

    # spec: RFC 8794
    # chapter 11.1.18

    # When a ﬂoat value is represented textually in an EBML Schema, such as within a default or
    # range value, the ﬂoat values MUST be expressed as Hexadecimal Floating-Point Constants as
    # deﬁned in the C11 standard [ISO9899] (see Section 6.4.4.2 on Floating Constants). Table 9
    # provides examples of expressions of ﬂoat ranges.

    try:
        return float.fromhex(inp)
    except ValueError:
        return None


class WrapperFloat(RangeWrapper[float]):

    @override
    def parse(self: Self, value: str) -> Optional[float]:
        return xml_parse_float_safely(value)


def xml_float_range_optional(value: Optional[str]) -> Optional[EBMLSchemaRange[float]]:
    return xml_any_range_optional(value, WrapperFloat())


def xml_length_range_optional(value: Optional[str]) -> Optional[LengthRange]:
    result = xml_int_range_optional(value)

    if result is None:
        return None

    return LengthRange(result)


def xml_float(value: str | float) -> float:
    if isinstance(value, float):
        return value

    result = xml_parse_float_safely(value)

    if result is None:
        msg = f"Invalid number: {value}"
        raise RuntimeError(msg)

    return result


def xml_float_optional(value: Optional[str | float]) -> Optional[float]:
    if value is None:
        return None

    return xml_float(value)


def xml_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value

    raise NotImplementedError("TODO")


def xml_datetime_optional(value: Optional[str | datetime]) -> Optional[datetime]:
    if value is None:
        return None

    return xml_datetime(value)


def xml_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value

    return value.encode()


def xml_bytes_optional(value: Optional[str | bytes]) -> Optional[bytes]:
    if value is None:
        return None

    return xml_bytes(value)


def xml_default_value[A](value: Optional[A]) -> A | DefaultOptions:
    if value is None:
        return DefaultEmpty()

    return value


def ebml_read_spec_xml(name: str) -> EBMLSpec:  # noqa: PLR0915

    file = Path(__file__).parent / name

    if not file.exists():
        msg = f"Spec XMl file '{file}' doesn't exist"
        raise RuntimeError(msg)

    xml_content = parse_xml(file, XMLParser())  # noqa: S314

    root = xml_content.getroot()

    def xml_local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    root_tag = xml_local_name(root.tag)

    if root_tag != "EBMLSchema":
        msg = f"Invalid root xml tag: {root_tag}: {root!s}"
        raise RuntimeError(msg)

    version = xml_int(xml_required(root.attrib, "version"))

    doc_type_str = xml_required(root.attrib, "docType")

    doc_type = DocType(type=doc_type_str, version=version)

    result: EBMLSpec = EBMLSpec(doc_type=doc_type)

    for element_entry in root:

        element_entry_tag = xml_local_name(element_entry.tag)

        if element_entry_tag != "element":
            msg = f"Invalid element xml tag: {element_entry_tag}: {element_entry!s}"
            raise RuntimeError(msg)

        allowed_attributes: set[str] = set()

        name = xml_required(element_entry.attrib, "name")
        path = xml_required(element_entry.attrib, "path")
        element_id = xml_int(xml_required(element_entry.attrib, "id"), 16)
        min_occurs = xml_int(element_entry.attrib.get("minOccurs", 0))
        max_occurs = xml_int_optional(element_entry.attrib.get("maxOccurs", None))
        element_type = EBMLElementType.from_str(
            xml_required(element_entry.attrib, "type"),
        )

        unknown_size_allowed = xml_boolean(
            element_entry.attrib.get("unknownsizeallowed", False),
        )

        recurring = xml_boolean(
            element_entry.attrib.get("recurring", False),
        )

        min_ver = xml_int(element_entry.attrib.get("minver", 1))
        max_ver = xml_int_optional(element_entry.attrib.get("maxver", None))

        recursive = xml_boolean(
            element_entry.attrib.get("recursive", False),
        )
        allowed_attributes.update(
            [
                "name",
                "path",
                "id",
                "minOccurs",
                "maxOccurs",
                "type",
                "unknownsizeallowed",
                "recurring",
                "minver",
                "maxver",
                "recursive",
            ],
        )

        advanced_type: EBMLAdvancedElementType
        match element_type:
            case EBMLElementType.SignedInteger | EBMLElementType.UnsignedInteger:
                default_int: int | DefaultOptions = xml_default_value(
                    xml_int_optional(element_entry.attrib.get("default", None)),
                )
                range_int = xml_int_range_optional(
                    element_entry.attrib.get("range", None),
                )

                allowed_attributes.update(["default", "range"])
                advanced_type = EBMLAdvancedElementTypeInteger(
                    type=element_type,
                    default=default_int,
                    range=range_int,
                )
            case EBMLElementType.Float:
                default_float: float | DefaultOptions = xml_default_value(
                    xml_float_optional(element_entry.attrib.get("default", None)),
                )
                range_float = xml_float_range_optional(
                    element_entry.attrib.get("range", None),
                )

                allowed_attributes.update(["default", "range"])
                advanced_type = EBMLAdvancedElementTypeFloat(
                    type=element_type,
                    default=default_float,
                    range=range_float,
                )
            case EBMLElementType.String | EBMLElementType.UTF8:
                default_str: str | DefaultOptions = xml_default_value(
                    element_entry.attrib.get("default", None),
                )
                length = xml_length_range_optional(
                    element_entry.attrib.get("length", None),
                )

                allowed_attributes.update(["default", "length"])
                advanced_type = EBMLAdvancedElementTypeString(
                    type=element_type,
                    default=default_str,
                    length=length,
                )
            case EBMLElementType.Date:
                default_datetime: datetime | DefaultOptions = xml_default_value(
                    xml_datetime_optional(
                        element_entry.attrib.get("default", None),
                    ),
                )

                allowed_attributes.update(["default"])
                advanced_type = EBMLAdvancedElementTypeDate(
                    type=element_type,
                    default=default_datetime,
                )
            case EBMLElementType.Master:
                advanced_type = EBMLAdvancedElementTypeMaster(
                    type=element_type,
                )
            case EBMLElementType.Binary:
                default_binary: bytes | DefaultOptions = xml_default_value(
                    xml_bytes_optional(
                        element_entry.attrib.get("default", None),
                    ),
                )
                length = xml_length_range_optional(
                    element_entry.attrib.get("length", None),
                )

                allowed_attributes.update(["default", "length"])
                advanced_type = EBMLAdvancedElementTypeBinary(
                    type=element_type,
                    default=default_binary,
                    length=length,
                )
            case _:
                assert_never(element_type)

        description_elem = element_entry.find("documentation")
        description = None if description_elem is None else description_elem.text

        not_processed_attributes = set(element_entry.attrib.keys()).difference(
            allowed_attributes,
        )

        if len(not_processed_attributes) != 0:
            msg = f"Encountered some not allowed attributes: {not_processed_attributes}"
            raise RuntimeError(msg)

        occurrences = ebml_occurrences_from_values(min_occurs, max_occurs)

        versions = ebml_versions_from_values(min_ver, max_ver)

        element: EBMLElementDescription = EBMLElementDescriptionGeneric(
            name=name,
            id=EBMLElementIDParsed.from_checked(element_id),
            occurrences=occurrences,
            type=advanced_type,
            description=description,
            unknown_size_allowed=unknown_size_allowed,
            versions=versions,
            path=path,
            recurring=recurring,
            recursive=recursive,
        )

        result.append(element)

    return result


def filter_spec_elements(
    spec: EBMLSpec,
    cb: Callable[[EBMLElementDescription], bool],
) -> EBMLSpec:

    result = EBMLSpec(doc_type=spec.doc_type)
    for element in spec.elements:

        should_include = cb(element)

        if should_include:
            result.append(element)

    return result
