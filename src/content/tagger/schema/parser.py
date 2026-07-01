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
from helper.utils import parse_float_safely, parse_int_safely

type EBMLOccurrences = int | tuple[int, int] | tuple[int, None]


def ebml_occurrences_from_values(
    min_occurrences: int,
    max_occurrences: Optional[int],
) -> EBMLOccurrences:
    if min_occurrences < 0:
        msg = f"min occurrences is negative: {min_occurrences}"
        raise RuntimeError(msg)

    if max_occurrences is None:
        return (min_occurrences, None)

    if max_occurrences < 0:
        msg = f"max occurrences is negative: {max_occurrences}"
        raise RuntimeError(msg)

    if min_occurrences == max_occurrences:
        return min_occurrences

    if min_occurrences > max_occurrences:
        msg = f"min occurrences is greater than max occurrences: {min_occurrences} > {max_occurrences}"
        raise RuntimeError(msg)

    return (min_occurrences, max_occurrences)


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


@decorate_class(slots=True)
class DefaultRequired:
    pass


@decorate_class(slots=True)
class DefaultEmpty:
    pass


DefaultOptions = DefaultRequired | DefaultEmpty


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


class EBMLSchemaRange[A: (int, float)]:
    # range is start inclusive, end exclusive
    __underlying: (
        A | EBMLSchemaRangeNot[A] | tuple[A, A] | tuple[A, None] | tuple[None, A]
    )

    def __init__(
        self: Self,
        value: (
            A | EBMLSchemaRangeNot[A] | tuple[A, A] | tuple[A, None] | tuple[None, A]
        ),
    ) -> None:
        self.__underlying = value

    def valid(self: Self, value: A) -> Result[None, str]:
        if isinstance(self.__underlying, (int, float)):
            # TODO. check if the value is the same type as A
            if value != self.__underlying:
                return Err(f"Value needs to be {self.__underlying} but was {value}")
        elif isinstance(self.__underlying, tuple):
            min_inclusive, max_exclusive = self.__underlying
            if min_inclusive is not None and value < min_inclusive:
                return Err(
                    f"Value needs to be between [{min_inclusive}, {max_exclusive}] but was below it: {value}",
                )

            if max_exclusive is not None and value >= max_exclusive:
                return Err(
                    f"Value needs to be between [{min_inclusive}, {max_exclusive}] but was above it: {value}",
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


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeInteger:
    type: Literal[EBMLElementType.SignedInteger, EBMLElementType.UnsignedInteger]
    default: int | DefaultOptions
    range: Optional[EBMLSchemaRange[int]]

    def validate(self: Self, value: int) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        range_valid = self.range.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeFloat:
    type: Literal[EBMLElementType.Float]
    default: float | DefaultOptions
    range: Optional[EBMLSchemaRange[float]]

    def validate(self: Self, value: float) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        range_valid = self.range.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


class LengthRange:

    __range: EBMLSchemaRange[int]

    def __init__(self: Self, value: EBMLSchemaRange[int]) -> None:
        self.__range = value

    def valid(self: Self, value: Sized) -> Result[None, str]:
        length = len(value)

        return self.__range.valid(length)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeString:
    type: Literal[EBMLElementType.String, EBMLElementType.UTF8]
    default: str | DefaultOptions
    length: Optional[LengthRange]

    def validate(self: Self, value: str) -> Result[None, str]:
        if self.length is None:
            return Ok(None)

        range_valid = self.length.valid(value)

        if range_valid.err():
            return Err(range_valid.as_err())

        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeDate:
    type: Literal[EBMLElementType.Date]
    default: datetime | DefaultOptions

    def validate(self: Self, value: datetime) -> Result[None, str]:
        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeMaster:
    type: Literal[EBMLElementType.Master]


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeBinary:
    type: Literal[EBMLElementType.Binary]
    default: bytes | DefaultOptions
    length: Optional[LengthRange]

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
class EBMLElementDescription:
    name: str
    id: int
    occurrences: EBMLOccurrences
    type: EBMLAdvancedElementType
    description: Optional[str]


@dataclass(slots=True, repr=True)
class EBMLSpec:
    elements: list[EBMLElementDescription]
    version: int

    EBMLSpecByName = dict[str, EBMLElementDescription]

    def elements_by_name(self: Self) -> EBMLSpecByName:
        return {element.name: element for element in self.elements}

    EBMLSpecById = dict[int, EBMLElementDescription]

    def elements_by_id(self: Self) -> EBMLSpecById:
        return {element.id: element for element in self.elements}


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


def xml_int_optional(value: Optional[str | int], base: int = 10) -> Optional[int]:
    if value is None:
        return None

    return xml_int(value, base)


class RangeWrapper[A](ABC):

    @abstractmethod
    def parse(self: Self, value: str) -> Optional[A]: ...

    @abstractmethod
    def ge(self: Self, value: A) -> A: ...

    @abstractmethod
    def gt(self: Self, value: A) -> A: ...

    @abstractmethod
    def le(self: Self, value: A) -> A: ...

    @abstractmethod
    def lt(self: Self, value: A) -> A: ...


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

    if "-" in val:
        # Case 3: old syntax <num>-<num>
        num1, num2 = val.split("-", 1)
        if "-" in num2:
            return Err(f"Invalid syntax, only one '-' allowed: '{val}'")

        num1_v = generic.parse(num1)

        if num1_v is None:
            return Err(f"Invalid starting number: '{num1}'")

        num2_v = generic.parse(num2)

        if num2_v is None:
            return Err(f"Invalid ending number: '{num2}'")

        old_range: tuple[A, A] = (num1_v, num2_v + 1)
        return Ok(EBMLSchemaRange(old_range))

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
        # Case 4: new syntax <bound_num>, <bound_num>
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

        new_range_start_inclusive: A

        match bound1_b:
            case Bound.GE:
                new_range_start_inclusive = generic.ge(bound1_num)
            case Bound.GT:
                new_range_start_inclusive = generic.gt(bound1_num)
            case _:
                return Err(
                    f"First boundary has to be the lower boundary: but was: {bound1_b}: '{val}'",
                )

        bound2_b, bound2_num = bound2_v.as_ok()

        new_range_end_exclusive: A

        match bound2_b:
            case Bound.LE:
                new_range_end_exclusive = generic.le(bound2_num)
            case Bound.LT:
                new_range_end_exclusive = generic.lt(bound2_num)
            case _:
                return Err(
                    f"Second boundary has to be the upper boundary: but was: {bound2_b}: '{val}'",
                )

        new_range: tuple[A, A] = (new_range_start_inclusive, new_range_end_exclusive)
        return Ok(EBMLSchemaRange(new_range))

    # Case 5: <bound_num>
    bound_v = parse_bound(val)

    if bound_v.err():
        return Err(f"Invalid bound: {bound_v.as_err()}")

    bound_b, bound_num = bound_v.as_ok()

    match bound_b:
        case Bound.LE:
            return Ok(EBMLSchemaRange((None, generic.le(bound_num))))
        case Bound.LT:
            return Ok(EBMLSchemaRange((None, generic.lt(bound_num))))
        case Bound.GE:
            return Ok(EBMLSchemaRange((generic.ge(bound_num), None)))
        case Bound.GT:
            return Ok(EBMLSchemaRange((generic.gt(bound_num), None)))
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

    @override
    def ge(self: Self, value: int) -> int:
        return value

    @override
    def gt(self: Self, value: int) -> int:
        return value + 1

    @override
    def le(self: Self, value: int) -> int:
        return value

    @override
    def lt(self: Self, value: int) -> int:
        return value + 1


def xml_int_range_optional(value: Optional[str]) -> Optional[EBMLSchemaRange[int]]:
    return xml_any_range_optional(value, WrapperInt())


class WrapperFloat(RangeWrapper[float]):
    # TODO: use better representation, don't use epsilon, use proper >= > differentiation and math intervals
    __eps: float = 0.00000001

    @override
    def parse(self: Self, value: str) -> Optional[float]:
        return parse_float_safely(value)

    @override
    def ge(self: Self, value: float) -> float:
        return value

    @override
    def gt(self: Self, value: float) -> float:
        return value + self.__eps

    @override
    def le(self: Self, value: float) -> float:
        return value

    @override
    def lt(self: Self, value: float) -> float:
        return value + self.__eps


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

    result = parse_float_safely(value)

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


def ebml_read_spec_xml(name: str) -> EBMLSpec:

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

    result: EBMLSpec = EBMLSpec(version=version, elements=[])

    def append_element(element: EBMLElementDescription) -> None:
        id_value = result.elements_by_id().get(element.id, None)
        if id_value is not None:
            msg = f"Duplicate ID: {id_value}: {element}"
            raise RuntimeError(msg)

        name_value = result.elements_by_name().get(element.name, None)
        if name_value is not None:
            msg = f"Duplicate name: {name_value}: {element}"
            raise RuntimeError(msg)

        result.elements.append(element)

    for element_entry in root:

        element_entry_tag = xml_local_name(element_entry.tag)

        if element_entry_tag != "element":
            msg = f"Invalid element xml tag: {element_entry_tag}: {element_entry!s}"
            raise RuntimeError(msg)

        allowed_attributes: set[str] = set()

        name = xml_required(element_entry.attrib, "name")
        _path = xml_required(element_entry.attrib, "path")
        element_id = xml_int(xml_required(element_entry.attrib, "id"), 16)
        min_occurs = xml_int(element_entry.attrib.get("minOccurs", 0))
        max_occurs = xml_int_optional(element_entry.attrib.get("maxOccurs", None))
        element_type = EBMLElementType.from_str(
            xml_required(element_entry.attrib, "type"),
        )

        allowed_attributes.update(
            ["name", "path", "id", "minOccurs", "maxOccurs", "type"]
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

        element: EBMLElementDescription = EBMLElementDescription(
            name=name,
            id=element_id,
            occurrences=occurrences,
            type=advanced_type,
            description=description,
        )

        append_element(element)

    return result


def filter_spec_elements(
    spec: EBMLSpec,
    cb: Callable[[EBMLElementDescription], bool],
) -> EBMLSpec:

    result = EBMLSpec(elements=[], version=spec.version)
    for element in spec.elements:

        should_include = cb(element)

        if should_include:
            result.elements.append(element)

    return result
