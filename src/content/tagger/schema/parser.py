from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Self, TypeIs, assert_never, assert_type
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
class EBMLAdvancedElementTypeInteger:
    type: Literal[EBMLElementType.SignedInteger, EBMLElementType.UnsignedInteger]
    default: int | DefaultOptions
    range: Optional[tuple[int, int]]

    def validate(self: Self, value: int) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        min_inclusive, max_exclusive = self.range
        if value < min_inclusive:
            return Err(
                f"value needs to be between [{min_inclusive},{max_exclusive}] but was below it: {value}",
            )

        if value >= max_exclusive:
            return Err(
                f"value needs to be between [{min_inclusive},{max_exclusive}] but was above it: {value}",
            )

        return Ok(None)


@dataclass(slots=True, repr=True)
class EBMLAdvancedElementTypeFloat:
    type: Literal[EBMLElementType.Float]
    default: float | DefaultOptions
    range: Optional[tuple[float, float]]

    def validate(self: Self, value: float) -> Result[None, str]:
        if self.range is None:
            return Ok(None)

        min_inclusive, max_exclusive = self.range
        if value < min_inclusive:
            return Err(
                f"value needs to be between [{min_inclusive},{max_exclusive}] but was below it: {value}",
            )

        if value >= max_exclusive:
            return Err(
                f"value needs to be between [{min_inclusive},{max_exclusive}] but was above it: {value}",
            )

        return Ok(None)


class LengthRange:
    # range is start inclusive, end exclusive
    type Underlying = int | tuple[int, int]

    __underlying: Underlying

    def __init__(self: Self, value: Underlying) -> None:
        self.__underlying = value

    def valid(self: Self, value: Sized) -> Result[None, str]:
        length = len(value)

        if isinstance(self.__underlying, int):
            if length != self.__underlying:
                return Err(f"Length needs to be {self.__underlying} but was {length}")
        elif isinstance(self.__underlying, tuple):
            min_inclusive, max_exclusive = self.__underlying
            if length < min_inclusive:
                return Err(
                    f"Length needs to be between [{min_inclusive},{max_exclusive}] but was below it: {length}",
                )

            if length >= max_exclusive:
                return Err(
                    f"Length needs to be between [{min_inclusive},{max_exclusive}] but was above it: {length}",
                )

        return Ok(None)


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
        msg = f"Invlaid number: {value}"
        raise RuntimeError(msg)

    return result


def xml_int_optional(value: Optional[str | int], base: int = 10) -> Optional[int]:
    if value is None:
        return None

    return xml_int(value, base)


def xml_int_range_optional(value: Optional[str]) -> Optional[tuple[int, int]]:
    if value is None:
        return None

    raise NotImplementedError("TODO")


def xml_float_range_optional(value: Optional[str]) -> Optional[tuple[float, float]]:
    if value is None:
        return None

    raise NotImplementedError("TODO")


def xml_length_range_optional(value: Optional[str]) -> Optional[LengthRange]:
    if value is None:
        return None

    raise NotImplementedError("TODO")


def xml_float(value: str | float) -> float:
    if isinstance(value, float):
        return value

    result = parse_float_safely(value)

    if result is None:
        msg = f"Invlaid number: {value}"
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
