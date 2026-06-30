from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Self

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result

type EBMLOccurrences = int | tuple[int, int] | Literal["any"]


class EBMLElementType(Enum):
    SignedInteger = "si"
    UnsignedInteger = "ui"
    Float = "f"
    String = "s"
    UTF8 = "utf-8"
    Date = "d"
    Master = "m"
    Binary = "b"


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


class IntRange:
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
    length: Optional[IntRange]

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
    length: Optional[IntRange]

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
    description: str


@dataclass(slots=True, repr=True)
class EBMLSpec:
    elements: list[EBMLElementDescription]
    version: int

    EBMLSpecByName = dict[str, EBMLElementDescription]

    def elements_by_name(self: Self) -> EBMLSpecByName:
        raise "TODO"

    EBMLSpecById = dict[int, EBMLElementDescription]

    def elements_by_id(self: Self) -> EBMLSpecById:
        raise "TODO"


def ebml_read_spec_xml(name: str) -> EBMLSpec:

    file = Path(__file__).parent / name

    if not file.exists():
        msg = f"Spec XMl file '{file}' doesn't exist"
        raise RuntimeError(msg)

    result: EBMLSpec("TODO")

    for element_entry in element_entries:
        name = element_entry["name"]
        _path = element_entry["path"]
        id = element_entry["id"]
        minOccurs = element_entry.get("minOccurs", 0)
        maxOccurs = element_entry.get("maxOccurs", "unbounded")
        type = element_entry["type"]

        # match type:
        #     <xs:attribute name="range"/>
        #     <xs:attribute name="length"/>
        #     <xs:attribute name="default"/>

        description = element_entry.children["documentation"].text

    raise "TODO"


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
