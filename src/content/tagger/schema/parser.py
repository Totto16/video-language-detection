from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Self

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


@dataclass(slots=True, repr=True)
class EBMLElementDescription:
    name: str
    id: int
    occurrences: EBMLOccurrences
    type: EBMLElementType
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
