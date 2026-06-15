from abc import ABC, abstractmethod
from typing import Self, override

from helper.result import Result
from helper.utils import parse_int_safely


class Filter(ABC):

    @abstractmethod
    def todo(self: Self) -> None: ...


class FilterFactory(ABC):

    @abstractmethod
    @property
    def name(self: Self) -> str: ...

    @abstractmethod
    @property
    def prefix(self: Self) -> str: ...

    @abstractmethod
    def get_from_string(self: Self, value: str) -> Result[Filter, str]: ...


class ConfigFilter(Filter):

    def __init__(self: Self) -> None:
        super().__init__()


class ConfigFilterFactory(FilterFactory):

    type __Item = str | int

    def __init__(self: Self) -> None:
        super().__init__()

    def __parse_filter_string(self: Self, inp: str) -> __Item:
        num = parse_int_safely(inp)
        if num is not None:
            return num

        return inp

    @override
    @property
    def name(self: Self) -> str:
        return "config"

    @override
    @property
    def prefix(self: Self) -> str:
        return "c"

    @override
    def get_from_string(self: Self, value: str) -> Result[ConfigFilter, str]:
        raise NotImplementedError("TODO")


# TODO: support more complex args
# config filter by name or idx
# series filter by name (regex)
# episode filter by some id (video file uuid?)
# path filter (regex)
# file extension can be done with the path filter (regex)

# "-f "c:0" -f "s~:landman" -f "e:<id>" -f "p=:media" -e "check" -e "summary" -e "validate" -e "test"
__all_available_filter_factories: list[FilterFactory] = [ConfigFilterFactory()]


def validate_all_filter_factories() -> dict[str, FilterFactory]:
    factories: dict[str, FilterFactory] = {}
    for factory in __all_available_filter_factories:
        if factory.prefix in factories:
            msg = f"Duplicate filter prefix: {factory.prefix}"
            raise RuntimeError(msg)

        factories[factory.prefix] = factory

    return factories


all_available_filter_factories: dict[str, FilterFactory] = (
    validate_all_filter_factories()
)
