from abc import ABC, abstractmethod
from typing import Self, override

from helper.result import Err, Ok, Result
from helper.utils import parse_int_safely


class Filter(ABC):

    @abstractmethod
    def factory_name(self: Self) -> str: ...


class FilterFactory(ABC):

    @property
    @abstractmethod
    def name(self: Self) -> str: ...

    @property
    @abstractmethod
    def prefix(self: Self) -> str: ...

    @abstractmethod
    def get_from_string(self: Self, value: str) -> Result[Filter, str]: ...


class ConfigFilter(Filter):
    type __Item = str | int

    __value: __Item

    def __init__(self: Self, value: __Item) -> None:
        super().__init__()

        self.__value = value

    @property
    def value(self: Self) -> __Item:
        return self.__value

    @staticmethod
    def from_string(inp: str) -> "ConfigFilter":
        num = parse_int_safely(inp)
        if num is not None:
            return ConfigFilter(num)

        return ConfigFilter(inp)

    @override
    def factory_name(self: Self) -> str:
        return "config"


class ConfigFilterFactory(FilterFactory):

    def __init__(self: Self) -> None:
        super().__init__()

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
        return Ok(ConfigFilter.from_string(value))


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


def parse_filter(arg: str) -> Result[Filter,str]:
    temp = arg.split(":", 1)
    if len(temp) == 1:
        msg = f"Invalid config string, expected <prefix>:<value> but got: {arg}"
        return Err(msg)

    if len(temp) != 2:
        msg = f"Implementation error, only two values expected, but got {len(temp)}"
        return Err(msg)

    prefix, value = temp

    factory = all_available_filter_factories.get(prefix, None)  # noqa: SIM910

    if factory is None:
        msg = f"Invalid config prefix '{prefix}', no filter factory has that prefix"
        return Err(msg)

    filter_val = factory.get_from_string(value)

    if filter_val.err():
        msg = f"Invalid filter value for filter {factory.name}: {filter_val.as_err()}"
        return Err(msg)

    return Ok(filter_val.as_ok())
