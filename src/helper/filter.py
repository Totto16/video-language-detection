from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Never, Optional, Self, assert_never, override

from helper.result import Err, Ok, Result
from helper.utils import parse_int_safely


class Filter(ABC):

    @staticmethod
    @abstractmethod
    def factory_name() -> str: ...


class FilterFactory(ABC):

    @staticmethod
    @abstractmethod
    def name() -> str: ...

    @staticmethod
    @abstractmethod
    def prefix() -> str: ...

    @abstractmethod
    def help(self: Self) -> str: ...

    @abstractmethod
    def get_from_string(self: Self, value: str) -> Result[Filter, str]: ...


class SpecialFilterType(Enum):
    Empty = "empty"
    All = "all"
    Default = "default"


class SpecialFilter(Filter):
    __name: str
    __type: SpecialFilterType

    def __init__(self: Self, name: str, typ: SpecialFilterType) -> None:
        super().__init__()

        self.__name = name
        self.__type = typ

    @staticmethod
    @override
    def factory_name() -> str:
        return "default"

    @property
    def name(self: Self) -> str:
        return self.__name

    @property
    def type(self: Self) -> SpecialFilterType:
        return self.__type


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

    @staticmethod
    @override
    def factory_name() -> str:
        return ConfigFilterFactory.name()


class ConfigFilterFactory(FilterFactory):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    @staticmethod
    def name() -> str:
        return "config"

    @override
    @staticmethod
    def prefix() -> str:
        return "c"

    @override
    def help(self: Self) -> str:
        return "the config to use, accepted values are: the name or the index"

    @override
    def get_from_string(self: Self, value: str) -> Result[ConfigFilter, str]:
        return Ok(ConfigFilter.from_string(value))


class ExecuteStep(Enum):
    Check = "check"
    Summary = "summary"
    Validate = "validate"


class ExecuteFilter(Filter):
    __step: ExecuteStep

    def __init__(self: Self, step: ExecuteStep) -> None:
        super().__init__()

        self.__step = step

    @property
    def step(self: Self) -> ExecuteStep:
        return self.__step

    @staticmethod
    def from_string(inp: str) -> Result["ExecuteFilter", str]:
        try:
            step = ExecuteStep(inp)
            return Ok(ExecuteFilter(step))
        except ValueError:
            return Err(f"Invalid step value: {inp}")

    @override
    @staticmethod
    def factory_name() -> str:
        return ExecuteFilterFactory.name()


class ExecuteFilterFactory(FilterFactory):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    @staticmethod
    def name() -> str:
        return "execute"

    @override
    @staticmethod
    def prefix() -> str:
        return "e"

    @override
    def help(self: Self) -> str:
        values = ", ".join(f"'{step.value}'" for step in list(ExecuteStep))
        return f"the steps to execute, accepted values are: {values}"

    @override
    def get_from_string(self: Self, value: str) -> Result[ExecuteFilter, str]:
        return ExecuteFilter.from_string(value)


@dataclass
class ExecuteSteps:
    check: bool
    summary: bool
    validate: bool

    @staticmethod
    def default() -> "ExecuteSteps":
        return ExecuteSteps(check=True, summary=True, validate=True)

    @staticmethod
    def all() -> "ExecuteSteps":
        return ExecuteSteps(check=True, summary=True, validate=True)

    @staticmethod
    def empty() -> "ExecuteSteps":
        return ExecuteSteps(check=False, summary=False, validate=False)


def __execute_steps_from_filter_impl(
    execute_filter: list[ExecuteFilter | SpecialFilter],
) -> ExecuteSteps:
    if len(execute_filter) == 0:
        return ExecuteSteps.default()

    result = ExecuteSteps.empty()
    for filter_val in execute_filter:
        if isinstance(filter_val, SpecialFilter):
            if filter_val.name == ExecuteFilter.factory_name():
                result = ExecuteSteps.empty()
                match filter_val.type:
                    case SpecialFilterType.All:
                        result = ExecuteSteps.all()
                    case SpecialFilterType.Empty:
                        result = ExecuteSteps.empty()
                    case SpecialFilterType.Default:
                        result = ExecuteSteps.default()
                    case _:
                        assert_never(filter_val.type)

        elif isinstance(filter_val, ExecuteFilter):
            match filter_val.step:
                case ExecuteStep.Check:
                    if result.check:
                        msg = "check is already true, duplicate step detected"
                        raise RuntimeError(msg)
                    result.check = True
                case ExecuteStep.Summary:
                    if result.summary:
                        msg = "summary is already true, duplicate step detected"
                        raise RuntimeError(msg)
                    result.summary = True
                case ExecuteStep.Validate:
                    if result.validate:
                        msg = "validate is already true, duplicate step detected"
                        raise RuntimeError(msg)
                    result.validate = True
                case _:
                    assert_never(filter_val.step)
        else:
            assert_never(filter_val)

    return result


def execute_steps_from_filter(
    filters: list[Filter],
) -> ExecuteSteps:
    execute_filter: list[ExecuteFilter | SpecialFilter] = [
        filter_val
        for filter_val in filters
        if isinstance(filter_val, (ExecuteFilter, SpecialFilter))
    ]

    return __execute_steps_from_filter_impl(execute_filter)


class ValidatorFilter(Filter):
    __name: str

    def __init__(self: Self, name: str) -> None:
        super().__init__()

        self.__name = name

    @property
    def name(self: Self) -> str:
        return self.__name

    @override
    @staticmethod
    def factory_name() -> str:
        return ValidatorFilterFactory.name()


class ValidatorFilterFactory(FilterFactory):
    __available_validators: set[str]

    def __init__(self: Self, available_validators: set[str]) -> None:
        super().__init__()

        self.__available_validators = available_validators

    @override
    @staticmethod
    def name() -> str:
        return "validator"

    @override
    @staticmethod
    def prefix() -> str:
        return "v"

    @override
    def help(self: Self) -> str:
        values = ", ".join(
            f"'{validator}'" for validator in self.__available_validators
        )
        return f"the validators to use, accepted values are: {values}"

    @override
    def get_from_string(self: Self, value: str) -> Result[ValidatorFilter, str]:
        if value in self.__available_validators:
            return Ok(ValidatorFilter(value))

        return Err(f"Invalid validator: {value}")


special_help_values: list[str] = ["help", "h", "?"]
special_empty_values: list[str] = ["-", "~"]
special_all_values: list[str] = ["@"]
special_default_values: list[str] = ["!"]

special_values: list[str] = [
    *special_help_values,
    *special_empty_values,
    *special_all_values,
    *special_default_values,
]


@dataclass
class FilterHelpOptions:
    cb: Optional[Callable[[], Never]]


class FilterManager:
    __factories: dict[str, FilterFactory]
    __help_options: FilterHelpOptions

    # TODO: support more complex args
    # series filter by name (regex)
    # episode filter by some id (video file uuid?)
    # path filter (regex)
    # file extension can be done with the path filter (regex)

    # -f "s~:landman" -f "e:<id>" -f "p=:media"

    def __init__(
        self: Self,
        available_validators: set[str],
        help_options: FilterHelpOptions,
    ) -> None:
        __all_available_filter_factories: list[FilterFactory] = [
            ConfigFilterFactory(),
            ExecuteFilterFactory(),
            ValidatorFilterFactory(available_validators),
        ]

        all_available_filter_factories: dict[str, FilterFactory] = (
            FilterManager.__validate_all_filter_factories(
                __all_available_filter_factories,
            )
        )

        self.__factories = all_available_filter_factories
        self.__help_options = help_options

    @staticmethod
    def __validate_all_filter_factories(
        all_available_filter_factories: list[FilterFactory],
    ) -> dict[str, FilterFactory]:
        factories: dict[str, FilterFactory] = {}
        factory_names: set[str] = set()
        for factory in all_available_filter_factories:
            factory_prefix = factory.prefix()
            if factory_prefix in factories:
                msg = f"Duplicate filter prefix: {factory_prefix}"
                raise RuntimeError(msg)

            factory_name = factory.name()
            if factory_name in factory_names:
                msg = f"Duplicate filter name: {factory_name}"
                raise RuntimeError(msg)

            if factory_prefix in special_values:
                msg = f"invalid prefix '{factory_prefix}', it is a special prefix"
                raise RuntimeError(msg)

            factories[factory_prefix] = factory
            factory_names.add(factory_name)

        return factories

    def parse_filter(self: Self, arg: str) -> Result[Filter, str]:

        if arg in special_help_values:
            if self.__help_options.cb is not None:
                self.__print_help_impl()
                res: Never = self.__help_options.cb()
                assert_never(res)
            else:
                return Err("Got help arg, but help is not supported")

        temp = arg.split(":", 1)
        if len(temp) == 1:
            msg = f"Invalid config string, expected <prefix>:<value> but got: {arg}"
            return Err(msg)

        if len(temp) != 2:
            msg = f"Implementation error, only two values expected, but got {len(temp)}"
            return Err(msg)

        prefix, value = temp

        factory = self.__factories.get(prefix, None)

        if factory is None:
            msg = f"Invalid config prefix '{prefix}', no filter factory has that prefix"
            return Err(msg)

        if value in special_help_values:
            if self.__help_options.cb is not None:
                self.__print_help_impl_for_factory(factory)
                res = self.__help_options.cb()
                assert_never(res)
            else:
                return Err(
                    f"Got help arg for factory '{factory.name()}', but help is not supported",
                )

        if value in special_empty_values:
            return Ok(SpecialFilter(factory.name(), SpecialFilterType.Empty))

        if value in special_all_values:
            return Ok(SpecialFilter(factory.name(), SpecialFilterType.All))

        if value in special_default_values:
            return Ok(SpecialFilter(factory.name(), SpecialFilterType.Default))

        filter_val = factory.get_from_string(value)

        if filter_val.err():
            msg = f"Invalid filter value for filter '{factory.name}': {filter_val.as_err()}"
            return Err(msg)

        return Ok(filter_val.as_ok())

    # ruff: disable[T201]
    def __print_help_impl_for_factory(self: Self, factory: FilterFactory) -> None:
        print(f"Filter '{factory.name()}'")
        print(f"\tprefix: '{factory.prefix()}'")
        print(f"\tvalue: {factory.help()}")
        print()

    def __print_help_impl(self: Self) -> None:
        print("Filter help:")
        print()

        for factory in self.__factories.values():
            self.__print_help_impl_for_factory(factory)

        print()
        print("Special values:")
        special_values_help_text: list[tuple[str, list[str], str]] = [
            (
                "empty",
                special_empty_values,
                "but as the filter state can be default (no filter provided) and you can add one with <prefix>, there needs to be a method, to set it to empty",
            ),
            (
                "all",
                special_all_values,
                "but it is a shortcut to specifying all available filters",
            ),
            (
                "default",
                special_default_values,
                "but this helps to reset the state to the default",
            ),
        ]
        for name, prefixes, but in special_values_help_text:
            print(
                f"\t{(", ".join(f"'{p}'" for p in prefixes))}: Only after the <prefix>. Reset to {name}, this reset the filter to it's defintion of '{name}', which may mean different things per filter, {but}",
            )
        print(
            "\t'help', 'h', '?': As standalone or after prefix. Prints the helper either for all filters, or if it is found after a prefix, for the current one",
        )

    # ruff: enable[T201]
