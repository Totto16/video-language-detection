import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Never, Optional, Self, assert_never, override

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result
from helper.translation import get_translator
from helper.utils import parse_int_safely

_ = get_translator()


@decorate_class(slots=True)
class Filter(ABC):

    @staticmethod
    @abstractmethod
    def factory_name() -> str: ...


@decorate_class(slots=True)
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
    def get_from_string(
        self: Self,
        value: str,
        options: Optional[str],
    ) -> Result[Filter, str]: ...


class SpecialFilterType(Enum):
    Empty = "empty"
    All = "all"
    Default = "default"


@decorate_class(slots=True)
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


@decorate_class(slots=True)
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


@decorate_class(slots=True)
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
    def get_from_string(
        self: Self,
        value: str,
        options: Optional[str],
    ) -> Result[ConfigFilter, str]:
        if options is not None:
            return Err(f"No options supported, but got: {options}")

        return Ok(ConfigFilter.from_string(value))


class ExecuteStep(Enum):
    Check = "check"
    Summary = "summary"
    Validate = "validate"


@decorate_class(slots=True)
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


@decorate_class(slots=True)
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
    def get_from_string(
        self: Self,
        value: str,
        options: Optional[str],
    ) -> Result[ExecuteFilter, str]:
        if options is not None:
            return Err(f"No options supported, but got: {options}")

        return ExecuteFilter.from_string(value)


@dataclass(slots=True, repr=True)
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


@decorate_class(slots=True)
class ValidatorFilter(Filter):
    __name: str
    __options: Optional[str]

    def __init__(self: Self, name: str, options: Optional[str]) -> None:
        super().__init__()

        self.__name = name
        self.__options = options

    @property
    def name(self: Self) -> str:
        return self.__name

    @property
    def options(self: Self) -> Optional[str]:
        return self.__options

    @override
    @staticmethod
    def factory_name() -> str:
        return ValidatorFilterFactory.name()


@dataclass(slots=True, repr=True)
class ValidatorChecks:
    check: Callable[[str, Optional[str]], Result[None, str]]
    names: set[str]


@decorate_class(slots=True)
class ValidatorFilterFactory(FilterFactory):
    __validator_checks: ValidatorChecks

    def __init__(self: Self, validator_checks: ValidatorChecks) -> None:
        super().__init__()

        self.__validator_checks = validator_checks

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
            f"'{validator}'" for validator in self.__validator_checks.names
        )
        return f"the validators to use, accepted values are: {values}"

    @override
    def get_from_string(
        self: Self,
        value: str,
        options: Optional[str],
    ) -> Result[ValidatorFilter, str]:
        result = self.__validator_checks.check(value, options)
        if result.ok():
            return Ok(ValidatorFilter(value, options))

        return Err(f"Invalid validator: {result.as_err()}")


class PathFilterType(Enum):
    Positive = "positive"
    Negative = "negative"


@decorate_class(slots=True)
class PathFilter(Filter):
    __type: PathFilterType
    __pattern: re.Pattern[str]

    def __init__(self: Self, pattern: re.Pattern[str], typ: PathFilterType) -> None:
        super().__init__()

        self.__pattern = pattern
        self.__type = typ

    @property
    def pattern(self: Self) -> re.Pattern[str]:
        return self.__pattern

    @property
    def type(self: Self) -> PathFilterType:
        return self.__type

    @override
    @staticmethod
    def factory_name() -> str:
        return PathFilterFactory.name()

    def __str__(self: Self) -> str:
        return f"<PathFilter type: {self.__type.name} pattern: {self.__pattern}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(("PathFilter", self.__type.value, self.__pattern))

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, PathFilter):
            return (self.__type, self.__pattern) == (other.type, other.pattern)

        return False


@decorate_class(slots=True)
class PathFilterFactory(FilterFactory):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    @staticmethod
    def name() -> str:
        return "path"

    @override
    @staticmethod
    def prefix() -> str:
        return "p"

    @override
    def help(self: Self) -> str:
        return """the path to scan, accepted values are: in the form '<filter_type>?<regex_type>:<regex_value>'
                regex_type: <required> '=' means exact match or '~' means regex match
                filter_type: <optional> '+' means positive alias include only those values or '-' means negative, or exclude all those values"""

    def __from_exact(self: Self, exact: str) -> re.Pattern[str]:
        base = re.escape(exact)
        return re.compile(f"^{base}$")

    def __from_regex(self: Self, pattern: str) -> re.Pattern[str]:
        return re.compile(pattern)

    @override
    def get_from_string(
        self: Self,
        value: str,
        options: Optional[str],
    ) -> Result[PathFilter, str]:
        if options is None:
            return Err(_("Options are required"))

        val: str
        type_str: Optional[str]
        if len(value) == 1:
            val = value
            type_str = None
        if len(value) == 2:
            val = value[1]
            type_str = value[0]

        pattern: re.Pattern[str]
        match val:
            case "=":
                pattern = self.__from_exact(options)
            case "~":
                pattern = self.__from_regex(options)
            case _:
                return Err(f"Unsupported regex_type: {val}, use '=' or '~'")

        typ_val: PathFilterType
        match type_str:
            case None:
                typ_val = PathFilterType.Positive
            case "+":
                typ_val = PathFilterType.Positive
            case "-":
                typ_val = PathFilterType.Negative
            case _:
                return Err(f"Unsupported filter_type: {type_str}, use '+' or '-'")

        return Ok(PathFilter(pattern, typ_val))


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


@dataclass(slots=True, repr=True)
class FilterHelpOptions:
    cb: Optional[Callable[[], Never]]


@decorate_class(slots=True)
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
        validator_checks: ValidatorChecks,
        help_options: FilterHelpOptions,
    ) -> None:
        __all_available_filter_factories: list[FilterFactory] = [
            ConfigFilterFactory(),
            ExecuteFilterFactory(),
            ValidatorFilterFactory(validator_checks),
            PathFilterFactory(),
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

        temp = arg.split(":", 2)
        if len(temp) == 1:
            msg = f"Invalid config string, expected <prefix>:<value> but got: {arg}"
            return Err(msg)

        prefix: str
        value: str
        options: Optional[str]

        if len(temp) == 2:
            prefix, value = temp
            options = None
        elif len(temp) == 3:
            prefix, value, options = temp
        else:
            msg = f"Implementation error, only two values expected, but got {len(temp)}"
            return Err(msg)

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

        filter_val = factory.get_from_string(value, options)

        if filter_val.err():
            msg = f"Invalid filter value for filter '{factory.name()}': {filter_val.as_err()}"
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
