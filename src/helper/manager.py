from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Annotated,
    Any,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    cast,
    override,
)

import enlighten
import pydantic
import pydantic_core

from helper.decorator import decorate_class
from helper.translation import get_translator

ManagerJustify = enlighten.Justify


class _ImplJustifyEnum(Enum):

    CENTER = "center"
    LEFT = "ljust"
    RIGHT = "rjust"


class _PydanticJustifyAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:

        return pydantic_core.core_schema.enum_schema(
            _ImplJustifyEnum,
            list(_ImplJustifyEnum.__members__.values()),
        )


_ = get_translator()


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar
class StatusBarOptions(TypedDict, total=False):
    color: str
    justify: Annotated[enlighten.Justify, _PydanticJustifyAnnotation]
    min_delta: float  # = 0.1
    status_format: str


type AdditionalArgs = dict[str, Any]


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.status_bar
class StatusBarGetOptions(StatusBarOptions, total=False):
    autorefresh: bool
    additional_args: AdditionalArgs


class SupportsFloat(Protocol):
    __slots__ = ()

    @abstractmethod
    def __float__(self) -> float: ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        return pydantic_core.core_schema.float_schema()


# actual type int, but implementation and python allows classes, which support int() or float()
type NumberLike = int | float | SupportsFloat


def number_like_convert_to_serializable(number_like: NumberLike) -> float:
    return float(number_like)


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter
class CounterOptions(TypedDict, total=False):
    bar_format: str
    count: NumberLike  # = 0,
    color: str
    desc: str
    leave: bool  # = True
    total: NumberLike
    unit: str


# see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
class StatusBarInterfaceUpdateOptions(TypedDict, total=False):
    force: bool
    additional_args: AdditionalArgs

@decorate_class(slots=True)
class StatusBarInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @abstractmethod
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None: ...


# NOTE: only StatusBarInterface supports AdditionalArgs atm!

@decorate_class(slots=True)
class CounterInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter.update
    @abstractmethod
    def update(self: Self, incr: NumberLike = 1, *, force: bool = False) -> None: ...

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.Counter.close
    @abstractmethod
    def close(self: Self, *, clear: bool = False) -> None: ...

@decorate_class(slots=True)
class ManagerInterface(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.status_bar
    @abstractmethod
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface: ...

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.counter
    @abstractmethod
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface: ...

    @abstractmethod
    def stop(
        self: Self,
    ) -> None: ...

@decorate_class(slots=True)
class TuiStatusBar(StatusBarInterface):
    __impl: enlighten.StatusBar

    def __init__(self: Self, impl: enlighten.StatusBar) -> None:
        super().__init__()
        self.__impl = impl

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        modified_fields: StatusBarInterfaceUpdateOptions = {**fields}

        if modified_fields.get("additional_args") is not None:
            additional_args: AdditionalArgs = modified_fields["additional_args"]
            del modified_fields["additional_args"]
            for key, value in additional_args.items():
                if modified_fields.get(key) is not None:
                    msg = f"Trying to overwrite normal option key '{key}' in enlighten Manager implementation"
                    raise RuntimeError(msg)

                cast(AdditionalArgs, modified_fields)[key] = value

        return self.__impl.update(**modified_fields)

@decorate_class(slots=True)
class TuiCounter(CounterInterface):
    __impl: enlighten.Counter

    def __init__(self: Self, impl: enlighten.Counter) -> None:
        super().__init__()
        self.__impl = impl

    @override
    def update(self: Self, incr: NumberLike = 1, *, force: bool = False) -> None:
        return self.__impl.update(incr=incr, force=force)

    @override
    def close(self: Self, *, clear: bool = False) -> None:
        return self.__impl.close(clear=clear)

@decorate_class(slots=True)
class TuiManager(ManagerInterface):
    __impl: enlighten.Manager

    def __init__(self: Self) -> None:
        super().__init__()
        manager = enlighten.get_manager()
        if not isinstance(manager, enlighten.Manager):
            msg = _("UNREACHABLE (not runnable in notebooks)")
            raise TypeError(msg)

        self.__impl = manager

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        modified_kwargs: StatusBarGetOptions = {**kwargs}

        if modified_kwargs.get("additional_args") is not None:
            additional_args: AdditionalArgs = modified_kwargs["additional_args"]
            del modified_kwargs["additional_args"]
            for key, value in additional_args.items():
                if modified_kwargs.get(key) is not None:
                    msg = f"Trying to overwrite normal option key '{key}' in enlighten Manager implementation"
                    raise RuntimeError(msg)

                cast(AdditionalArgs, modified_kwargs)[key] = value

        status_bar = self.__impl.status_bar(
            **modified_kwargs,
        )
        return TuiStatusBar(impl=status_bar)

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.NotebookManager.counter
    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        counter = self.__impl.counter(
            position=None,
            **kwargs,
        )
        return TuiCounter(impl=counter)

    def stop(
        self: Self,
    ) -> None:
        return self.__impl.stop()

@decorate_class(slots=True)
class NoopStatusBar(StatusBarInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        pass

@decorate_class(slots=True)
class NoopCounter(CounterInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def update(self: Self, incr: NumberLike = 1, *, force: bool = False) -> None:
        pass

    @override
    def close(self: Self, *, clear: bool = False) -> None:
        pass

@decorate_class(slots=True)
class NoopManager(ManagerInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        return NoopStatusBar()

    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        return NoopCounter()

    def stop(
        self: Self,
    ) -> None:
        pass


# 64 KB
PROGRESS_CHUNK_SIZE: int = 64 * 1024
