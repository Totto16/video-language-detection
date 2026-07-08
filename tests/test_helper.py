from abc import ABC, abstractmethod
import re
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Optional, Protocol, Self, override

from conftest import FancyEq

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result


def file_duplicates(
    paths: list[tuple[Path, str]],
) -> AbstractContextManager[list[tuple[Path, str]]]:

    @decorate_class(slots=True)
    class DuplicatesCtx(AbstractContextManager[list[tuple[Path, str]]]):
        __paths: list[tuple[Path, str]]

        def __init__(self: Self) -> None:
            super().__init__()
            self.__paths = []

        @override
        def __enter__(self: Self) -> list[tuple[Path, str]]:
            results: list[tuple[Path, str]] = []
            for path, name in paths:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix="video_language_detect_tests_",
                ) as f:
                    f.write(path.read_bytes())
                    results.append((Path(f.file.name), name))

            return results

        @override
        def __exit__(
            self: Self,
            _exc_type: Optional[type[BaseException]],
            _exc_val: Optional[BaseException],
            _exc_tb: Optional[TracebackType],
        ) -> Literal[False]:  # actually bool
            for path, _name in self.__paths:
                path.unlink(missing_ok=True)

            return False

    return DuplicatesCtx()


@decorate_class(slots=True)
class TestResult[T, E](ABC):

    @abstractmethod
    def __eq__(self: Self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self: Self) -> int: ...


@decorate_class(slots=True)
class _AnyResultValueClass:
    pass


_AnyResultValue = _AnyResultValueClass()


@decorate_class(slots=True)
class OkResult[T, O = None](FancyEq, TestResult[T, O]):
    __value: T | _AnyResultValueClass

    def __init__(self: Self, value: T | _AnyResultValueClass = _AnyResultValue) -> None:
        self.__value = value

    def __eq_other[S](self: Self, other_value: S) -> Result[None, list[str]]:
        if self.__value is _AnyResultValue:
            return Ok(None)

        res = FancyEq.compare(self.__value, other_value)
        if res.ok():
            result = res.as_ok()
            if result is None:
                return Ok(None)

            return Err(result)

        if self.__value != other_value:
            return Err(["Ok value not the same", str(self.__value), str(other_value)])

        return Ok(None)

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, Err):
            return (
                True,
                lambda: Err(["Expected Ok, but got Err", str(self), str(other)]),
            )

        if isinstance(other, Ok):
            return (True, lambda: self.__eq_other(other.as_ok()))

        return (
            False,
            lambda: Err(["Invalid compare type", str(type(self)), str(type(other))]),
        )

    def __eq__(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[1]().ok()

    @override
    def supports_fancy_eq(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[0]

    @override
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]:
        supports_fancy_eq, cb = self.__eq_impl(other)
        assert supports_fancy_eq
        return cb().err_or(None)

    def __str__(self: Self) -> str:
        if self.__value is _AnyResultValue:
            return "<Any OkResult>"

        return f"<OkResult {self.__value}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(("OkResult", self.__value))


@decorate_class(slots=True)
class ErrResult[E, O = None](FancyEq, TestResult[O, E]):
    __value: E | _AnyResultValueClass

    def __init__(self: Self, value: E | _AnyResultValueClass = _AnyResultValue) -> None:
        self.__value = value

    def __eq_other[S](self: Self, other_value: S) -> Result[None, list[str]]:
        if self.__value is _AnyResultValue:
            return Ok(None)

        if self.__value == other_value:
            return Ok(None)

        return Err(["Err value not the same", str(self.__value), str(other_value)])

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, Err):
            return (True, lambda: self.__eq_other(other.as_err()))

        if isinstance(other, Ok):
            return (
                True,
                lambda: Err(["Expected Err, but got Ok", str(self), str(other)]),
            )

        return (
            False,
            lambda: Err(["Invalid compare type", str(type(self)), str(type(other))]),
        )

    def __eq__(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[1]().ok()

    @override
    def supports_fancy_eq(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[0]

    @override
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]:
        supports_fancy_eq, cb = self.__eq_impl(other)
        assert supports_fancy_eq
        return cb().err_or(None)

    def __str__(self: Self) -> str:
        if self.__value is _AnyResultValue:
            return "<Any ErrResult>"

        return f"<ErrResult {self.__value}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(("ErrResult", self.__value))


def re_exact_string(value: str) -> re.Pattern[str]:
    base = re.escape(value)
    return re.compile(f"^{base}$")


if TYPE_CHECKING:
    # check protocol
    _check1: TestResult[None, str] = ErrResult[str]("")
    _check2: TestResult[str, None] = OkResult[str]("")
