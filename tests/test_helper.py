import re
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Self, override

from conftest import FancyEq

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result


def file_duplicates(paths: list[Path]) -> AbstractContextManager[list[Path]]:

    @decorate_class(slots=True)
    class DuplicatesCtx(AbstractContextManager[list[Path]]):
        __paths: list[Path]

        def __init__(self: Self) -> None:
            super().__init__()
            self.__paths = []

        @override
        def __enter__(self: Self) -> list[Path]:
            results: list[Path] = []
            for path in paths:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix="video_language_detect_tests_",
                ) as f:
                    f.write(path.read_bytes())
                    results.append(Path(f.file.name))

            return results

        @override
        def __exit__(
            self: Self,
            _exc_type: Optional[type[BaseException]],
            _exc_val: Optional[BaseException],
            _exc_tb: Optional[TracebackType],
        ) -> Literal[False]:  # actually bool
            for path in self.__paths:
                path.unlink(missing_ok=True)

            return False

    return DuplicatesCtx()


@decorate_class(slots=True)
class AnyResultValue:
    pass


_AnyResultValue = AnyResultValue()


@decorate_class(slots=True)
class OkResult(FancyEq):
    __value: Any

    def __init__(self: Self, value: Any = _AnyResultValue) -> None:
        self.__value = value

    def __eq_other(self: Self, other_value: Any) -> Result[None, list[str]]:
        if isinstance(self.__value, AnyResultValue):
            return Ok(None)

        if self.__value == other_value:
            return Ok(None)

        return Err(["Ok value not the same", str(self.__value), str(other_value)])

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, Err):
            return (True, lambda: Err(["Expected Ok, but got Err", str(other)]))

        if isinstance(other, Ok):
            return (True, lambda: self.__eq_other(other.as_ok()))

        return (False, lambda: Err(["Invalid compare type", str(type(other))]))

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
        return "<OkResult {self.__value}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(id(self))


@decorate_class(slots=True)
class ErrResult(FancyEq):
    __value: Any

    def __init__(self: Self, value: Any = _AnyResultValue) -> None:
        self.__value = value

    def __eq_other(self: Self, other_value: Any) -> Result[None, list[str]]:
        if isinstance(self.__value, AnyResultValue):
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
            return (True, lambda: Err(["Expected Err, but got Ok", str(other)]))

        return (False, lambda: Err(["Invalid compare type", str(type(other))]))

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
        return "<ErrResult {self.__value}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(id(self))


def re_exact_string(value: str) -> re.Pattern[str]:
    base = re.escape(value)
    return re.compile(f"^{base}$")
