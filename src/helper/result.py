from typing import Optional, Self


class Result[T, E]:
    __value: Optional[T]
    __error: Optional[E]

    def __init__(self: Self, *, _value: Optional[T], _error: Optional[E]) -> None:
        self.__value = _value
        self.__error = _error

    @staticmethod
    def ok(value: T) -> "Result[T, E]":
        return Result(_value=value, _error=None)

    @staticmethod
    def err(error: E) -> "Result[T, E]":
        return Result(_value=None, _error=error)

    def __assert_variant(self: Self) -> None:
        if self.__value is None and self.__error is None:
            msg = "Result: both values are None"
            raise RuntimeError(msg)
        if self.__value is not None and self.__error is not None:
            msg = "Result: no value is None"
            raise RuntimeError(msg)

    def is_ok(self: Self) -> bool:
        self.__assert_variant()
        return self.__value is not None

    def is_err(self: Self) -> bool:
        self.__assert_variant()
        return self.__error is not None

    def get_ok(self: Self) -> T:
        if self.__value is not None:
            return self.__value

        msg = f"Called get_ok() on error: {self.__error}"
        raise RuntimeError(msg)

    def get_err(self: Self) -> E:
        if self.__error is not None:
            return self.__error

        msg = f"Called get_err() on ok: {self.__value}"
        raise RuntimeError(msg)
