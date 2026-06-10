from typing import TYPE_CHECKING, Never, Protocol, Self


class Result[T, E](Protocol):
    def ok(self: Self) -> bool: ...

    def err(self: Self) -> bool: ...

    def as_ok(self: Self) -> T: ...

    def as_err(self: Self) -> E: ...

    def ok_or[U](self: Self, default: U) -> T | U: ...

    def err_or[U](self: Self, default: U) -> E | U: ...


class Ok[T, O = None](Result[T, O]):
    __value: T

    def __init__(self: Self, value: T) -> None:
        self.__value = value

    def ok(self: Self) -> bool:
        return True

    def err(self: Self) -> bool:
        return False

    def as_ok(self: Self) -> T:
        return self.__value

    def as_err[E](self: Self) -> Never:
        msg = "Called as_err() on ok"
        raise RuntimeError(msg)

    def ok_or[U](self: Self, default: U) -> T:  # noqa: ARG002
        return self.__value

    def err_or[U](self: Self, default: U) -> U:
        return default

    def __str__(self: Self) -> str:
        return f"<Ok {self.as_ok()!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


class Err[E, O = None](Result[O, E]):
    __error: E

    def __init__(self: Self, error: E) -> None:
        self.__error = error

    def ok(self: Self) -> bool:
        return False

    def err(self: Self) -> bool:
        return True

    def as_ok[T](self: Self) -> Never:
        msg = "Called as_ok() on error"
        raise RuntimeError(msg)

    def as_err(self: Self) -> E:
        return self.__error

    def ok_or[U](self: Self, default: U) -> U:
        return default

    def err_or[U](self: Self, default: U) -> E:  # noqa: ARG002
        return self.__error

    def __str__(self: Self) -> str:
        return f"<Err {self.as_err()!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


if TYPE_CHECKING:
    # check protocol
    _check1: Result[None, str] = Err[str]("")
    _check2: Result[str, None] = Ok[str]("")
