from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Self, override

from prompt_toolkit.shortcuts import clear


class ClearContextManager(AbstractContextManager["ClearContextManager"]):
    __clear_on_entry: bool

    def __init__(
        self: Self,
        *,
        clear_on_entry: bool,
    ) -> None:
        super().__init__()

        self.__clear_on_entry = clear_on_entry

    @override
    def __enter__(self: Self) -> Self:
        if self.__clear_on_entry:
            Terminal.clear()
        return self

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> None:
        Terminal.clear()


class Terminal:

    @staticmethod
    def clear() -> None:
        clear()

    @staticmethod
    def clear_block(
        *,
        clear_on_entry: bool = True,
    ) -> ClearContextManager:
        return ClearContextManager(clear_on_entry=clear_on_entry)
