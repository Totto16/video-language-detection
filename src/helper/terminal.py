from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Self, override

from prompt_toolkit.shortcuts import clear


class ClearContextManager(AbstractContextManager["ClearContextManager"]):

    @override
    def __enter__(self: Self) -> Self:
        Terminal.clear()
        return self

    @override
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        Terminal.clear()


class Terminal:

    @staticmethod
    def clear() -> None:
        clear()

    @staticmethod
    def clear_block() -> ClearContextManager:
        return ClearContextManager()
