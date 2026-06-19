from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, override

from helper.decorator import decorate_class


@decorate_class(slots=True)
class ErrorMode(ABC):

    def __init__(self: Self) -> None:
        super().__init__()

    @abstractmethod
    def write_error(self: Self, error: str) -> None: ...

@decorate_class(slots=True)
class ErrorModeNone(ErrorMode):
    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def write_error(self: Self, error: str) -> None:
        pass

@decorate_class(slots=True)
class ErrorModeFile(ErrorMode):
    __file: Path

    def __init__(self: Self, file: Path) -> None:
        super().__init__()
        self.__file = file

    @override
    def write_error(self: Self, error: str) -> None:
        with self.__file.open(mode="a") as f:
            print(error, file=f)
