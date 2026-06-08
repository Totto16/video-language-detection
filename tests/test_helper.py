import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Self, Unpack, override

from helper.manager import (
    CounterInterface,
    CounterOptions,
    ManagerInterface,
    NumberLike,
    StatusBarGetOptions,
    StatusBarInterface,
    StatusBarInterfaceUpdateOptions,
)


class TestStatusBar(StatusBarInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    # see: https://python-enlighten.readthedocs.io/en/stable/api.html#enlighten.StatusBar.update
    @override
    def update(
        self: Self,
        **fields: Unpack[StatusBarInterfaceUpdateOptions],
    ) -> None:
        pass


class TestCounter(CounterInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def update(self: Self, incr: NumberLike = 1, *, force: bool = False) -> None:
        pass

    @override
    def close(self: Self, *, clear: bool = False) -> None:
        pass


class TestManager(ManagerInterface):

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def status_bar(
        self: Self,
        **kwargs: Unpack[StatusBarGetOptions],
    ) -> StatusBarInterface:
        return TestStatusBar()

    @override
    def counter(self: Self, **kwargs: Unpack[CounterOptions]) -> CounterInterface:
        return TestCounter()

    def stop(
        self: Self,
    ) -> None:
        pass


def file_duplicates(paths: list[Path]) -> AbstractContextManager[list[Path]]:

    class DuplicatesCtx(AbstractContextManager[list[Path]]):
        __paths: list[Path]

        def __init__(self: Self) -> None:
            super().__init__()
            self.__paths = []

        @override
        def __enter__(self: Self) -> list[Path]:
            results: list[Path] = []
            for path in paths:
                with tempfile.NamedTemporaryFile(delete=False) as f:
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
