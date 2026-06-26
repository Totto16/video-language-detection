import re
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Self, override

from helper.decorator import decorate_class
from helper.result import Err, Ok


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
class OkResult:

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Err):
            return False
        if isinstance(value, Ok):
            return True

        msg = f"Invalid comapre type for OkResult and {type(value)}"
        raise ValueError(msg)

    def __str__(self: Self) -> str:
        return "<OkResult>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(id(self))

@decorate_class(slots=True)
class ErrResult:

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Err):
            return True
        if isinstance(value, Ok):
            return False

        msg = f"Invalid comapre type for ErrResult and {type(value)}"
        raise ValueError(msg)

    def __str__(self: Self) -> str:
        return "<ErrResult>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(id(self))



def re_exact_string(value: str) -> re.Pattern[str]:
    base = re.escape(value)
    return re.compile(f"^{base}$")
