from collections.abc import Callable
from typing import Any, Optional

from helper.result import Err, Ok, Result


def parse_int_safely(inp: str, base: int = 10) -> Optional[int]:
    try:
        return int(inp, base)
    except ValueError:
        return None


def parse_float_safely(inp: str) -> Optional[float]:
    try:
        return float(inp)
    except ValueError:
        return None


def dict_at[A, B](dct: dict[A, B], key: A) -> Result[B, None]:
    pass


def dict_has[A](dct: dict[A, Any], key: A) -> bool:
    pass


def dict_wrapper[A](fn: Callable[[], A]) -> Result[A, None]:
    try:
        return Ok(fn())
    except KeyError:
        return Err(None)
