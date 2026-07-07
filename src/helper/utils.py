from collections.abc import Callable
from typing import Any, Optional, TypeIs

from helper.decorator import decorate_class
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


@decorate_class(slots=True)
class _MissingValueClass:
    pass


_MissingValue = _MissingValueClass()


def _is_missing_value[A](a: A | _MissingValueClass) -> TypeIs[_MissingValueClass]:
    return a is _MissingValue


def dict_at[A, B](dct: dict[A, B], key: A) -> Result[B, None]:
    value = dct.get(key, _MissingValue)

    if _is_missing_value(value):
        return Err(None)

    return Ok(value)


def dict_has[A](dct: dict[A, Any], key: A) -> bool:
    return dict_at(dct, key).ok()


def dict_wrapper[A](fn: Callable[[], A]) -> Result[A, None]:
    try:
        return Ok(fn())
    except KeyError:
        return Err(None)
