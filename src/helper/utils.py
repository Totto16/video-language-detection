from typing import Optional, TypeIs

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
