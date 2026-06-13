from typing import Optional


def parse_int_safely(inp: str, base: int = 10) -> Optional[int]:
    try:
        return int(inp, base)
    except ValueError:
        return None
