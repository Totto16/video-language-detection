from typing import Never, NoReturn


def assert_never(x: Never) -> NoReturn:
    msg = f"UNREACHABLE: Unhandled type: {type(x).__name__}"
    raise RuntimeError(msg)
