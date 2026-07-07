from abc import ABC, abstractmethod
from typing import Optional, Self

import pytest
from fixtures import cached_file_manager, mark_as_used, video_file_dict

from helper.decorator import decorate_class
from helper.result import Err, Ok, Result
from helper.translation import DEFAULT_LANGUAGE, TRANSLATION_DIR, TRANSLATION_DOMAIN

mark_as_used(video_file_dict)
mark_as_used(cached_file_manager)


@decorate_class(slots=True)
class FancyEq(ABC):

    @abstractmethod
    def supports_fancy_eq(self: Self, other: object) -> bool: ...

    @abstractmethod
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]: ...

    @staticmethod
    def compare(value1: object, value2: object) -> Result[Optional[list[str]], None]:
        if isinstance(value1, FancyEq):
            if value1.supports_fancy_eq(value2):
                return Ok(value1.fancy_eq(value2))

            return Ok(
                [
                    "No fancy eq supported between:",
                    str(type(value1)),
                    str(type(value2)),
                ],
            )

        if isinstance(value2, FancyEq):
            if value2.supports_fancy_eq(value1):
                return Ok(value2.fancy_eq(value1))

            return Ok(
                [
                    "No fancy eq supported between:",
                    str(type(value2)),
                    str(type(value1)),
                ],
            )

        return Err(None)


def pytest_assertrepr_compare(
    config: pytest.Config,  # noqa: ARG001
    op: str,
    left: object,
    right: object,
) -> Optional[list[str]]:
    if op == "==":
        res = FancyEq.compare(left, right)

        if res.err():
            return None

        return res.as_ok()

    return None


def fixed_translator() -> None:
    import gettext  # noqa: PLC0415

    translation = gettext.translation(
        TRANSLATION_DOMAIN,
        localedir=TRANSLATION_DIR,
        languages=[DEFAULT_LANGUAGE],
    )
    translation.install()


fixed_translator()
