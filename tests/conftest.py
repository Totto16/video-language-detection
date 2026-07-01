from abc import ABC, abstractmethod
from typing import Optional, Self

import pytest
from fixtures import cached_file_manager, mark_as_used, video_file_dict

from helper.decorator import decorate_class
from helper.translation import DEFAULT_LANGUAGE, TRANSLATION_DIR, TRANSLATION_DOMAIN

mark_as_used(video_file_dict)
mark_as_used(cached_file_manager)


@decorate_class(slots=True)
class FancyEq(ABC):

    @abstractmethod
    def supports_fancy_eq(self: Self, other: object) -> bool: ...

    @abstractmethod
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]: ...


def pytest_assertrepr_compare(
    config: pytest.Config,  # noqa: ARG001
    op: str,
    left: object,
    right: object,
) -> Optional[list[str]]:
    if op == "==" and isinstance(left, FancyEq) and left.supports_fancy_eq(right):
        return left.fancy_eq(right)

    if op == "==" and isinstance(right, FancyEq) and right.supports_fancy_eq(left):
        return right.fancy_eq(left)

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
