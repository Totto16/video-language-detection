from dataclasses import dataclass
from typing import Annotated, Optional

from content.extensions.trakt import TraktConfig
from helper.apischema import OneOf


@dataclass(slots=True, repr=True)
class ExtensionsConfig:
    trakt: Annotated[
        Optional[TraktConfig],
        OneOf,
    ]

    @staticmethod
    def default() -> "ExtensionsConfig":
        return ExtensionsConfig(trakt=None)
