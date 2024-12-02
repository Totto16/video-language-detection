from dataclasses import dataclass
from typing import Any, Literal, Optional, Self, override

from content.general import MissingOverrideError, SchemaType
from content.metadata.metadata import MetadataHandle
from content.shared import ScanType


class Provider:
    __name: str

    def __init__(self: Self, name: str) -> None:
        self.__name = name

    def get_series_metadata(
        self: Self,
        series_name: str,  # noqa: ARG002
    ) -> Optional[Any]:
        raise MissingOverrideError

    def get_season_metadata(
        self: Self,
        series_data: Any,  # noqa: ARG002
        season: int,  # noqa: ARG002
    ) -> Optional[Any]:
        raise MissingOverrideError

    def get_episode_metadata(
        self: Self,
        series_data: Any,  # noqa: ARG002
        season_data: Any,  # noqa: ARG002
        episode: int,  # noqa: ARG002
    ) -> Optional[Any]:
        raise MissingOverrideError

    def should_scan(
        self: Self,
        scan_type: ScanType,  # noqa: ARG002
        metadata: Optional[MetadataHandle],  # noqa: ARG002
    ) -> bool:
        raise MissingOverrideError

    def can_scan(self: Self) -> bool:
        raise MissingOverrideError

    @property
    def name(self: Self) -> str:
        return self.__name

    @staticmethod
    def get_metadata_schema() -> SchemaType:
        raise MissingOverrideError


@dataclass
class MissingProviderMetadataConfig:
    type: Literal["none"]


class MissingProvider(Provider):

    def __init__(self: Self) -> None:
        super().__init__("<missing>")

    @override
    def get_episode_metadata(
        self: Self,
        series_data: Any,
        season_data: Any,
        episode: int,
    ) -> Optional[Any]:
        msg = "Method 'get_episode_metadata' on MissingProvider called"
        raise RuntimeError(msg)

    @override
    def should_scan(
        self: Self,
        scan_type: ScanType,
        metadata: Optional[MetadataHandle],
    ) -> bool:
        return False

    @override
    def can_scan(self: Self) -> bool:
        return False
