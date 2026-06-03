from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional, Self, override

from apischema.objects import ObjectField

from content.metadata.metadata import InternalMetadataType
from content.shared import ScanType
from helper.apischema import SchemaType
from helper.translation import get_translator

_ = get_translator()


class Provider(ABC):
    __name: str

    def __init__(self: Self, name: str) -> None:
        super().__init__()
        self.__name = name

    @abstractmethod
    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> Optional[object]: ...

    @abstractmethod
    def get_season_metadata(
        self: Self,
        series_data: object,
        season: int,
    ) -> Optional[object]: ...

    @abstractmethod
    def get_episode_metadata(
        self: Self,
        series_data: object,
        season_data: object,
        episode: int,
    ) -> Optional[object]: ...

    @abstractmethod
    def should_scan(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool: ...

    @abstractmethod
    def can_scan(self: Self) -> bool: ...

    @property
    def name(self: Self) -> str:
        return self.__name

    @staticmethod
    @abstractmethod
    def get_metadata_schema() -> Mapping[str, ObjectField]: ...


@dataclass
class MissingProviderMetadataConfig:
    type: Literal["none"]


class MissingProvider(Provider):

    def __init__(self: Self) -> None:
        super().__init__("<missing>")

    @override
    def get_episode_metadata(
        self: Self,
        series_data: object,
        season_data: object,
        episode: int,
    ) -> Optional[object]:
        msg = _("Method 'get_episode_metadata' on MissingProvider called")
        raise RuntimeError(msg)

    @override
    def should_scan(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool:
        return False

    @override
    def can_scan(self: Self) -> bool:
        return False

    @override
    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> Optional[object]:
        msg = _("Method 'get_series_metadata' on MissingProvider called")
        raise RuntimeError(msg)

    @override
    def get_season_metadata(
        self: Self,
        series_data: object,
        season: int,
    ) -> Optional[object]:
        msg = _("Method 'get_season_metadata' on MissingProvider called")
        raise RuntimeError(msg)

    @override
    @staticmethod
    def get_metadata_schema() -> SchemaType:
        msg = _("Method 'get_metadata_schema' on MissingProvider called")
        raise RuntimeError(msg)
