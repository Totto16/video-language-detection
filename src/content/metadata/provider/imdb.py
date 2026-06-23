# TODO: see: https://github.com/Totto16/imdb-dataset-to-postgresql


from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Self, override

from apischema import schema
from apischema.objects import ObjectField, object_fields

from content.metadata.interfaces import Provider
from content.metadata.metadata import InternalMetadataType
from content.shared import ScanType
from helper.apischema import OneOf


@dataclass(slots=True, repr=True)
class IMDBConfig:
    url: str


@dataclass(slots=True, repr=True)
class IMDBMetadataConfig:
    type: Literal["imdb"]
    config: Annotated[Optional[IMDBConfig], OneOf]


@dataclass(slots=True, repr=True)
@schema()
class IMDBMetadataSchema:
    data: None
    provider: Literal["imdb"]


# TODO: implement correctly based on IMDB2sql
class IMDBProvider(Provider):
    __config: IMDBConfig

    def __init__(self: Self, config: IMDBConfig) -> None:
        super().__init__("imdb")
        self.__config = config

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
    @staticmethod
    def get_metadata_schema() -> Mapping[str, ObjectField]:
        return object_fields(IMDBMetadataSchema)

    @override
    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> Optional[object]:
        msg = "TODO"
        raise NotImplementedError(msg)

    @override
    def get_season_metadata(
        self: Self,
        series_data: object,
        season: int,
    ) -> Optional[object]:
        msg = "TODO"
        raise NotImplementedError(msg)

    @override
    def get_episode_metadata(
        self: Self,
        series_data: object,
        season_data: object,
        episode: int,
    ) -> Optional[object]:
        msg = "TODO"
        raise NotImplementedError(msg)
