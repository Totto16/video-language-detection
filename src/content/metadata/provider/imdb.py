# TODO: see: https://github.com/Totto16/imdb-dataset-to-postgresql


from dataclasses import dataclass
from typing import Literal, Optional, Self, override

from apischema import schema

from content.metadata.interfaces import Provider
from content.metadata.metadata import InternalMetadataType
from content.shared import ScanType
from helper.apischema import SchemaType, get_schema


@dataclass
class IMDBConfig:
    url: str


@dataclass
class IMDBMetadataConfig:
    type: Literal["imdb"]
    config: Optional[IMDBConfig]


@dataclass
@schema()
class IMDBMetadataSchema:
    data: None
    provider: Literal["imdb"]


# TODO: implment correctly based on IMDB2sql
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
    def get_metadata_schema() -> SchemaType:
        return get_schema(IMDBMetadataSchema, emit_type="deserialize")
