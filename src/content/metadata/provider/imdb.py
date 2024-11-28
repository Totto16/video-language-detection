# TODO: see: https://github.com/Totto16/imdb-dataset-to-postgresql


from dataclasses import dataclass
from typing import Literal, Optional, Self, override

from content.metadata.interfaces import Provider
from content.metadata.metadata import MetadataHandle
from content.shared import ScanType


@dataclass
class IMDBConfig:
    url: str


@dataclass
class IMDBMetadataConfig:
    type: Literal["imdb"]
    config: Optional[IMDBConfig]


# TODO


class IMDBProvider(Provider):
    __config: IMDBConfig

    def __init__(self: Self, config: IMDBConfig) -> None:
        super().__init__("imdb")
        self.__config = config

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
