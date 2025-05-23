from typing import Any, Optional, Self

from content.metadata.interfaces import Provider
from content.metadata.metadata import InternalMetadataType, MetadataHandle, SkipHandle
from content.shared import ScanType


class MetadataScanner:
    __provider: Provider

    def __init__(
        self: Self,
        provider: Provider,
    ) -> None:
        self.__provider = provider

    def should_scan(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool:
        return self.__provider.should_scan(scan_type, metadata)

    def can_scan(self: Self) -> bool:
        return self.__provider.can_scan()

    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> InternalMetadataType:
        provider_name = self.__provider.name

        data: Optional[Any] = self.__provider.get_series_metadata(series_name)

        if isinstance(data, SkipHandle):
            return SkipHandle()

        if data is not None:
            return MetadataHandle(provider_name, data)

        return None

    def get_season_metadata(
        self: Self,
        series_handle: MetadataHandle,
        season: int,
    ) -> InternalMetadataType:
        provider_name = self.__provider.name
        if series_handle.provider != provider_name:
            return None

        data: Optional[Any] = self.__provider.get_season_metadata(
            series_handle.data,
            season,
        )

        if isinstance(data, SkipHandle):
            return SkipHandle()

        if data is not None:
            return MetadataHandle(provider_name, data)

        return None

    def get_episode_metadata(
        self: Self,
        series_handle: MetadataHandle,
        season_handle: MetadataHandle,
        episode: int,
    ) -> InternalMetadataType:
        provider_name: str = self.__provider.name
        if series_handle.provider != provider_name:
            return None

        if season_handle.provider != provider_name:
            return None

        data: Optional[Any] = self.__provider.get_episode_metadata(
            series_handle.data,
            season_handle.data,
            episode,
        )

        if isinstance(data, SkipHandle):
            return SkipHandle()

        if data is not None:
            return MetadataHandle(provider_name, data)

        return None
