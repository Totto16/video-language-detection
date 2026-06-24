from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from logging import Logger
from typing import Annotated, Literal, Optional, Self, override

from apischema import schema, type_name
from apischema.metadata import none_as_undefined
from apischema.objects import ObjectField, object_fields
from requests import HTTPError
from themoviedb.tmdb import TMDb

from content.metadata.interfaces import Provider
from content.metadata.metadata import (
    InternalMetadataType,
    SkipHandle,
    SkipMetadata,
    should_skip_metadata,
)
from content.shared import ScanType
from helper.apischema import OneOf
from helper.log import get_logger
from helper.translation import get_translator

logger: Logger = get_logger()
_ = get_translator()


@dataclass(slots=True, repr=True)
class TMDBConfig:
    api_key: str
    language: Optional[str] = field(default=None, metadata=none_as_undefined)
    region: Optional[str] = field(default=None, metadata=none_as_undefined)


@dataclass(slots=True, repr=True)
class TMDBMetadataConfig:
    type: Literal["tmdb"]
    config: Annotated[Optional[TMDBConfig], OneOf]


@schema()
@type_name("TMDBSeriesMetadata")
@dataclass(slots=True, repr=True)
class SeriesMetadata:
    episodes_count: Optional[int]
    seasons_count: Optional[int]
    status: Optional[str]
    type: Optional[str]
    vote_average: Optional[float]
    vote_count: Optional[int]
    first_air_date: Optional[date]
    original_name: Optional[str]
    series_id: int
    metadata_type: Literal["series"]

    def __str__(self: Self) -> str:
        return "<TMDBSeriesMetadata ...>"

    def __repr__(self: Self) -> str:
        return str(self)


@schema()
@type_name("TMDBSeasonMetadata")
@dataclass(slots=True, repr=True)
class SeasonMetadata:
    air_date: Optional[date]
    episodes_count: Optional[int]
    name: Optional[str]
    season_number: int
    season_id: int
    metadata_type: Literal["season"]

    def __str__(self: Self) -> str:
        return "<TMDBSeasonMetadata ...>"

    def __repr__(self: Self) -> str:
        return str(self)


@type_name("TMDBEpisodeMetadata")
@schema()
@dataclass(slots=True, repr=True)
class EpisodeMetadata:
    air_date: Optional[date]
    runtime: Optional[int]
    vote_average: Optional[float]
    vote_count: Optional[int]
    name: Optional[str]
    episode_number: int
    metadata_type: Literal["episode"]

    def __str__(self: Self) -> str:
        return "<TMDBEpisodeMetadata ...>"

    def __repr__(self: Self) -> str:
        return str(self)


@schema()
@dataclass(slots=True, repr=True)
class TMDBSkipMetadata(SkipMetadata):
    reason: str
    metadata_type: Literal["skip"]

    def __str__(self: Self) -> str:
        return f"<TMDBSkipMetadata reason: {self.reason}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class SeriesHandle:
    series_id: int

    def __str__(self: Self) -> str:
        return f"<SeriesHandle series_id: {self.series_id}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class SeasonHandle:
    parent: SeriesHandle
    season_number: int

    def as_tuple(self: Self) -> tuple[int, int]:
        return (self.parent.series_id, self.season_number)

    def __str__(self: Self) -> str:
        return (
            f"<SeriesHandle parent: {self.parent} season_number: {self.season_number}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class EpisodeHandle:
    id: int

    def __str__(self: Self) -> str:
        return f"<EpisodeHandle id: {self.id}>"

    def __repr__(self: Self) -> str:
        return str(self)


MetadataData = Annotated[
    TMDBSkipMetadata | EpisodeMetadata | SeasonMetadata | SeriesMetadata,
    OneOf,
]


@dataclass(slots=True, repr=True)
@schema()
class TMDBMetadataSchema:
    data: MetadataData
    provider: Literal["tmdb"]


class TMDBProvider(Provider):
    __client: TMDb

    def __init__(self: Self, config: TMDBConfig) -> None:
        super().__init__("tmdb")
        self.__client = TMDb(
            key=config.api_key,
            language=config.language,
            region=config.region,
        )

    @override
    def should_scan(
        self: Self,
        scan_type: ScanType,
        metadata: InternalMetadataType,
    ) -> bool:
        if scan_type == ScanType.first_scan:
            return True

        if metadata is None:  # noqa: SIM103
            return True

        # TODO: a rescan should be requested, or done in newly added things, that likely changed

        return False

    # TODO: if we are being rate limited, return false here
    @override
    def can_scan(self: Self) -> bool:
        return True

    def __get_metadata_for_season(
        self: Self,
        series_data: object,
    ) -> Optional[SeriesHandle | SkipHandle]:
        if isinstance(series_data, SeriesMetadata):
            return SeriesHandle(series_id=series_data.series_id)

        if isinstance(series_data, SkipMetadata):
            return SkipHandle()

        msg = _("Expected SeriesMetadata, but got other metadata data")
        logger.warning(msg)

        return None

    def __get_metadata_for_episode(
        self: Self,
        series_data: object,
        season_data: object,
    ) -> Optional[SeasonHandle | SkipHandle]:
        series_id = self.__get_metadata_for_season(series_data)
        if series_id is None:
            return None

        if should_skip_metadata(series_id):
            return SkipHandle()

        if isinstance(season_data, SeasonMetadata):
            return SeasonHandle(
                parent=series_id,
                season_number=season_data.season_number,
            )

        if isinstance(season_data, SkipMetadata):
            return SkipHandle()

        msg = _("Expected SeasonMetadata, but got other metadata data")
        logger.warning(msg)

        return None

    @override
    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> Optional[object]:
        try:
            search_result = self.__client.search().tv(series_name)
            if search_result.results is None:
                return None

            if len(search_result.results) == 0:
                return None

            # TODO: present this to the user and ask
            tv_id = search_result.results[0].id

            if tv_id is None:
                return None

            result = self.__client.tv(tv_id).details()

            if result.id is None:
                return None

            return SeriesMetadata(
                result.number_of_episodes,
                result.number_of_seasons,
                result.status,
                result.type,
                result.vote_average,
                result.vote_count,
                result.first_air_date,
                result.original_name,
                result.id,
                "series",
            )
        except HTTPError as err:
            logger.error(err)  # noqa: TRY400
            return None
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None

    @override
    def get_season_metadata(
        self: Self,
        series_data: object,
        season: int,
    ) -> Optional[object]:
        series_metadata = self.__get_metadata_for_season(series_data)
        if series_metadata is None:
            return None

        if should_skip_metadata(series_metadata):
            return SkipHandle()

        try:
            result = self.__client.season(series_metadata.series_id, season).details()

            if result.id is None:
                return None

            return SeasonMetadata(
                result.air_date,
                len(result.episodes) if result.episodes is not None else None,
                result.name,
                result.season_number or season,
                result.id,
                "season",
            )
        except HTTPError as err:
            logger.error(err)  # noqa: TRY400
            return None
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None

    @override
    def get_episode_metadata(
        self: Self,
        series_data: object,
        season_data: object,
        episode: int,
    ) -> Optional[object]:
        metadata_for_episode = self.__get_metadata_for_episode(series_data, season_data)

        if metadata_for_episode is None:
            return None

        if should_skip_metadata(metadata_for_episode):
            return SkipHandle()

        series_id, season_number = metadata_for_episode.as_tuple()

        try:
            result = self.__client.episode(series_id, season_number, episode).details()

            if result.episode_number is None:
                return None

            return EpisodeMetadata(
                result.air_date,
                result.runtime,
                result.vote_average,
                result.vote_count,
                result.name,
                result.episode_number,
                "episode",
            )
        except HTTPError as err:
            logger.error(err)  # noqa: TRY400
            return None
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None

    @override
    @staticmethod
    def get_metadata_schema() -> Mapping[str, ObjectField]:
        return object_fields(TMDBMetadataSchema)
