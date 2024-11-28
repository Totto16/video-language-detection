from dataclasses import dataclass, field
from datetime import date
from logging import Logger
from typing import Any, Literal, Optional, Self, override

from apischema.metadata import none_as_undefined
from requests import HTTPError
from themoviedb import TMDb

from content.metadata.interfaces import Provider
from content.shared import ScanType
from helper.log import get_logger

logger: Logger = get_logger()


@dataclass
class TMDBConfig:
    api_key: str
    language: Optional[str] = field(default=None, metadata=none_as_undefined)
    region: Optional[str] = field(default=None, metadata=none_as_undefined)


@dataclass
class TMDBMetadataConfig:
    type: Literal["tmdb"]
    config: Optional[TMDBConfig]


@dataclass
class SeriesMetadata:
    episodes_count: Optional[int]
    seasons_count: Optional[int]
    status: Optional[str]
    type: Optional[str]
    vote_average: Optional[float]
    vote_count: Optional[int]
    first_air_date: Optional[date]
    series_id: int

    # TODO: serialize and deserialize this
    def as_dict(self: Self) -> dict[str, Any]:
        return {
            # TODO
        }


@dataclass
class SeasonMetadata:
    air_date: Optional[date]
    episodes_count: Optional[int]
    name: Optional[str]
    season_number: Optional[int]
    season_id: int

    # TODO: serialize and deserialize this
    def as_dict(self: Self) -> dict[str, Any]:
        return {
            "air_date": self.air_date,
            "episodes_count": self.episodes_count,
            "name": self.name,
            "season_number": self.season_number,
            "season_id": self.season_id,
        }


@dataclass
class EpisodeMetadata:
    air_date: Optional[date]
    runtime: Optional[int]
    vote_average: Optional[float]
    vote_count: Optional[int]
    name: Optional[str]
    episode_number: int

    # TODO: serialize and deserialize this
    def as_dict(self: Self) -> dict[str, Any]:
        return {
            "air_date": self.air_date,
            "runtime": self.runtime,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count,
            "name": self.name,
            "episode_number": self.episode_number,
        }


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
    ) -> bool:
        if scan_type == ScanType.first_scan:  # noqa: SIM103
            return True

        # TODO: a rescan should be requested, or done in newly added things, that likely changed

        return False

    # TODO: if we are being rate limited, return false here
    @override
    def can_scan(self: Self) -> bool:
        return True

    def __get_metadata_for_season(
        self: Self,
        series_data: Any,
    ) -> Optional[int]:
        if isinstance(series_data, SeriesMetadata):
            return series_data.series_id

        return None

    def __get_metadata_for_episode(
        self: Self,
        series_data: Any,
        season_data: Any,
    ) -> Optional[tuple[int, int]]:
        series_id = self.__get_metadata_for_season(series_data)
        if series_id is None:
            return None

        if isinstance(season_data, SeasonMetadata):
            return (series_id, season_data.season_id)

        return None

    @override
    def get_series_metadata(
        self: Self,
        series_name: str,
    ) -> Optional[Any]:
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
                result.id,
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
        series_data: Any,
        season: int,
    ) -> Optional[Any]:
        season_metadata = self.__get_metadata_for_season(series_data)

        if season_metadata is None:
            return None

        try:
            result = self.__client.season(season_metadata, season).details()

            if result.season_id is None:
                return None

            return SeasonMetadata(
                result.air_date,
                len(result.episodes) if result.episodes is not None else None,
                result.name,
                result.season_number,
                result.season_id,
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
        series_data: Any,
        season_data: Any,
        episode: int,
    ) -> Optional[Any]:
        episode_metadata = self.__get_metadata_for_episode(series_data, season_data)

        if episode_metadata is None:
            return None

        series_id, season_id = episode_metadata

        try:
            result = self.__client.episode(series_id, season_id, episode).details()

            if result.episode_number is None:
                return None

            return EpisodeMetadata(
                result.air_date,
                result.runtime,
                result.vote_average,
                result.vote_count,
                result.name,
                result.episode_number,
            )
        except HTTPError as err:
            logger.error(err)  # noqa: TRY400
            return None
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None
