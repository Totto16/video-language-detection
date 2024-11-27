from dataclasses import dataclass
from datetime import date
from logging import Logger
from typing import Any, Literal, Optional, Self, override

from themoviedb import TMDb

from content.metadata.interfaces import Provider
from content.shared import ScanType
from helper.log import get_logger

logger: Logger = get_logger()


@dataclass
class Config:
    api_key: str
    language: Optional[str]
    region: Optional[str]


@dataclass
class TMDBMetadataConfig:
    type: Literal["tmdb"]
    config: Optional[Config]


@dataclass
class EpisodeMetadata:
    air_date: Optional[date]
    runtime: Optional[int]
    vote_average: Optional[float]
    vote_count: Optional[int]
    episode_number: int

    # TODO: serialize and deserialize this
    def as_dict(self: Self) -> dict[str, Any]:
        return {
            "air_date": self.air_date,
            "runtime": self.runtime,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count,
            "episode_number": self.episode_number,
        }


class TMDBProvider(Provider):
    __client: TMDb

    def __init__(self: Self, config: Config) -> None:
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

    def __get_metadata_for_episode(
        self: Self,
        series_data: Any,
        season_data: Any,
    ) -> Optional[tuple[int, int]]:
        # TODO: extract the needed things, after we know the internal structure of the data for season and series
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

        series_id, season = episode_metadata

        try:
            result = self.__client.episode(series_id, season, episode).details()

            if result.episode_number is None:
                return None

            return EpisodeMetadata(
                result.air_date,
                result.runtime,
                result.vote_average,
                result.vote_count,
                result.episode_number,
            )
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None
