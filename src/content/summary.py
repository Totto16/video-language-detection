from collections.abc import Iterable
from enum import StrEnum
from typing import (
    Optional,
    Self,
)

from content.general import EpisodeDescription, SeasonDescription, SeriesDescription
from content.language import Language
from content.metadata.metadata import (
    InternalMetadataType,
    should_skip_metadata,
)
from content.shared import MetadataKind
from content.video_metadata import VideoMetadata
from helper.decorator import decorate_class

CollectionDescription = str

type IdentifierDescription = (
    tuple[EpisodeDescription]
    | tuple[SeasonDescription, EpisodeDescription]
    | tuple[SeriesDescription, SeasonDescription, EpisodeDescription]
    | tuple[
        CollectionDescription,
        SeriesDescription,
        SeasonDescription,
        EpisodeDescription,
    ]
)


class MetadataType(StrEnum):
    ok = "ok"
    missing = "missing"
    skipped = "skipped"

    def __str__(self: Self) -> str:
        return f"<MetadataType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class VideoMetadataType(StrEnum):
    ok = "ok"
    missing = "missing"

    def __str__(self: Self) -> str:
        return f"<VideoMetadataType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


type LanguageDict = dict[Language, int]
type MetadataSubDict = dict[MetadataType, int]
type MetadataDict = dict[MetadataKind, MetadataSubDict]
type VideoMetadataDict = dict[VideoMetadataType, int]


type MetadataInput = tuple[MetadataKind, InternalMetadataType]


def metadata_handle_to_type(
    handle: InternalMetadataType,
) -> MetadataType:
    if handle is None:
        return MetadataType.missing

    if should_skip_metadata(handle):
        return MetadataType.skipped

    return MetadataType.ok


@decorate_class(slots=True)
class Summary:
    __complete: bool
    __detailed: bool

    __descriptions: list[IdentifierDescription]

    __language: LanguageDict
    __metadata: MetadataDict
    __duplicates: list[IdentifierDescription]
    __missing: list[IdentifierDescription]
    __video_metadata: VideoMetadataDict

    def __init__(
        self: Self,
        languages: list[Language],
        metadatas: list[MetadataInput],
        video_metadata: list[VideoMetadataType],
        descriptions: list[IdentifierDescription],
        *,
        detailed: bool = False,
    ) -> None:
        def get_lang_dict(language: Language) -> LanguageDict:
            dct: LanguageDict = {}
            dct[language] = 1
            return dct

        self.__language = Summary.__combine_language_dicts(
            get_lang_dict(language) for language in languages
        )

        def get_metadata_dict(metadata: MetadataInput) -> MetadataDict:
            dct: MetadataDict = {}
            kind, handle = metadata
            dct[kind] = {
                MetadataType.ok: 0,
                MetadataType.missing: 0,
                MetadataType.skipped: 0,
            }
            dct[kind][metadata_handle_to_type(handle)] += 1
            return dct

        self.__metadata = Summary.__combine_metadata_dicts(
            get_metadata_dict(metadata=metadata) for metadata in metadatas
        )

        def get_video_metadata_dict(typ: VideoMetadataType) -> VideoMetadataDict:
            dct: VideoMetadataDict = {
                VideoMetadataType.ok: 0,
                VideoMetadataType.missing: 0,
            }
            dct[typ] += 1
            return dct

        self.__video_metadata = Summary.__combine_video_metadata_dicts(
            get_video_metadata_dict(typ=v) for v in video_metadata
        )

        self.__duplicates = []
        self.__missing = []
        self.__complete = False
        self.__detailed = detailed
        self.__descriptions = descriptions

    @property
    def complete(self: Self) -> bool:
        return self.__complete

    @property
    def detailed(self: Self) -> bool:
        return self.__detailed

    @property
    def duplicates(self: Self) -> list[IdentifierDescription]:
        return self.__duplicates

    @property
    def missing(self: Self) -> list[IdentifierDescription]:
        return self.__missing

    @staticmethod
    def construct_for_episode(
        language: Language,
        metadata: InternalMetadataType,
        description: EpisodeDescription,
        video_metadata: Optional[VideoMetadata],
        *,
        detailed: bool,
    ) -> "Summary":

        return Summary(
            languages=[language],
            metadatas=[(MetadataKind.episode, metadata)],
            video_metadata=[
                (
                    VideoMetadataType.missing
                    if video_metadata is None
                    else VideoMetadataType.ok
                ),
            ],
            descriptions=[(description,)],
            detailed=detailed,
        )

    def combine_episodes(
        self: Self,
        description: SeasonDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if isinstance(desc[0], EpisodeDescription):
                self.__descriptions.append((description, desc[0]))

        self.__language = Summary.__combine_language_dicts(
            [self.__language, summary.language],
        )

        self.__metadata = Summary.__combine_metadata_dicts(
            [self.__metadata, summary.metadata],
        )

        self.__video_metadata = Summary.__combine_video_metadata_dicts(
            [self.__video_metadata, summary.video_metadata],
        )

    @staticmethod
    def construct_for_season(
        metadata: InternalMetadataType,
        description: SeasonDescription,
        episode_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary(
            languages=[],
            metadatas=[(MetadataKind.season, metadata)],
            video_metadata=[],
            descriptions=[],
            detailed=detailed,
        )
        for episode_summary in episode_summaries:
            summary.combine_episodes(description, episode_summary)

        return summary

    def combine_seasons(
        self: Self,
        description: SeriesDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if len(desc) == 2:
                self.__descriptions.append(
                    (description, desc[0], desc[1]),
                )

        self.__language = Summary.__combine_language_dicts(
            [self.__language, summary.language],
        )

        self.__metadata = Summary.__combine_metadata_dicts(
            [self.__metadata, summary.metadata],
        )

        self.__video_metadata = Summary.__combine_video_metadata_dicts(
            [self.__video_metadata, summary.video_metadata],
        )

    @staticmethod
    def construct_for_series(
        metadata: InternalMetadataType,
        description: SeriesDescription,
        season_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary(
            languages=[],
            metadatas=[(MetadataKind.series, metadata)],
            video_metadata=[],
            descriptions=[],
            detailed=detailed,
        )
        for season_summary in season_summaries:
            summary.combine_seasons(description, season_summary)

        return summary

    def combine_series(
        self: Self,
        description: CollectionDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if len(desc) == 3:
                self.__descriptions.append(
                    (description, desc[0], desc[1], desc[2]),
                )

        self.__language = Summary.__combine_language_dicts(
            [self.__language, summary.language],
        )

        self.__metadata = Summary.__combine_metadata_dicts(
            [self.__metadata, summary.metadata],
        )

        self.__video_metadata = Summary.__combine_video_metadata_dicts(
            [self.__video_metadata, summary.video_metadata],
        )

    @staticmethod
    def construct_for_collection(
        description: CollectionDescription,
        series_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary(
            languages=[],
            metadatas=[],
            video_metadata=[],
            descriptions=[],
            detailed=detailed,
        )
        for series_summary in series_summaries:
            summary.combine_series(description, series_summary)

        return summary

    @staticmethod
    def __combine_language_dicts(
        inp: Iterable[LanguageDict],
    ) -> LanguageDict:
        dct: LanguageDict = {}
        for input_dict in inp:
            for language, amount in input_dict.items():
                if language not in dct:
                    dct[language] = 0

                dct[language] += amount

        return dct

    @staticmethod
    def __combine_metadata_dicts(
        inp: Iterable[MetadataDict],
    ) -> MetadataDict:
        dct: MetadataDict = {}

        def combine_dicts(
            dict1: MetadataSubDict,
            dict2: MetadataSubDict,
        ) -> MetadataSubDict:
            final_dct: MetadataSubDict = {
                MetadataType.ok: 0,
                MetadataType.missing: 0,
                MetadataType.skipped: 0,
            }
            for key, value in dict1.items():
                final_dct[key] += value

            for key, value in dict2.items():
                final_dct[key] += value

            return final_dct

        for input_dict in inp:
            for key, value in input_dict.items():
                if key not in dct:
                    dct[key] = {
                        MetadataType.ok: 0,
                        MetadataType.missing: 0,
                        MetadataType.skipped: 0,
                    }
                dct[key] = combine_dicts(dct[key], value)

        return dct

    @staticmethod
    def __combine_video_metadata_dicts(
        inp: Iterable[VideoMetadataDict],
    ) -> VideoMetadataDict:
        dct: VideoMetadataDict = {
            VideoMetadataType.ok: 0,
            VideoMetadataType.missing: 0,
        }

        def combine_dicts(
            dict1: VideoMetadataDict,
            dict2: VideoMetadataDict,
        ) -> VideoMetadataDict:
            final_dct: VideoMetadataDict = {
                VideoMetadataType.ok: 0,
                VideoMetadataType.missing: 0,
            }
            for key, value in dict1.items():
                final_dct[key] += value

            for key, value in dict2.items():
                final_dct[key] += value

            return final_dct

        for input_dict in inp:
            dct = combine_dicts(dct, input_dict)

        return dct

    @staticmethod
    def combine_summaries(
        input_iterable: Iterable["Summary"],
    ) -> tuple[LanguageDict, MetadataDict, VideoMetadataDict]:
        input_list: list[tuple[LanguageDict, MetadataDict, VideoMetadataDict]] = [
            (inp.language, inp.metadata, inp.video_metadata) for inp in input_iterable
        ]

        lang_dict = Summary.__combine_language_dicts(inp[0] for inp in input_list)

        metadata_dict = Summary.__combine_metadata_dicts(inp[1] for inp in input_list)

        video_metadata_dict = Summary.__combine_video_metadata_dicts(
            inp[2] for inp in input_list
        )

        return (lang_dict, metadata_dict, video_metadata_dict)

    @property
    def descriptions(self: Self) -> list[IdentifierDescription]:
        return self.__descriptions

    @property
    def language(self: Self) -> LanguageDict:
        return self.__language

    @property
    def metadata(self: Self) -> MetadataDict:
        return self.__metadata

    @property
    def video_metadata(self: Self) -> VideoMetadataDict:
        return self.__video_metadata

    # TODO: human readable
    def __str__(self: Self) -> str:
        return repr(self)

    # TODO: this isn't finished yet
    def __repr__(self: Self) -> str:
        return f"<Summary languages: {self.__language}>"
