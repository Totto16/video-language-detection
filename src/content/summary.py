from collections.abc import Iterable
from typing import (
    Optional,
    Self,
)

from classifier import Language
from content.general import EpisodeDescription, SeasonDescription, SeriesDescription
from content.metadata.metadata import MetadataHandle
from content.shared import MetadataKind

CollectionDescription = str

IdentifierDescription = (
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


LanguageDict = dict[Language, int]
MetadataSubDict = dict[bool, int]
MetadataDict = dict[MetadataKind, MetadataSubDict]

MetadataInput = tuple[MetadataKind, Optional[MetadataHandle]]


class Summary:
    __complete: bool
    __detailed: bool

    __descriptions: list[IdentifierDescription]

    __language: LanguageDict
    __metadata: MetadataDict
    __duplicates: list[IdentifierDescription]
    __missing: list[IdentifierDescription]

    def __init__(
        self: Self,
        languages: list[Language],
        metadatas: list[MetadataInput],
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
            dct[kind] = {True: 0, False: 0}
            dct[kind][handle is not None] += 1
            return dct

        self.__metadata = Summary.__combine_metadata_dicts(
            get_metadata_dict(metadata) for metadata in metadatas
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
        metadata: Optional[MetadataHandle],
        description: EpisodeDescription,
        *,
        detailed: bool,
    ) -> "Summary":

        return Summary(
            [language],
            [(MetadataKind.episode, metadata)],
            [(description,)],
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

    @staticmethod
    def construct_for_season(
        metadata: Optional[MetadataHandle],
        description: SeasonDescription,
        episode_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary([], [(MetadataKind.season, metadata)], [], detailed=detailed)
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

    @staticmethod
    def construct_for_series(
        metadata: Optional[MetadataHandle],
        description: SeriesDescription,
        season_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary([], [(MetadataKind.series, metadata)], [], detailed=detailed)
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

    @staticmethod
    def construct_for_collection(
        description: CollectionDescription,
        series_summaries: Iterable["Summary"],
        *,
        detailed: bool,
    ) -> "Summary":
        summary = Summary([], [], [], detailed=detailed)
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
                if dct.get(language) is None:
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
            final_dct: MetadataSubDict = {True: 0, False: 0}
            for key, value in dict1.items():
                final_dct[key] += value

            for key, value in dict2.items():
                final_dct[key] += value

            return final_dct

        for input_dict in inp:
            for key, value in input_dict.items():
                if dct.get(key) is None:
                    dct[key] = {True: 0, False: 0}
                dct[key] = combine_dicts(dct[key], value)

        return dct

    @staticmethod
    def combine_summaries(
        input_list: Iterable["Summary"],
    ) -> tuple[LanguageDict, MetadataDict]:
        lang_dict = Summary.__combine_language_dicts(inp.language for inp in input_list)

        metadata_dict = Summary.__combine_metadata_dicts(
            inp.metadata for inp in input_list
        )

        return (lang_dict, metadata_dict)

    @property
    def descriptions(self: Self) -> list[IdentifierDescription]:
        return self.__descriptions

    @property
    def language(self: Self) -> LanguageDict:
        return self.__language

    @property
    def metadata(self: Self) -> MetadataDict:
        return self.__metadata

    # TODO: human readable
    def __str__(self: Self) -> str:
        return repr(self)

    # TODO: this isn't finished yet
    def __repr__(self: Self) -> str:
        return f"<Summary languages: {self.__language}>"
