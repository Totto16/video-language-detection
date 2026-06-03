from abc import ABC, abstractmethod
from functools import reduce
from logging import Logger
from typing import Any, NewType, Optional, Self, assert_never, cast, override

from content.base_class import Content
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import SeasonDescription, SeriesDescription
from content.language import Language
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from helper.classifier import ModelLanguage
from helper.log import get_logger
from helper.translation import get_translator

_ = get_translator()
logger: Logger = get_logger()

type ReporterWhere = tuple[
    SeriesDescription,
    SeasonDescription,
    EpisodeContent,
] | tuple[
    SeriesDescription,
    SeasonContent,
] | SeriesContent | CollectionContent


class ValidatorReporter(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    def format_where(
        self: Self,
        where: ReporterWhere,
    ) -> str:
        if isinstance(where, CollectionContent):
            return _("Collection {name}").format(name=where.description)
        if isinstance(where, SeriesContent):
            return _("Series {name}").format(name=where.description.name)

        if isinstance(where, tuple):
            if len(where) == 2:
                series, season = where
                return _("Series {name} Season {season}").format(
                    name=series.name,
                    season=season.description.season,
                )

            series, season, episode = where
            return _("Series {name} Season {season} Episode {episode}").format(
                name=series.name,
                season=season.season,
                episode=episode.description.episode,
            )

        assert_never(where)

    @abstractmethod
    def emit_error(
        self: Self,
        name: str,
        where: ReporterWhere,
        message: str,
    ) -> None: ...


class TuiValidatorReporter(ValidatorReporter):
    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def emit_error(
        self: Self,
        name: str,
        where: ReporterWhere,
        message: str,
    ) -> None:
        where_str = self.format_where(where)
        logger.error(
            _("Validator '{name}' error at {where}: {err}").format(
                name=name,
                where=where_str,
                err=message,
            ),
        )


class Validator[ED, SD, S2D, CD](ABC):
    __reporter: ValidatorReporter
    __name: str

    def __init__(self: Self, reporter: ValidatorReporter, name: str) -> None:
        super().__init__()
        self.__reporter = reporter
        self.__name = name

    def emit_error(
        self: Self,
        where: ReporterWhere,
        message: str,
    ) -> None:
        self.__reporter.emit_error(self.__name, where, message)

    def __validate_seasons_impl(
        self: Self, series: SeriesDescription, contents: list[SeasonContent]
    ) -> list[SD]:
        state: list[SD] = []

        for content in contents:
            local_state = [
                self.validate_episode(episode, series, content.description)
                for episode in content.episodes
            ]
            state.append(
                self.validate_season(content, series=series, result=local_state)
            )
        return state

    def __validate_series_impl(self: Self, contents: list[SeriesContent]) -> list[S2D]:
        state: list[S2D] = []

        for content in contents:
            local_state = self.__validate_seasons_impl(
                series=content.description,
                contents=content.seasons,
            )
            state.append(self.validate_series(content, local_state))

        return state

    def __validate_root_impl(self: Self, contents: list[Content]) -> None:
        state: list[S2D | CD] = []

        for content in contents:
            if isinstance(content, CollectionContent):
                local_state = self.__validate_series_impl(content.series)
                state.append(self.validate_collection(content, local_state))
            elif isinstance(content, SeriesContent):
                local_state = self.__validate_series_impl([content])
                state.extend(local_state)
            elif isinstance(content, SeasonContent):
                msg = _("'SeasonContent' not valid for this state")
                raise TypeError(msg)
            elif isinstance(content, EpisodeContent):
                msg = _("'EpisodeContent' not valid for this state ")
                raise TypeError(msg)
            else:
                msg = _("invalid type for 'Content': {typ}").format(typ=type(content))
                raise TypeError(msg)

    def validate(self: Self, contents: list[Content]) -> None:
        self.__validate_root_impl(contents)

    # helper for multipel validators to be typed correctly
    class __AnyClass:
        pass

    __Any1 = NewType("__Any1", __AnyClass)
    __Any2 = NewType("__Any2", __AnyClass)
    __Any3 = NewType("__Any3", __AnyClass)
    __Any4 = NewType("__Any4", __AnyClass)

    @staticmethod
    def __validate_multiple_seasons_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        series: SeriesDescription,
        contents: list[SeasonContent],
    ) -> list[list["Validator.__Any2"]]:
        state: list[list[Validator.__Any2]] = []

        for content in contents:
            local_states: list[list[Validator.__Any1]] = [
                [
                    validator.validate_episode(
                        episode,
                        series=series,
                        season=content.description,
                    )
                    for episode in content.episodes
                ]
                for validator in validators
            ]

            state.append(
                [
                    validator.validate_season(
                        content,
                        series=series,
                        result=local_state,
                    )
                    for validator, local_state in zip(
                        validators,
                        local_states,
                        strict=True,
                    )
                ],
            )
        return state

    @staticmethod
    def __validate_multiple_series_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        contents: list[SeriesContent],
    ) -> list[list["Validator.__Any3"]]:
        state: list[list[Validator.__Any3]] = []

        for content in contents:
            local_states: list[list[Any]] = Validator.__validate_multiple_seasons_impl(
                validators,
                series=content.description,
                contents=content.seasons,
            )
            state.append(
                [
                    validator.validate_series(content, local_state)
                    for validator, local_state in zip(
                        validators,
                        local_states,
                        strict=True,
                    )
                ],
            )

        return state

    @staticmethod
    def __validate_multiple_root_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        contents: list[Content],
    ) -> None:
        state: list[list[Validator.__Any4] | list[Validator.__Any3]] = []

        for content in contents:
            if isinstance(content, CollectionContent):
                local_states = Validator.__validate_multiple_series_impl(
                    validators,
                    content.series,
                )
                state.append(
                    [
                        validator.validate_collection(content, local_state)
                        for validator, local_state in zip(
                            validators,
                            local_states,
                            strict=True,
                        )
                    ],
                )
            elif isinstance(content, SeriesContent):
                local_states = Validator.__validate_multiple_series_impl(
                    validators,
                    [content],
                )
                state.extend(local_states)
            elif isinstance(content, SeasonContent):
                msg = _("'SeasonContent' not valid for this state")
                raise TypeError(msg)
            elif isinstance(content, EpisodeContent):
                msg = _("'EpisodeContent' not valid for this state ")
                raise TypeError(msg)
            else:
                msg = _("invalid type for 'Content': {typ}").format(typ=type(content))
                raise TypeError(msg)

    @staticmethod
    def validate_multiple(
        validators: list["Validator[Any, Any, Any, Any]"],
        contents: list[Content],
    ) -> None:
        Validator.__validate_multiple_root_impl(validators, contents)

    @abstractmethod
    def validate_episode(
        self: Self,
        episode: EpisodeContent,
        series: SeriesDescription,
        season: SeasonDescription,
    ) -> ED: ...

    @abstractmethod
    def validate_season(
        self: Self,
        season: SeasonContent,
        series: SeriesDescription,
        result: list[ED],
    ) -> SD: ...

    @abstractmethod
    def validate_series(
        self: Self,
        series: SeriesContent,
        result: list[SD],
    ) -> S2D: ...

    @abstractmethod
    def validate_collection(
        self: Self,
        collection: CollectionContent,
        result: list[S2D],
    ) -> CD: ...

    @abstractmethod
    def validate_all(
        self: Self,
        content: SeriesContent | CollectionContent,
        result: list[S2D | CD],
    ) -> None: ...


# language validators, check if the language is a correct one
class LanguageValidator(Validator[None, None, None, None]):
    __model_language: ModelLanguage

    def __init__(
        self: Self,
        reporter: ValidatorReporter,
        model_language: ModelLanguage,
    ) -> None:
        super().__init__(reporter, "language")
        self.__model_language = model_language

    @override
    def validate_episode(
        self: Self,
        episode: EpisodeContent,
        series: SeriesDescription,
        season: SeasonDescription,
    ) -> None:
        if episode.language in [Language.no_language(), Language.get_default()]:
            return

        is_valid = self.__model_language.is_valid_language(episode.language)

        if is_valid is None:
            return

        self.emit_error(
            (series, season, episode),
            _("Invalid language in episode: {err}").format(err=is_valid),
        )

    @override
    def validate_season(
        self: Self, season: SeasonContent, series: SeriesDescription, result: list[None]
    ) -> None:
        pass

    @override
    def validate_series(
        self: Self,
        series: SeriesContent,
        result: list[None],
    ) -> None:
        pass

    @override
    def validate_collection(
        self: Self,
        collection: CollectionContent,
        result: list[None],
    ) -> None:
        pass

    @override
    def validate_all(
        self: Self,
        content: SeriesContent | CollectionContent,
        result: list[None],
    ) -> None:
        pass


# language validators, check if the language is a correct one
class LanguageConsistencyValidator(
    Validator[
        tuple[EpisodeContent, Optional[Language]],
        tuple[SeasonContent, list[Language]],
        None,
        None,
    ],
):
    def __init__(
        self: Self,
        reporter: ValidatorReporter,
    ) -> None:
        super().__init__(reporter, "language consistency")

    @override
    def validate_episode(
        self: Self,
        episode: EpisodeContent,
        series: SeriesDescription,
        season: SeasonDescription,
    ) -> tuple[EpisodeContent, Optional[Language]]:
        if episode.language in [Language.no_language(), Language.get_default()]:
            return (episode, None)

        return (episode, episode.language)

    @override
    def validate_season(
        self: Self,
        season: SeasonContent,
        series: SeriesDescription,
        result: list[tuple[EpisodeContent, Optional[Language]]],
    ) -> tuple[SeasonContent, list[Language]]:

        # this checks for non continuity (in episodes) of the language e.g. lang1..., lang2... works, but not lang1...,lang2, lang1..., so either lang2  is wrong, or lang1 is wrongfully detected

        def process_languages(
            acc: list[Language],
            elem: tuple[EpisodeContent, Optional[Language]],
        ) -> list[Language]:
            episode, language = elem
            if language is None:
                return acc

            if len(acc) == 0:
                return [language]

            last_lang = acc[-1]

            if last_lang == language:
                return acc

            if len(acc) == 1:
                acc.append(last_lang)
                return acc

            self.emit_error(
                (series, season.description, episode),
                _(
                    "Language in season is not consistent, the last episode had the language {lang1!s} but this one has {lang2!s}"  # noqa: COM812
                ).format(lang1=last_lang, lang2=language),
            )
            return acc

        ret: list[Language] = reduce(
            process_languages,
            result,
            cast(list[Language], []),
        )

        if len(ret) == 0:
            self.emit_error((series, season), _("no language detected in season"))

        return (season, ret)

    @override
    def validate_series(
        self: Self,
        series: SeriesContent,
        result: list[tuple[SeasonContent, list[Language]]],
    ) -> None:
        # this checks for non continuity (in seasons) of the language e.g. lang1..., lang2... works, but not lang1...,lang2, lang1..., so either lang2  is wrong, or lang1 is wrongfully detected
        # this is more complex, as the change may occur in episodes alias is the sublists

        def process_languages(
            acc: list[Language],
            elem: tuple[SeasonContent, list[Language]],
        ) -> list[Language]:
            season, languages = elem
            if len(languages) == 0:
                self.emit_error(
                    (series.description, season), _("no language detected in season")
                )
                return acc

            def process_language(lang: Language) -> None:
                if len(acc) == 0:
                    return acc.append(lang)

                last_lang = acc[-1]

                if last_lang == lang:
                    return None

                if len(acc) == 1:
                    acc.append(last_lang)
                    return None

                self.emit_error(
                    (series.description, season),
                    _(
                        "Language in series is not consistent, the last season had the language {lang1!s} but this one has {lang2!s}"  # noqa: COM812
                    ).format(lang1=last_lang, lang2=lang),
                )

                return None

            for lang in languages:
                process_language(lang)

            return acc

        ret: list[Language] = reduce(
            process_languages,
            result,
            cast(list[Language], []),
        )

        if len(ret) == 0:
            self.emit_error(series, _("no language detected in series"))

    @override
    def validate_collection(
        self: Self,
        collection: CollectionContent,
        result: list[None],
    ) -> None:
        pass

    @override
    def validate_all(
        self: Self,
        content: SeriesContent | CollectionContent,
        result: list[None],
    ) -> None:
        pass
