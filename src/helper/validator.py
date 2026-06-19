from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from logging import Logger
from pathlib import Path
from typing import Any, NewType, Optional, Self, assert_never, cast, override

from content.base_class import Content
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import SeasonDescription, SeriesDescription
from content.language import Language
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from content.tagger.tagger import get_tagger_for_file
from helper.classifier import ModelLanguage
from helper.filter import (
    Filter,
    SpecialFilter,
    SpecialFilterType,
    ValidatorChecks,
    ValidatorFilter,
)
from helper.log import get_logger
from helper.manager import NoopManager
from helper.result import Err, Ok, Result
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
] | SeriesContent | CollectionContent | Path


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

        if isinstance(where, Path):
            return _("File '{file}'").format(file=str(where))

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


@dataclass
class ValidatorParams:
    reporter: ValidatorReporter
    model_language: ModelLanguage


class Validator[ED, SD, S2D, CD](ABC):
    __reporter: ValidatorReporter
    __name: str

    def __init__(self: Self, reporter: ValidatorReporter, name: str) -> None:
        super().__init__()
        self.__reporter = reporter
        self.__name = name

    @property
    def name(self: Self) -> str:
        return self.__name

    def emit_error(
        self: Self,
        where: ReporterWhere,
        message: str,
    ) -> None:
        self.__reporter.emit_error(self.__name, where, message)

    def __validate_episodes_impl(
        self: Self,
        series: SeriesDescription,
        season: SeasonDescription,
        contents: list[EpisodeContent],
    ) -> list[ED]:
        state: list[ED] = [
            self.validate_episode(content, series=series, season=season)
            for content in contents
        ]
        return state

    def __validate_seasons_impl(
        self: Self,
        series: SeriesDescription,
        contents: list[SeasonContent],
    ) -> list[SD]:
        state: list[SD] = []

        for content in contents:
            local_state = self.__validate_episodes_impl(
                series=series,
                season=content.description,
                contents=content.episodes,
            )
            state.append(
                self.validate_season(content, series=series, result=local_state),
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

        root_content: list[SeriesContent | CollectionContent] = []

        for content in contents:
            if isinstance(content, CollectionContent):
                local_state = self.__validate_series_impl(content.series)
                state.append(self.validate_collection(content, local_state))
            elif isinstance(content, SeriesContent):
                local_state = self.__validate_series_impl([content])
                if len(local_state) != 1:
                    msg = "UNREACHABLE"
                    raise RuntimeError(msg)
                state.append(local_state[0])
            elif isinstance(content, SeasonContent):
                msg = _("'SeasonContent' not valid for this state")
                raise TypeError(msg)
            elif isinstance(content, EpisodeContent):
                msg = _("'EpisodeContent' not valid for this state ")
                raise TypeError(msg)
            else:
                msg = _("invalid type for 'Content': {typ}").format(typ=type(content))
                raise TypeError(msg)

            root_content.append(content)

        self.validate_all(root_content, state)

    def validate(self: Self, contents: list[Content]) -> None:
        self.__validate_root_impl(contents)

    # helper for multipel validators to be typed correctly
    class __AnyClass:
        pass

    __Any1 = NewType("__Any1", __AnyClass)
    __Any2 = NewType("__Any2", __AnyClass)
    __Any3 = NewType("__Any3", __AnyClass)
    __Any4 = NewType("__Any4", __AnyClass)

    @dataclass
    class __ValidatorState[S]:
        data: list[S]

    @staticmethod
    def __validate_multiple_episodes_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        series: SeriesDescription,
        season: SeasonDescription,
        contents: list[EpisodeContent],
    ) -> list["Validator.__ValidatorState[Validator.__Any1]"]:
        state: list[Validator.__ValidatorState[Validator.__Any1]] = [
            Validator.__ValidatorState([]) for _ in validators
        ]

        for content in contents:
            for i, validator in enumerate(validators):
                state[i].data.append(
                    validator.validate_episode(content, series=series, season=season),
                )

        return state

    @staticmethod
    def __validate_multiple_seasons_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        series: SeriesDescription,
        contents: list[SeasonContent],
    ) -> list["Validator.__ValidatorState[Validator.__Any2]"]:
        state: list[Validator.__ValidatorState[Validator.__Any2]] = [
            Validator.__ValidatorState([]) for _ in validators
        ]

        for content in contents:
            local_states: list[Validator.__ValidatorState[Validator.__Any1]] = (
                Validator.__validate_multiple_episodes_impl(
                    validators,
                    series=series,
                    season=content.description,
                    contents=content.episodes,
                )
            )

            for i, validator, local_state in zip(
                range(len(validators)),
                validators,
                local_states,
                strict=True,
            ):
                state[i].data.append(
                    validator.validate_season(
                        content,
                        series=series,
                        result=local_state.data,
                    ),
                )

        return state

    @staticmethod
    def __validate_multiple_series_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        contents: list[SeriesContent],
    ) -> list["Validator.__ValidatorState[Validator.__Any3]"]:
        state: list[Validator.__ValidatorState[Validator.__Any3]] = [
            Validator.__ValidatorState([]) for _ in validators
        ]

        for content in contents:
            local_states: list[Validator.__ValidatorState[Validator.__Any2]] = (
                Validator.__validate_multiple_seasons_impl(
                    validators,
                    series=content.description,
                    contents=content.seasons,
                )
            )

            for i, validator, local_state in zip(
                range(len(validators)),
                validators,
                local_states,
                strict=True,
            ):
                state[i].data.append(
                    validator.validate_series(
                        content,
                        result=local_state.data,
                    ),
                )

        return state

    @staticmethod
    def __validate_multiple_root_impl(
        validators: list["Validator[__Any1, __Any2, __Any3, __Any4]"],
        contents: list[Content],
    ) -> None:
        state: list[Validator.__ValidatorState[Validator.__Any4 | Validator.__Any3]] = [
            Validator.__ValidatorState([]) for _ in validators
        ]

        root_content: list[SeriesContent | CollectionContent] = []

        for content in contents:
            if isinstance(content, CollectionContent):
                local_states = Validator.__validate_multiple_series_impl(
                    validators,
                    content.series,
                )

                for i, validator, local_state in zip(
                    range(len(validators)),
                    validators,
                    local_states,
                    strict=True,
                ):
                    state[i].data.append(
                        validator.validate_collection(
                            content,
                            result=local_state.data,
                        ),
                    )

            elif isinstance(content, SeriesContent):
                local_states = Validator.__validate_multiple_series_impl(
                    validators,
                    [content],
                )

                for i, _validator, local_state in zip(
                    range(len(validators)),
                    validators,
                    local_states,
                    strict=True,
                ):
                    if len(local_state.data) != 1:
                        msg = "UNREACHABLE"
                        raise RuntimeError(msg)

                    state[i].data.append(local_state.data[0])

            elif isinstance(content, SeasonContent):
                msg = _("'SeasonContent' not valid for this state")
                raise TypeError(msg)
            elif isinstance(content, EpisodeContent):
                msg = _("'EpisodeContent' not valid for this state ")
                raise TypeError(msg)
            else:
                msg = _("invalid type for 'Content': {typ}").format(typ=type(content))
                raise TypeError(msg)

            root_content.append(content)

        for validator, root_state in zip(
            validators,
            state,
            strict=True,
        ):
            validator.validate_all(root_content, root_state.data)

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
        contents: list[SeriesContent | CollectionContent],
        result: list[S2D | CD],
    ) -> None: ...

    @staticmethod
    @abstractmethod
    def names() -> list[str]: ...

    @staticmethod
    @abstractmethod
    def validate_options(
        options: Optional[str],
    ) -> Result[Any, str]: ...

    @staticmethod
    @abstractmethod
    def from_params(
        params: ValidatorParams,
        options: Optional[str],
    ) -> Result["Validator[ED, SD, S2D, CD]", str]: ...

    @staticmethod
    @abstractmethod
    def is_default() -> bool: ...


# language validator, check if the language is a correct one
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
        self: Self,
        season: SeasonContent,
        series: SeriesDescription,
        result: list[None],
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
        contents: list[SeriesContent | CollectionContent],
        result: list[None],
    ) -> None:
        pass

    @staticmethod
    @override
    def names() -> list[str]:
        return ["language", "language_check", "valid_language"]

    @staticmethod
    @override
    def validate_options(
        options: Optional[str],
    ) -> Result[None, str]:
        if options is not None:
            return Err(f"No options supported, but got: {options}")

        return Ok(None)

    @staticmethod
    @override
    def from_params(
        params: ValidatorParams,
        options: Optional[str],
    ) -> Result["LanguageValidator", str]:
        result = LanguageValidator.validate_options(options)
        if result.err():
            return Err(result.as_err())

        return Ok(LanguageValidator(params.reporter, params.model_language))

    @staticmethod
    @override
    def is_default() -> bool:
        return True


# language consistency validator, check if the language is consistent across episodes and seasons
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
                    (series.description, season),
                    _("no language detected in season"),
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
        contents: list[SeriesContent | CollectionContent],
        result: list[None],
    ) -> None:
        pass

    @staticmethod
    @override
    def names() -> list[str]:
        return ["language consistency", "language_consistency"]

    @staticmethod
    @override
    def validate_options(
        options: Optional[str],
    ) -> Result[None, str]:
        if options is not None:
            return Err(f"No options supported, but got: {options}")

        return Ok(None)

    @staticmethod
    @override
    def from_params(
        params: ValidatorParams,
        options: Optional[str],
    ) -> Result["LanguageConsistencyValidator", str]:
        result = LanguageConsistencyValidator.validate_options(options)
        if result.err():
            return Err(result.as_err())

        return Ok(LanguageConsistencyValidator(params.reporter))

    @staticmethod
    @override
    def is_default() -> bool:
        return True


@dataclass
class TagOptions:
    strict: bool


# tags validator, checks, that every file has tags
class TagsValidator(Validator[None, None, None, None]):
    __options: TagOptions

    def __init__(
        self: Self,
        reporter: ValidatorReporter,
        options: TagOptions,
    ) -> None:
        super().__init__(reporter, "tags")

        self.__options = options

    @override
    def validate_episode(
        self: Self,
        episode: EpisodeContent,
        series: SeriesDescription,
        season: SeasonDescription,
    ) -> None:
        handle_result = get_tagger_for_file(episode.scanned_file.path)

        if handle_result.err():
            if self.__options.strict:
                self.emit_error(
                    episode.scanned_file.path,
                    _("File not tagged: can't get tagger handle: {err}").format(
                        err=handle_result.as_err(),
                    ),
                )
            return

        handle = handle_result.as_ok()

        manager = NoopManager()

        try:
            with handle.r_ctx(manager=manager) as ctx:
                tags = ctx.get_tags()
                if tags.uuid is None:
                    self.emit_error(
                        episode.scanned_file.path,
                        _("File not tagged"),
                    )
        except (RuntimeError, ValueError, TypeError) as err:
            self.emit_error(
                episode.scanned_file.path,
                _("File not tagged: {err}").format(err=str(err)),
            )

    @override
    def validate_season(
        self: Self,
        season: SeasonContent,
        series: SeriesDescription,
        result: list[None],
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
        contents: list[SeriesContent | CollectionContent],
        result: list[None],
    ) -> None:
        pass

    @staticmethod
    @override
    def names() -> list[str]:
        return ["tags", "has_tags"]

    @staticmethod
    @override
    def validate_options(
        options: Optional[str],
    ) -> Result[TagOptions, str]:
        if options is None:
            return Ok(TagOptions(strict=False))

        result = TagOptions(strict=False)
        for c in options:
            if c == "s":
                result.strict = True
            else:
                return Err(f"Invalid options flag: {c}")

        return Ok(result)

    @staticmethod
    @override
    def from_params(
        params: ValidatorParams,
        options: Optional[str],
    ) -> Result["TagsValidator", str]:
        result = TagsValidator.validate_options(options)
        if result.err():
            return Err(result.as_err())

        tag_options = result.as_ok()

        return Ok(TagsValidator(params.reporter, tag_options))

    @staticmethod
    @override
    def is_default() -> bool:
        return False


# TODO: metadata checks, check if no duplicates are found, missing episodes, missing seasons
# check language consistency

# TODO: find duplicates, e.g. simpson s32e10
# TODO: also display missing episodes / episodes with the "wrong" language etc


ValidatorGetCb = Callable[
    [ValidatorParams, Optional[str]],
    Result[Validator[Any, Any, Any, Any], str],
]

ValidatorValidateOptionsCb = Callable[
    [Optional[str]],
    Result[Any, str],
]


@dataclass
class ValidatorEntry:
    get: ValidatorGetCb
    validate_options: ValidatorValidateOptionsCb


def __get_all_validators_available_impl() -> dict[str, ValidatorEntry]:
    validators: dict[
        str,
        ValidatorEntry,
    ] = {}

    validator_classes: list[type[Validator[Any, Any, Any, Any]]] = [
        LanguageValidator,
        LanguageConsistencyValidator,
        TagsValidator,
    ]

    for validator_class in validator_classes:
        names = validator_class.names()
        for name in names:
            if validators.get(name, None) is not None:  # noqa: SIM910
                msg = f"Duplicate validator name: {name}"
                raise RuntimeError(msg)

            validators[name] = ValidatorEntry(
                validator_class.from_params,
                validator_class.validate_options,
            )

    return validators


__all_validators_available_impl: dict[str, ValidatorEntry] = (
    __get_all_validators_available_impl()
)


def __validator_check_impl(name: str, options: Optional[str]) -> Result[None, str]:
    if name not in __all_validators_available_impl:
        return Err(f"Not a valid validator name: {name}")

    entry = __all_validators_available_impl[name]

    options_res = entry.validate_options(options)

    if options_res.err():
        return Err(f"Invalid options for validator {name}: {options_res.as_err()}")

    return Ok(None)


__all_validator_names_available_impl: set[str] = set(
    __all_validators_available_impl.keys(),
)

validator_checks: ValidatorChecks = ValidatorChecks(
    check=__validator_check_impl,
    names=__all_validator_names_available_impl,
)


def __get_default_validators_impl(
    params: ValidatorParams,
) -> list[Validator[Any, Any, Any, Any]]:
    result: list[Validator[Any, Any, Any, Any]] = []
    for entry in __all_validators_available_impl.values():
        validator_res = entry.get(params, None)
        if validator_res.err():
            msg = f"Implementation error: no options should always return Ok, but got {validator_res.as_err()}"
            raise RuntimeError(msg)

        validator = validator_res.as_ok()

        if validator.is_default():
            result.append(validator)

    return result


def __get_all_validators_impl(
    params: ValidatorParams,
) -> list[Validator[Any, Any, Any, Any]]:
    return [
        entry.get(params, None).as_ok()
        for entry in __all_validators_available_impl.values()
    ]


def __get_validators_impl(
    params: ValidatorParams,
    filters: list[ValidatorFilter | SpecialFilter],
) -> list[Validator[Any, Any, Any, Any]]:
    if len(filters) == 0:
        return __get_default_validators_impl(params)

    def to_dict(
        lst: list[Validator[Any, Any, Any, Any]],
    ) -> dict[str, Validator[Any, Any, Any, Any]]:
        return {validator.name: validator for validator in lst}

    result: dict[str, Validator[Any, Any, Any, Any]] = {}

    for filter_item in filters:
        if isinstance(filter_item, SpecialFilter):
            if filter_item.name == ValidatorFilter.factory_name():
                match filter_item.type:
                    case SpecialFilterType.All:
                        result = to_dict(__get_all_validators_impl(params))
                    case SpecialFilterType.Empty:
                        result = {}
                    case SpecialFilterType.Default:
                        result = to_dict(__get_default_validators_impl(params))
                    case _:
                        assert_never(filter_item.type)

        elif isinstance(filter_item, ValidatorFilter):
            validator_entry = __all_validators_available_impl[filter_item.name]

            validator_res = validator_entry.get(params, filter_item.options)
            if validator_res.err():
                msg = f"Validator get error: {validator_res.as_err()}"
                raise RuntimeError(msg)

            validator = validator_res.as_ok()

            validator_name = validator.name

            if result.get(validator_name, None) is not None:
                msg = f"Validator is already present, duplicate is not allowed: {validator_name}"
                raise RuntimeError(msg)

            result[validator_name] = validator
        else:
            assert_never(filter_item)

    return list(result.values())


def get_validators(
    params: ValidatorParams,
    filters: list[Filter],
) -> list[Validator[Any, Any, Any, Any]]:
    validator_filter: list[ValidatorFilter | SpecialFilter] = [
        filter_val
        for filter_val in filters
        if isinstance(filter_val, (ValidatorFilter, SpecialFilter))
    ]

    return __get_validators_impl(params, validator_filter)
