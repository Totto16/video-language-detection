import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, assert_never, override

from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from helper.decorator import decorate_class
from helper.filter import (
    Filter,
    PathFilter,
    PathFilterType,
    SpecialFilter,
    SpecialFilterType,
)
from helper.translation import get_translator

_ = get_translator()


class ContentFilterStatus(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    @abstractmethod
    def should_ignore(self: Self) -> bool: ...


class NoContentFilterStatus(ContentFilterStatus):
    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def should_ignore(self: Self) -> bool:
        return False


class ContentFilterStatusMatch(ContentFilterStatus):
    __type: PathFilterType

    def __init__(self: Self, typ: PathFilterType) -> None:
        super().__init__()

        self.__type = typ

    @override
    def should_ignore(self: Self) -> bool:
        match self.__type:
            case PathFilterType.Negative:
                return True
            case PathFilterType.Positive:
                return False
            case _:
                assert_never(self.__type)


class ContentFilterStatusNoMatch(ContentFilterStatus):
    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def should_ignore(self: Self) -> bool:
        return True


@decorate_class(slots=True)
class ContentFilter:
    __filter: list[PathFilter]

    def __init__(self: Self, filters: list[PathFilter]) -> None:
        self.__filter = filters

    @staticmethod
    def __get_suitable_filters(
        filters: list[PathFilter | SpecialFilter],
    ) -> list[PathFilter]:
        if len(filters) == 0:
            return []

        result: set[PathFilter] = set()

        for filter_item in filters:
            if isinstance(filter_item, SpecialFilter):
                if filter_item.name == PathFilter.factory_name():
                    match filter_item.type:
                        case SpecialFilterType.All:
                            result = set()
                        case SpecialFilterType.Empty:
                            result = {
                                PathFilter(
                                    re.compile(".*"),
                                    typ=PathFilterType.Negative,
                                ),
                            }
                        case SpecialFilterType.Default:
                            result = set()
                        case _:
                            assert_never(filter_item.type)

            elif isinstance(filter_item, PathFilter):
                if filter_item in result:
                    msg = _(
                        "Path filter is already present, duplicate is not allowed: {filter}"  # noqa: COM812
                    ).format(filter=filter_item)
                    raise RuntimeError(msg)

                result.add(filter_item)
            else:
                assert_never(filter_item)

        return list(result)

    @staticmethod
    def from_filters(filters: list[Filter]) -> "ContentFilter":
        suitable_filter: list[PathFilter | SpecialFilter] = [
            filter_val
            for filter_val in filters
            if isinstance(filter_val, (PathFilter, SpecialFilter))
        ]

        resolved_filters = ContentFilter.__get_suitable_filters(suitable_filter)
        return ContentFilter(resolved_filters)

    def __dir_status(self: Self, directory: Path) -> ContentFilterStatus:
        if not directory.is_dir():
            msg = f"Not a dir: {directory}"
            raise RuntimeError(msg)

        for flt in self.__filter:
            if flt.pattern.match(str(object=directory.absolute())) is not None:
                return ContentFilterStatusMatch(flt.type)

        return ContentFilterStatusNoMatch()

    def __file_status(self: Self, file: Path) -> ContentFilterStatus:
        if not file.is_file():
            msg = f"Not a file: {file}"
            raise RuntimeError(msg)

        for flt in self.__filter:
            if flt.pattern.match(str(object=file.absolute())) is not None:
                return ContentFilterStatusMatch(flt.type)

        return ContentFilterStatusNoMatch()

    def collection_status(
        self: Self,
        collection: CollectionContent,
    ) -> ContentFilterStatus:

        if len(self.__filter) == 0:
            return NoContentFilterStatus()

        status = self.__dir_status(collection.scanned_file.path)

        if not isinstance(status, ContentFilterStatusNoMatch):
            return status

        # TODO: use collection filter

        return ContentFilterStatusNoMatch()

    def series_status(self: Self, series: SeriesContent) -> ContentFilterStatus:

        if len(self.__filter) == 0:
            return NoContentFilterStatus()

        status = self.__dir_status(series.scanned_file.path)

        if not isinstance(status, ContentFilterStatusNoMatch):
            return status

        # TODO: use series filter

        return ContentFilterStatusNoMatch()

    def season_status(self: Self, series: SeasonContent) -> ContentFilterStatus:

        if len(self.__filter) == 0:
            return NoContentFilterStatus()

        status = self.__dir_status(series.scanned_file.path)

        if not isinstance(status, ContentFilterStatusNoMatch):
            return status

        # TODO: use season filter

        return ContentFilterStatusNoMatch()

    def episode_status(self: Self, series: EpisodeContent) -> ContentFilterStatus:

        if len(self.__filter) == 0:
            return NoContentFilterStatus()

        status = self.__file_status(series.scanned_file.path)

        if not isinstance(status, ContentFilterStatusNoMatch):
            return status

        # TODO: use episode filter

        return ContentFilterStatusNoMatch()
