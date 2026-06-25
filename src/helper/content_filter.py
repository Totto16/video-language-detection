import re
from pathlib import Path
from typing import Self, assert_never

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

    def __should_ignore_dir(self: Self, directory: Path) -> bool:
        if not directory.is_dir():
            msg = f"Not a dir: {directory}"
            raise RuntimeError(msg)

        for flt in self.__filter:
            if flt.pattern.match(str(object=directory.absolute())) is not None:
                return flt.type == PathFilterType.Negative

        return True

    def __should_ignore_file(self: Self, file: Path) -> bool:
        if not file.is_file():
            msg = f"Not a file: {file}"
            raise RuntimeError(msg)

        for flt in self.__filter:
            if flt.pattern.match(str(object=file.absolute())) is not None:
                return flt.type == PathFilterType.Negative

        return True

    def should_ignore_collection(self: Self, collection: CollectionContent) -> bool:

        if len(self.__filter) == 0:
            return False

        if self.__should_ignore_dir(collection.scanned_file.path):
            return True

        # TODO: use collection filter

        return False

    def should_ignore_series(self: Self, series: SeriesContent) -> bool:

        if len(self.__filter) == 0:
            return False

        if self.__should_ignore_dir(series.scanned_file.path):
            return True

        # TODO: use series filter

        return False

    def should_ignore_season(self: Self, series: SeasonContent) -> bool:

        if len(self.__filter) == 0:
            return False

        if self.__should_ignore_dir(series.scanned_file.path):
            return True

        # TODO: use series filter

        return False

    def should_ignore_episode(self: Self, series: EpisodeContent) -> bool:

        if len(self.__filter) == 0:
            return False

        if self.__should_ignore_file(series.scanned_file.path):
            return True

        # TODO: use series filter

        return False
