import re
from typing import Self, assert_never

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
