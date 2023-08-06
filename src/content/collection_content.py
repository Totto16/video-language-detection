#!/usr/bin/env python3


from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    Self,
    cast,
)

from apischema import alias
from classifier import Classifier
from enlighten import Manager
from typing_extensions import override

from content.base_class import (
    Content,
    ContentCharacteristic,
    ContentDict,
    process_folder,
)
from content.general import (
    Callback,
    CollectionDescription,
    ContentType,
    NameParser,
    ScannedFile,
    Summary,
)
from content.series_content import SeriesContent


class CollectionContentDict(ContentDict):
    description: CollectionDescription
    series: list[SeriesContent]


@dataclass(slots=True, repr=True)
class CollectionContent(Content):
    __type: Literal[ContentType.collection] = field(metadata=alias("type"))
    __description: CollectionDescription = field(metadata=alias("description"))
    __series: list[SeriesContent] = field(metadata=alias("series"))

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "CollectionContent":
        return CollectionContent(ContentType.collection, scanned_file, path.name, [])

    @property
    def description(self: Self) -> str:
        return self.__description

    @property
    def series(self: Self) -> list[SeriesContent]:
        return self.__series

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for serie in self.__series:
            summary.combine_series(self.description, serie.summary(detailed))

        return summary

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, Manager],
        name_parser: NameParser,
        *,
        parent_folders: list[str],
        classifier: Classifier,
        rescan: bool = False,
    ) -> None:
        if not rescan:
            contents: list[Content] = process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
            )
            for content in contents:
                if isinstance(content, SeriesContent):
                    self.__series.append(content)
                else:
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in CollectionContent",
                    )

        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
                rescan=cast(list[Content], self.__series),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__series):
                if not isinstance(content, SeriesContent):
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in CollectionContent",
                    )
