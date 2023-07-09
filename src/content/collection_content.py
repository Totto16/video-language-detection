#!/usr/bin/env python3


from json import JSONEncoder
from pathlib import Path
from typing import (
    Any,
    Optional,
    Self,
    cast,
)

from classifier import Classifier
from content.base_class import Content, ContentCharacteristic, ContentDict, process_folder
from content.general import (
    Callback,
    CollectionDescription,
    ContentType,
    NameParser,
    ScannedFile,
    Summary,
)
from content.series_content import SeriesContent
from enlighten import Manager
from typing_extensions import override


class CollectionContentDict(ContentDict):
    description: CollectionDescription
    series: list[SeriesContent]


class CollectionContent(Content):
    __description: CollectionDescription
    __series: list[SeriesContent]

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "CollectionContent":
        return CollectionContent(scanned_file, path.name, [])

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: CollectionDescription,
        series: list[SeriesContent],
    ) -> None:
        super().__init__(ContentType.collection, scanned_file)

        self.__description = description
        self.__series = series

    @property
    def description(self: Self) -> str:
        return self.__description

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for serie in self.__series:
            summary.combine_series(self.description, serie.summary(detailed))

        return summary

    @override
    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = self.__description
        as_dict["series"] = self.__series
        return as_dict

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

    @staticmethod
    def from_dict(dct: CollectionContentDict) -> "CollectionContent":
        return CollectionContent(dct["scanned_file"], dct["description"], dct["series"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()
