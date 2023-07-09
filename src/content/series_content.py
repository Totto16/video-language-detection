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
    ContentType,
    NameParser,
    ScannedFile,
    SeriesDescription,
    Summary,
)
from content.season_content import SeasonContent
from enlighten import Manager
from typing_extensions import override


class SeriesContentDict(ContentDict):
    description: SeriesDescription
    seasons: list[SeasonContent]


class SeriesContent(Content):
    __description: SeriesDescription
    __seasons: list[SeasonContent]

    @staticmethod
    def from_path(
        path: Path,
        scanned_file: ScannedFile,
        name_parser: NameParser,
    ) -> "SeriesContent":
        description: Optional[SeriesDescription] = SeriesContent.parse_description(
            path.name,
            name_parser,
        )
        if description is None:
            raise NameError(f"Couldn't get SeriesDescription from '{path}'")

        return SeriesContent(scanned_file, description, [])

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: SeriesDescription,
        seasons: list[SeasonContent],
    ) -> None:
        super().__init__(ContentType.series, scanned_file)

        self.__description = description
        self.__seasons = seasons

    @property
    def description(self: Self) -> SeriesDescription:
        return self.__description

    @staticmethod
    def is_valid_name(
        name: str,
        name_parser: NameParser,
    ) -> bool:
        return SeriesContent.parse_description(name, name_parser) is not None

    @staticmethod
    def parse_description(
        name: str,
        name_parser: NameParser,
    ) -> Optional[SeriesDescription]:
        result = name_parser.parse_series_name(name)
        if result is None:
            return None

        name, year = result

        return SeriesDescription(name, year)

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for season in self.__seasons:
            summary.combine_seasons(self.description, season.summary(detailed))

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
                if isinstance(content, SeasonContent):
                    self.__seasons.append(content)
                else:
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeriesContent",
                    )

        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
                rescan=cast(list[Content], self.__seasons),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__seasons):
                if not isinstance(content, SeasonContent):
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeriesContent",
                    )

    @override
    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["seasons"] = self.__seasons
        return as_dict

    @staticmethod
    def from_dict(dct: SeriesContentDict) -> "SeriesContent":
        return SeriesContent(dct["scanned_file"], dct["description"], dct["seasons"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()
