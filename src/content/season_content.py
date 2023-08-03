#!/usr/bin/env python3


from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Optional,
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
from content.episode_content import EpisodeContent
from content.general import (
    Callback,
    ContentType,
    NameParser,
    ScannedFile,
    SeasonDescription,
    Summary,
)


class SeasonContentDict(ContentDict):
    description: SeasonDescription
    episodes: list[EpisodeContent]


@dataclass(slots=True, repr=True)
class SeasonContent(Content):
    __description: SeasonDescription = field(metadata=alias("description"))
    __episodes: list[EpisodeContent] = field(metadata=alias("episodes"))

    @staticmethod
    def from_path(
        path: Path,
        scanned_file: ScannedFile,
        name_parser: NameParser,
    ) -> "SeasonContent":
        description: Optional[SeasonDescription] = SeasonContent.parse_description(
            path.name,
            name_parser,
        )
        if description is None:
            raise NameError(f"Couldn't get SeasonDescription from '{path}'")

        return SeasonContent(ContentType.series, scanned_file, description, [])

    @staticmethod
    def is_valid_name(
        name: str,
        name_parser: NameParser,
    ) -> bool:
        return SeasonContent.parse_description(name, name_parser) is not None

    @staticmethod
    def parse_description(
        name: str,
        name_parser: NameParser,
    ) -> Optional[SeasonDescription]:
        result = name_parser.parse_season_name(name)
        if result is None:
            return None

        (season,) = result

        return SeasonDescription(season)

    @property
    def description(self: Self) -> SeasonDescription:
        return self.__description

    @property
    def episodes(self: Self) -> list[EpisodeContent]:
        return self.__episodes

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for episode in self.__episodes:
            summary.combine_episodes(self.description, episode.summary(detailed))

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
                if isinstance(content, EpisodeContent):
                    self.__episodes.append(content)
                else:
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeasonContent",
                    )
        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
                rescan=cast(list[Content], self.__episodes),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__episodes):
                if not isinstance(content, EpisodeContent):
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeasonContent",
                    )
