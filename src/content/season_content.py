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



class SeasonContent(Content):
    __description: SeasonDescription
    __episodes: list[EpisodeContent]

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

        return SeasonContent(scanned_file, description, [])

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: SeasonDescription,
        episodes: list[EpisodeContent],
    ) -> None:
        super().__init__(ContentType.season, scanned_file)

        self.__description = description
        self.__episodes = episodes

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

    @override
    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["episodes"] = self.__episodes
        return as_dict

    @staticmethod
    def from_dict(dct: SeasonContentDict) -> "SeasonContent":
        return SeasonContent(dct["scanned_file"], dct["description"], dct["episodes"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()
