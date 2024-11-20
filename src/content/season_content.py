from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
    cast,
    override,
)

from apischema import alias, schema

from content.base_class import (
    CallbackTuple,
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
    narrow_type,
)


class SeasonContentDict(ContentDict):
    description: SeasonDescription
    episodes: list[EpisodeContent]


@schema(extra=narrow_type(("type", Literal[ContentType.season])))
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
            msg = f"Couldn't get SeasonDescription from '{path}'"
            raise NameError(msg)

        return SeasonContent(ContentType.season, scanned_file, description, [])

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
    def summary(self: Self, *, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed=detailed)
        for episode in self.__episodes:
            summary.combine_episodes(
                self.description,
                episode.summary(detailed=detailed),
            )

        return summary

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, CallbackTuple],
        *,
        parent_folders: list[str],
        rescan: bool = False,
    ) -> None:
        if not rescan:
            contents: list[Content] = process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
            )
            for content in contents:
                if isinstance(content, EpisodeContent):
                    self.__episodes.append(content)
                else:
                    msg = f"No child with class '{content.__class__.__name__}' is possible in SeasonContent"
                    raise TypeError(msg)
        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                rescan=cast(list[Content], self.__episodes),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__episodes):
                if not isinstance(content, EpisodeContent):
                    msg = f"No child with class '{content.__class__.__name__}' is possible in SeasonContent"
                    raise TypeError(msg)
