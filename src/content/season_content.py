from dataclasses import dataclass, field
from logging import Logger
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
from content.metadata.metadata import HandlesType, MetadataHandle
from content.shared import ScanType
from helper.log import get_logger

logger: Logger = get_logger()


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
            raise NameError(msg, name="SeasonDescription")

        return SeasonContent(
            ContentType.season,
            scanned_file,
            None,
            description,
            [],
        )

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
        return Summary.construct_for_season(
            self.metadata,
            self.description,
            (episode.summary(detailed=detailed) for episode in self.__episodes),
            detailed=detailed,
        )

    def __get_handle(
        self: Self,
        handles: HandlesType,
    ) -> Optional[MetadataHandle]:
        if handles is None:
            return None

        if len(handles) != 1:
            msg = f"Length of handles is invalid, expected 1 but got {len(handles)}"
            logger.warning(msg)
            return None

        return handles[0]

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, CallbackTuple],
        *,
        handles: HandlesType,
        parent_folders: list[str],
        rescan: bool = False,
    ) -> None:
        _, scanner = callback.get_saved()

        series_handle = self.__get_handle(handles=handles)

        if not rescan:
            if (
                series_handle is not None
                and self.metadata is None
                and scanner.should_scan_metadata(
                    ScanType.first_scan,
                    self.metadata,
                )
            ):
                self._metadata = scanner.metadata_scanner.get_season_metadata(
                    series_handle,
                    self.description.season,
                )

            new_handles = self._get_new_handles(handles)

            contents: list[Content] = process_folder(
                self.scanned_file.path,
                callback=callback,
                handles=new_handles,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
            )
            for content in contents:
                if isinstance(content, EpisodeContent):
                    self.__episodes.append(content)
                else:
                    msg = f"No child with class '{content.__class__.__name__}' is possible in SeasonContent"
                    raise TypeError(msg)
            return

        if series_handle is not None and scanner.should_scan_metadata(
            ScanType.rescan,
            self.metadata,
        ):
            self._metadata = scanner.metadata_scanner.get_season_metadata(
                series_handle,
                self.description.season,
            )

        new_handles = self._get_new_handles(handles)

        # no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
        process_folder(
            self.scanned_file.path,
            callback=callback,
            handles=new_handles,
            parent_folders=[*parent_folders, self.scanned_file.path.name],
            parent_type=self.type,
            rescan=cast(list[Content], self.__episodes),
        )

        # since some are added unchecked, check again now!
        for content in cast(list[Content], self.__episodes):
            if not isinstance(content, EpisodeContent):
                msg = f"No child with class '{content.__class__.__name__}' is possible in SeasonContent"
                raise TypeError(msg)
