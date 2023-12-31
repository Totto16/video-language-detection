from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
    cast,
)

from apischema import alias, schema
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
    ContentType,
    NameParser,
    ScannedFile,
    SeriesDescription,
    Summary,
    narrow_type,
)
from content.season_content import SeasonContent


class SeriesContentDict(ContentDict):
    description: SeriesDescription
    seasons: list[SeasonContent]




@schema(extra=narrow_type(("type", Literal[ContentType.series])))
@dataclass(slots=True, repr=True)
class SeriesContent(Content):
    __description: SeriesDescription = field(metadata=alias("description"))
    __seasons: list[SeasonContent] = field(metadata=alias("seasons"))

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
            msg = f"Couldn't get SeriesDescription from '{path}'"
            raise NameError(msg)

        return SeriesContent(ContentType.series, scanned_file, description, [])

    @property
    def description(self: Self) -> SeriesDescription:
        return self.__description

    @property
    def seasons(self: Self) -> list[SeasonContent]:
        return self.__seasons

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
    def summary(self: Self, *, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed=detailed)
        for season in self.__seasons:
            summary.combine_seasons(self.description, season.summary(detailed=detailed))

        return summary

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, Manager],
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
                if isinstance(content, SeasonContent):
                    self.__seasons.append(content)
                else:
                    msg = f"No child with class '{content.__class__.__name__}' is possible in SeriesContent"
                    raise TypeError(msg)

        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                rescan=cast(list[Content], self.__seasons),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__seasons):
                if not isinstance(content, SeasonContent):
                    msg = f"No child with class '{content.__class__.__name__}' is possible in SeriesContent"
                    raise TypeError(msg)
