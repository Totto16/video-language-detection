from dataclasses import dataclass, field
from logging import Logger
from typing import Annotated, Optional, Self, override

from apischema import alias, schema

from config import FinalConfig
from content.base_class import Content
from content.general import (
    NameParser,
    ScannedFile,
)
from content.language import Language
from helper.apischema import OneOf
from helper.translation import get_translator
from helper.tui import launch_tui

_ = get_translator()


@dataclass
class TatortOptions:
    config: FinalConfig


@dataclass(slots=True, repr=True)
class TatortEpisodeDescription:
    name: str
    season: int = field(metadata=schema(min=0))
    episode: int = field(metadata=schema(min=1))
    number_all: int = field(metadata=schema(min=1))

    def __str__(self: Self) -> str:
        return f"<Episode season: {self.season} episode: {self.episode} name: {self.name} number_all: {self.number_all}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class TatortContent(Content):
    __scanned_file: ScannedFile = field(metadata=alias("scanned_file"))
    __description: TatortEpisodeDescription = field(metadata=alias("description"))


AllContent = Annotated[
    TatortContent,
    OneOf,
]


class TatortNameParser(NameParser):

    def __init__(self: Self) -> None:
        super().__init__(Language("de", "German"))

    @override
    def parse_episode_name(self: Self, name: str) -> Optional[tuple[str, int, int]]:
        # TODO
        return None

    @override
    def parse_season_name(self: Self, name: str) -> Optional[tuple[int]]:
        # TODO
        return None

    @override
    def parse_series_name(self: Self, name: str) -> Optional[tuple[str, int]]:
        # TODO
        return None


def process_tatort(logger: Logger, options: TatortOptions) -> None:

    name_parser = TatortNameParser()

    launch_tui(
        logger=logger,
        config=options.config,
        name_parser=name_parser,
        all_content_type=AllContent,
    )
