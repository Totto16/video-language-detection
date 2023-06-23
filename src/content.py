#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum
import hashlib
from os import path
from pathlib import Path
from typing import Optional
from typing_extensions import override
import re
from classifier import Language


class ScannedFileType(Enum):
    file = "file"
    folder = "folder"


class ContentType(Enum):
    series = "series"
    season = "season"
    episode = "episode"
    collection = "collection"


class MissingOverrideError(RuntimeError):
    pass


@dataclass
class Stats:
    checksum: Optional[str]
    mtime: float

    @staticmethod
    def hash_file(file_path: Path) -> str:
        if file_path.is_dir():
            raise RuntimeError("Can't take checksum of directory")

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as file:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: file.read(4096), b""):
                sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

    @staticmethod
    def from_file(file_path: Path, file_type: ScannedFileType) -> "Stats":
        checksum = (
            None if file_type == ScannedFileType.folder else Stats.hash_file(file_path)
        )

        mtime = Path(file_path).stat().st_mtime

        return Stats(checksum=checksum, mtime=mtime)


@dataclass
class ScannedFile:
    parents: list[str]
    name: str  # id
    type: ScannedFileType
    stats: Stats

    @staticmethod
    def from_scan(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> "ScannedFile":
        if len(parent_folders) > 3:
            raise RuntimeError(
                "No more than 3 parent folders are allowed: [collection] -> series -> season"
            )

        name = path.basename(file_path)

        stats = Stats.from_file(file_path, file_type)

        return ScannedFile(
            parents=parent_folders,
            name=name,
            type=file_type,
            stats=stats,
        )


class Content:
    __scanned_file: ScannedFile
    __type: ContentType

    def __init__(self, type: ContentType) -> None:
        self.__type: ContentType = type

    def languages(self) -> list[Language]:
        raise MissingOverrideError()

    @staticmethod
    def from_scan(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> Optional["Content"]:
        name = file_path.name

        # [collection] -> series -> season -> file_name
        parents: list[str] = [*parent_folders, name]

        top_is_collection: bool = not SeriesContent.is_valid_name(parents[0])

        if len(parent_folders) == 4:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return EpisodeContent(file_path)
        elif len(parents) == 3 and not top_is_collection:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return EpisodeContent(file_path)
        elif len(parents) == 3 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeasonContent(file_path)
        elif len(parents) == 2 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeasonContent(file_path)
        elif len(parents) == 2 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeriesContent(file_path)
        elif len(parents) == 1 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeriesContent(file_path)
        else:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return CollectionContent(file_path)


def parse_int_safely(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None


@dataclass
class EpisodeDescription:
    name: str
    season: int
    episode: int


class EpisodeContent(Content):
    __description: EpisodeDescription
    __language: Language

    def __init__(self, path: Path) -> None:
        super().__init__(ContentType.episode)
        description: Optional[EpisodeDescription] = EpisodeContent.parse_description(
            path.name
        )
        if description is None:
            raise RuntimeError(f"Couldn't get EpisodeDescription from {path}")

        self.__description = description
        self.__language = Language("un", "Unknown")

    @property
    def description(self) -> EpisodeDescription:
        return self.__description

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return SeriesContent.parse_description(name) is not None

    @staticmethod
    def parse_description(name: str) -> Optional[EpisodeDescription]:
        match = re.search(r"Episode (\d{2}) - (.*) \[S(\d{2})E(\d{2})\]\.(.*)", name)
        if match is None:
            return None

        groups = match.groups()
        if len(groups) != 5:
            return None

        _episode_num, name, _season, _episode, _extension = groups
        season = parse_int_safely(_season)
        if season is None:
            return None

        episode = parse_int_safely(_episode)
        if episode is None:
            return None

        return EpisodeDescription(name, season, episode)

    @override
    def languages(self) -> list[Language]:
        return [self.__language]


@dataclass
class SeasonDescription:
    season: int


class SeasonContent(Content):
    __description: SeasonDescription
    __episodes: list[EpisodeContent]

    def __init__(self, path: Path) -> None:
        super().__init__(ContentType.season)
        description: Optional[SeasonDescription] = SeasonContent.parse_description(
            path.name
        )
        if description is None:
            raise RuntimeError(f"Couldn't get EpisodeDescription from {path}")

        self.__description = description
        self.__episodes = []

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return SeasonContent.parse_description(name) is not None

    @staticmethod
    def parse_description(name: str) -> Optional[SeasonDescription]:
        match = re.search(r"Episode (\d{2}) - (.*) \[S(\d{2})E(\d{2})\]\.(.*)", name)
        if match is None:
            return None

        groups = match.groups()
        if len(groups) != 5:
            return None

        _episode_num, name, _season, _episode, _extension = groups
        season = parse_int_safely(_season)
        if season is None:
            return None

        episode = parse_int_safely(_episode)
        if episode is None:
            return None

        return SeasonDescription(season)

    @property
    def description(self) -> SeasonDescription:
        return self.__description

    @override
    def languages(self) -> list[Language]:
        languages: list[Language] = []
        for episode in self.__episodes:
            for language in episode.languages():
                if language not in languages:
                    languages.append(language)

        return languages


@dataclass
class SeriesDescription:
    name: str
    year: int


class SeriesContent(Content):
    __description: SeriesDescription
    __seasons: list[SeasonContent]

    def __init__(self, path: Path) -> None:
        super().__init__(ContentType.series)
        description: Optional[SeriesDescription] = SeriesContent.parse_description(
            path.name
        )
        if description is None:
            raise RuntimeError(f"Couldn't get SeriesDescription from {path}")

        self.__description = description
        self.__seasons = []

    @property
    def description(self) -> SeriesDescription:
        return self.__description

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return SeriesContent.parse_description(name) is not None

    @staticmethod
    def parse_description(name: str) -> Optional[SeriesDescription]:
        match = re.search(r"(.*) \((\d{4})\)", name)
        if match is None:
            return None

        groups = match.groups()
        if len(groups) != 2:
            return None

        name, _year = groups
        year = parse_int_safely(_year)
        if year is None:
            return None

        return SeriesDescription(name, year)

    @override
    def languages(self) -> list[Language]:
        languages: list[Language] = []
        for season in self.__seasons:
            for language in season.languages():
                if language not in languages:
                    languages.append(language)

        return languages


class CollectionContent(Content):
    __name: str
    __series: list[SeriesContent]

    def __init__(self, path: Path) -> None:
        super().__init__(ContentType.collection)
        self.__name = path.name
        self.__series = []

    @property
    def description(self) -> str:
        return self.__name

    @override
    def languages(self) -> list[Language]:
        languages: list[Language] = []
        for serie in self.__series:
            for language in serie.languages():
                if language not in languages:
                    languages.append(language)

        return languages
