#!/usr/bin/env python3

from dataclasses import dataclass
import dataclasses
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Optional, Callable, TypeVar, cast
from typing_extensions import override
import re
from classifier import Classifier, Language, WAVFile
from os import listdir
from json import JSONDecoder, JSONEncoder


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


# GLOBAL
classifier: Classifier


def parse_int_safely(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None


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
        checksum: Optional[str] = (
            None if file_type == ScannedFileType.folder else Stats.hash_file(file_path)
        )

        mtime: float = Path(file_path).stat().st_mtime

        return Stats(checksum=checksum, mtime=mtime)

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = {
            "checksum": self.checksum,
            "mtime": self.mtime,
        }
        return as_dict

    @staticmethod
    def from_dict(dct: dict[str, Any]) -> "Stats":
        return Stats(checksum=dct.get("checksum"), mtime=dct["mtime"])

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ScannedFile:
    path: Path
    parents: list[str]
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

        stats: Stats = Stats.from_file(file_path, file_type)

        return ScannedFile(
            path=file_path,
            parents=parent_folders,
            type=file_type,
            stats=stats,
        )

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = {
            "path": str(self.path.absolute()),
            "parents": self.parents,
            "type": self.type,
            "stats": self.stats,
        }
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict)

    def __repr__(self) -> str:
        return self.__str__()


T = TypeVar("T")


def process_folder(
    directory: Path,
    process_fn: Callable[[Path, ScannedFileType, list[str]], T],
    *,
    ignore_fn: Callable[
        [Path, ScannedFileType, list[str]], bool
    ] = lambda x, y, z: False,
    parent_folders: list[str] = [],
) -> list[T]:
    results: list[T] = []
    for file in listdir(directory):
        file_path: Path = Path(directory) / file

        file_type: ScannedFileType = (
            ScannedFileType.folder if file_path.is_dir() else ScannedFileType.file
        )
        should_ignore: bool = ignore_fn(file_path, file_type, parent_folders)
        if should_ignore:
            continue

        result: T = process_fn(file_path, file_type, parent_folders)
        results.append(result)

    return results


class Content:
    __scanned_file: ScannedFile
    __type: ContentType

    def __init__(self, type: ContentType, scanned_file: ScannedFile) -> None:
        self.__type = type
        self.__scanned_file = scanned_file

    def languages(self) -> list[Language]:
        raise MissingOverrideError()

    @property
    def type(self) -> ContentType:
        return self.__type

    @property
    def scanned_file(self) -> ScannedFile:
        return self.__scanned_file

    @staticmethod
    def from_scan(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> Optional["Content"]:
        scanned_file: ScannedFile = ScannedFile.from_scan(
            file_path, file_type, parent_folders
        )

        name = file_path.name

        # [collection] -> series -> season -> file_name
        parents: list[str] = [*parent_folders, name]

        top_is_collection: bool = not SeriesContent.is_valid_name(parents[0])

        if len(parent_folders) == 4:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return EpisodeContent(file_path, scanned_file)
        elif len(parents) == 3 and not top_is_collection:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return EpisodeContent(file_path, scanned_file)
        elif len(parents) == 3 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeasonContent(file_path, scanned_file)
        elif len(parents) == 2 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeasonContent(file_path, scanned_file)
        elif len(parents) == 2 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeriesContent(file_path, scanned_file)
        elif len(parents) == 1 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return SeriesContent(file_path, scanned_file)
        else:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting!"
                )

            return CollectionContent(file_path, scanned_file)

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = {
            "scanned_file": self.__scanned_file.as_dict(),
            "type": self.__type,
        }
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class EpisodeDescription:
    name: str
    season: int
    episode: int


class EpisodeContent(Content):
    __description: EpisodeDescription
    __language: Language

    def __init__(self, path: Path, scanned_file: ScannedFile) -> None:
        super().__init__(ContentType.episode, scanned_file)
        description: Optional[EpisodeDescription] = EpisodeContent.parse_description(
            path.name
        )
        if description is None:
            raise RuntimeError(f"Couldn't get EpisodeDescription from {path}")

        self.__description = description
        self.__language = self.__get_language()

    @property
    def description(self) -> EpisodeDescription:
        return self.__description

    def __get_language(self) -> Language:
        wav_file = WAVFile(self.scanned_file.path)

        language, accuracy = classifier.predict(wav_file)
        print(language, accuracy)
        return language

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

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict()
        as_dict["description"] = self.__description
        as_dict["language"] = self.__language
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SeasonDescription:
    season: int


class SeasonContent(Content):
    __description: SeasonDescription
    __episodes: list[EpisodeContent]

    def __init__(self, path: Path, scanned_file: ScannedFile) -> None:
        super().__init__(ContentType.season, scanned_file)
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

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict()
        as_dict["description"] = self.__description
        as_dict["episodes"] = self.__episodes
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SeriesDescription:
    name: str
    year: int


class SeriesContent(Content):
    __description: SeriesDescription
    __seasons: list[SeasonContent]

    def __init__(self, path: Path, scanned_file: ScannedFile) -> None:
        super().__init__(ContentType.series, scanned_file)
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

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict()
        as_dict["description"] = self.__description
        as_dict["seasons"] = self.__seasons
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


class CollectionContent(Content):
    __description: str
    __series: list[SeriesContent]

    def __init__(self, path: Path, scanned_file: ScannedFile) -> None:
        super().__init__(ContentType.collection, scanned_file)
        self.__description = path.name
        self.__series = []

    @property
    def description(self) -> str:
        return self.__description

    @override
    def languages(self) -> list[Language]:
        languages: list[Language] = []
        for serie in self.__series:
            for language in serie.languages():
                if language not in languages:
                    languages.append(language)

        return languages

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict()
        as_dict["description"] = self.__description
        as_dict["series"] = self.__series
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


VALUE_KEY: str = "__value__"
TYPE_KEY: str = "__type__"


class Encoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Content):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict()}
        elif isinstance(o, ScannedFile):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict()}
        elif isinstance(o, Path):
            return {TYPE_KEY: "Path", VALUE_KEY: str(o.absolute())}
        elif isinstance(o, Enum):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.name}
        elif isinstance(o, Stats):
            return {TYPE_KEY: "Stats", VALUE_KEY: o.as_dict()}
        elif dataclasses.is_dataclass(o):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: dataclasses.asdict(o)}
        else:
            return super().default(o)


class Decoder(JSONDecoder):
    def __init__(self) -> None:
        def object_hook(dct: dict[str, Any]) -> Any:
            type: Optional[str] = cast(Optional[str], dct.get(TYPE_KEY))
            if type is None:
                return dct
                # raise TypeError(f"Object has no type key: {dct}")

            value: Any = dct[VALUE_KEY]
            match type:
                case "ScannedFileType":
                    return ScannedFileType(value)
                case "Stats":
                    return Stats.from_dict(value)
                case _:
                    raise TypeError(
                        f"Object of type {type} is not JSON de-serializable"
                    )

        super().__init__(object_hook=object_hook)
