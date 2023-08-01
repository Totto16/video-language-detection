#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import JSONEncoder
from pathlib import Path
from typing import Any, Generic, Optional, Self, TypedDict, TypeVar

from classifier import Language
from enlighten import Manager


class ScannedFileType(Enum):
    file = "file"
    folder = "folder"

    @staticmethod
    def from_path(path: Path) -> "ScannedFileType":
        return ScannedFileType.folder if path.is_dir() else ScannedFileType.file

    def __str__(self: Self) -> str:
        return f"<ScannedFileType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class ContentType(Enum):
    series = "series"
    season = "season"
    episode = "episode"
    collection = "collection"

    def __str__(self: Self) -> str:
        return f"<ContentType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class MissingOverrideError(RuntimeError):
    pass


class StatsDict(TypedDict):
    checksum: Optional[str]
    mtime: float


CHECKSUM_BAR_FORMAT = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


@dataclass
class EpisodeDescription:
    name: str
    season: int
    episode: int

    def __str__(self: Self) -> str:
        return (
            f"<Episode season: {self.season} episode: {self.episode} name: {self.name}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class SeriesDescription:
    name: str
    year: int

    def __str__(self: Self) -> str:
        return f"<Series name: {self.name} year: {self.year}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class SeasonDescription:
    season: int

    def __str__(self: Self) -> str:
        return f"<Season season: {self.season}>y"

    def __repr__(self: Self) -> str:
        return str(self)


CollectionDescription = str

IdentifierDescription = (
    tuple[EpisodeDescription]
    | tuple[SeasonDescription, EpisodeDescription]
    | tuple[SeriesDescription, SeasonDescription, EpisodeDescription]
    | tuple[
        CollectionDescription,
        SeriesDescription,
        SeasonDescription,
        EpisodeDescription,
    ]
)


LanguageDict = dict[Language, int]


# TODO
class Summary:
    __complete: bool
    __detailed: bool

    __descriptions: list[IdentifierDescription]

    __languages: LanguageDict
    __duplicates: list[IdentifierDescription]
    __missing: list[IdentifierDescription]

    def __init__(
        self: Self,
        languages: list[Language],
        descriptions: list[IdentifierDescription],
        detailed: bool = False,
    ) -> None:
        def get_dict(language: Language) -> LanguageDict:
            dct: LanguageDict = {}
            dct[language] = 1
            return dct

        self.__languages = Summary.combine_langauge_dicts(
            [get_dict(language) for language in languages],
        )
        self.__duplicates = []
        self.__missing = []
        self.__complete = False
        self.__detailed = detailed
        self.__descriptions = descriptions

    @property
    def complete(self: Self) -> bool:
        return self.__complete

    @property
    def detailed(self: Self) -> bool:
        return self.__detailed

    @property
    def duplicates(self: Self) -> list[IdentifierDescription]:
        return self.__duplicates

    @property
    def missing(self: Self) -> list[IdentifierDescription]:
        return self.__missing

    @staticmethod
    def from_single(
        language: Language,
        description: EpisodeDescription,
        detailed: bool,
    ) -> "Summary":
        return Summary([language], [(description,)], detailed)

    @staticmethod
    def empty(detailed: bool) -> "Summary":
        return Summary([], [], detailed)

    def combine_episodes(
        self: Self,
        description: SeasonDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if isinstance(desc[0], EpisodeDescription):
                self.__descriptions.append((description, desc[0]))  # noqa: PERF401

        self.__languages = Summary.combine_langauge_dicts(
            [self.__languages, summary.languages],
        )

    def combine_seasons(
        self: Self,
        description: SeriesDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if (
                len(desc) == 2
                and isinstance(desc[0], SeasonDescription)
                and isinstance(desc[1], EpisodeDescription)
            ):
                self.__descriptions.append(  # noqa: PERF401
                    (description, desc[0], desc[1]),
                )

        self.__languages = Summary.combine_langauge_dicts(
            [self.__languages, summary.languages],
        )

    def combine_series(
        self: Self,
        description: CollectionDescription,
        summary: "Summary",
    ) -> None:
        for desc in summary.descriptions:
            if (
                len(desc) == 3
                and isinstance(desc[0], SeriesDescription)
                and isinstance(desc[1], SeasonDescription)
                and isinstance(desc[2], EpisodeDescription)
            ):
                self.__descriptions.append(  # noqa: PERF401
                    (description, desc[0], desc[1], desc[2]),
                )

        self.__languages = Summary.combine_langauge_dicts(
            [self.__languages, summary.languages],
        )

    @staticmethod
    def combine_langauge_dicts(inp: list[LanguageDict]) -> LanguageDict:
        dct: LanguageDict = {}
        for input_dict in inp:
            for language, amount in input_dict.items():
                if dct.get(language) is None:
                    dct[language] = 0

                dct[language] += amount

        return dct

    @property
    def descriptions(self: Self) -> list[IdentifierDescription]:
        return self.__descriptions

    @property
    def languages(self: Self) -> LanguageDict:
        return self.__languages

    # TODO
    def __str__(self: Self) -> str:
        return f"<Summary languages: {self.__languages}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class Stats:
    checksum: Optional[str]
    mtime: float

    @staticmethod
    def hash_file(file_path: Path, manager: Optional[Manager] = None) -> str:
        if file_path.is_dir():
            raise RuntimeError("Can't take checksum of directory")
        size: float = float(file_path.stat().st_size)
        bar: Optional[Any] = None
        if manager is not None:
            bar = manager.counter(
                total=size,
                desc="sha256 checksum",
                unit="B",
                leave=False,
                bar_format=CHECKSUM_BAR_FORMAT,
                color="red",
            )
            bar.update(0, force=True)
        sha256_hash = sha256()
        with open(str(file_path.absolute()), "rb") as file:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: file.read(4096), b""):
                sha256_hash.update(byte_block)
                if bar is not None:
                    bar.update(float(len(byte_block)))

            if bar is not None:
                bar.close(clear=True)

            return sha256_hash.hexdigest()

    @staticmethod
    def from_file(
        file_path: Path,
        file_type: ScannedFileType,
        *,
        generate_checksum: bool = True,
        manager: Optional[Manager] = None,
    ) -> "Stats":
        mtime: float = file_path.stat().st_mtime

        checksum: Optional[str] = (
            (
                None
                if file_type == ScannedFileType.folder
                else Stats.hash_file(file_path, manager=manager)
            )
            if generate_checksum
            else None
        )

        return Stats(checksum=checksum, mtime=mtime)

    def is_outdated(
        self: Self,
        path: Path,
        _type: ScannedFileType,
        manager: Optional[Manager] = None,
    ) -> bool:
        if _type == ScannedFileType.file:
            new_stats = Stats.from_file(
                path,
                _type,
                generate_checksum=False,
                manager=manager,
            )
            if new_stats.mtime <= self.mtime:
                return False

            # update the new mtime, since if we aren't outdated (per checksum), the parent caller wan't do it, if we are outdated, he will update it anyway
            self.mtime = new_stats.mtime

            with_checksum: Stats = Stats.from_file(path, _type, generate_checksum=True)
            if with_checksum.checksum == self.checksum:
                return False

            return True

        raise RuntimeError(
            "Outdated state fpr directories is not correctly reported by mtime or similar stats, so it isn't possible",
        )

    def as_dict(self: Self) -> dict[str, Any]:
        as_dict: dict[str, Any] = {
            "checksum": self.checksum,
            "mtime": self.mtime,
        }
        return as_dict

    @staticmethod
    def from_dict(dct: StatsDict) -> "Stats":
        return Stats(checksum=dct.get("checksum"), mtime=dct["mtime"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


@dataclass
class ScannedFile:
    path: Path
    parents: list[str]
    type: ScannedFileType  # noqa: A003
    stats: Stats

    @staticmethod
    def from_scan(
        file_path: Path,
        file_type: ScannedFileType,
        parent_folders: list[str],
    ) -> "ScannedFile":
        if len(parent_folders) > 3:
            raise RuntimeError(
                "No more than 3 parent folders are allowed: [collection] -> series -> season",
            )

        stats: Stats = Stats.from_file(file_path, file_type, generate_checksum=False)

        return ScannedFile(
            path=file_path,
            parents=parent_folders,
            type=file_type,
            stats=stats,
        )

    def generate_checksum(self: Self, manager: Optional[Manager] = None) -> None:
        self.stats = Stats.from_file(
            self.path,
            self.type,
            generate_checksum=True,
            manager=manager,
        )

    def is_outdated(self: Self, manager: Optional[Manager] = None) -> bool:
        return self.stats.is_outdated(self.path, self.type, manager=manager)

    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = {
            "path": encode(self.path),
            "parents": self.parents,
            "type": encode(self.type),
            "stats": encode(self.stats),
        }
        return as_dict

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


class NameParser:
    __language: Language

    def __init__(
        self: Self,
        language: Language = Language.unknown(),  # noqa: B008
    ) -> None:
        self.__language = language

    @property
    def language(self: Self) -> Language:
        return self.__language

    def parse_episode_name(self: Self, _name: str) -> Optional[tuple[str, int, int]]:
        raise MissingOverrideError

    def parse_season_name(self: Self, _name: str) -> Optional[tuple[int]]:
        raise MissingOverrideError

    def parse_series_name(self: Self, _name: str) -> Optional[tuple[str, int]]:
        raise MissingOverrideError


C = TypeVar("C")
CT = TypeVar("CT")
RT = TypeVar("RT")


class Callback(Generic[C, CT, RT]):
    def __init__(self: Self) -> None:
        pass

    def process(
        self: Self,
        file_path: Path,  # noqa: ARG002
        file_type: ScannedFileType,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        name_parser: NameParser,  # noqa: ARG002
        *,
        rescan: Optional[C] = None,  # noqa: ARG002
    ) -> Optional[C]:
        return None

    def ignore(
        self: Self,
        _file_path: Path,
        _file_type: ScannedFileType,
        _parent_folders: list[str],
    ) -> bool:
        return False

    def start(
        self: Self,
        amount: tuple[int, int, int],  # noqa: ARG002
        name: str,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        characteristic: CT,  # noqa: ARG002
    ) -> None:
        return None

    def progress(
        self: Self,
        name: str,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        characteristic: CT,  # noqa: ARG002
        *,
        amount: int = 1,  # noqa: ARG002
    ) -> None:
        return None

    def finish(
        self: Self,
        name: str,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        deleted: int,  # noqa: ARG002
        characteristic: CT,  # noqa: ARG002
    ) -> None:
        return None

    def get_saved(self: Self) -> RT:
        raise MissingOverrideError


SF = TypeVar("SF")


def safe_index(ls: list[SF], item: SF) -> Optional[int]:
    try:
        return ls.index(item)
    except ValueError:
        return None
