from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Optional,
    Self,
    TypedDict,
)

from apischema import schema
from enlighten import Manager

from content.language import Language
from content.metadata.metadata import HandlesType


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


class ContentType(str, Enum):
    series = "series"
    season = "season"
    episode = "episode"
    collection = "collection"
    numerated = "numerated"

    def __str__(self: Self) -> str:
        return f"<ContentType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class MissingOverrideError(RuntimeError):
    pass


class StatsDict(TypedDict):
    checksum: Optional[str]
    mtime: float


CHECKSUM_BAR_FORMAT: str = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


@dataclass(slots=True, repr=True)
class EpisodeDescription:
    name: str
    season: int = field(metadata=schema(min=0))
    episode: int = field(metadata=schema(min=1))

    def __str__(self: Self) -> str:
        return (
            f"<Episode season: {self.season} episode: {self.episode} name: {self.name}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class NumeratedDescription:
    name: str
    season: int = field(metadata=schema(min=0))
    episode: int = field(metadata=schema(min=1))
    number: int = field(metadata=schema(min=1))

    def __str__(self: Self) -> str:
        return f"<Episode season: {self.season} episode: {self.episode} name: {self.name} number: {self.number}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class SeriesDescription:
    name: str
    year: int = field(metadata=schema(min=1900))

    def __str__(self: Self) -> str:
        return f"<Series name: {self.name} year: {self.year}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class SeasonDescription:
    season: int = field(metadata=schema(min=0))

    def __str__(self: Self) -> str:
        return f"<Season season: {self.season}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class Stats:
    checksum: Optional[str]
    mtime: float

    @staticmethod
    def hash_file(file_path: Path, manager: Optional[Manager] = None) -> str:
        if file_path.is_dir():
            msg = "Can't take checksum of directory"
            raise RuntimeError(msg)
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
        with file_path.open(mode="rb") as file:
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
            return with_checksum.checksum != self.checksum

        msg = "Outdated state for directories is not correctly reported by mtime or similar stats, so it isn't possible"
        raise RuntimeError(msg)


@dataclass(slots=True, repr=True)
class ScannedFile:
    path: Path = field(
        metadata=schema(
            title="file path",
            description="The file path of the scanned file / folder",
        ),
    )
    parents: list[str] = field(
        metadata=schema(
            title="parent folders",
            description="The parent folders of this scanned file / folder",
            min_items=0,
            max_items=3,
            unique=True,
        ),
    )
    type: ScannedFileType = field(
        metadata=schema(
            title="file type",
            description="The type of the file: folder or file",
        ),
    )
    stats: Stats = field(
        metadata=schema(
            title="file stats",
            description="The stats of this file",
        ),
    )

    @staticmethod
    def from_scan(
        file_path: Path,
        file_type: ScannedFileType,
        parent_folders: list[str],
    ) -> "ScannedFile":
        if len(parent_folders) > 3:
            msg = "No more than 3 parent folders are allowed: [collection] -> series -> season"
            raise RuntimeError(msg)

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


class NameParser:
    __language: Language

    def __init__(self: Self, language: Optional[Language]) -> None:
        self.__language = language if language is not None else Language.get_default()

    @property
    def language(self: Self) -> Language:
        return self.__language

    def parse_episode_name(self: Self, _name: str) -> Optional[tuple[str, int, int]]:
        raise MissingOverrideError

    def parse_season_name(self: Self, _name: str) -> Optional[tuple[int]]:
        raise MissingOverrideError

    def parse_series_name(self: Self, _name: str) -> Optional[tuple[str, int]]:
        raise MissingOverrideError


class Callback[C, CT, RT]:
    def __init__(self: Self) -> None:
        pass

    def process(
        self: Self,
        file_path: Path,  # noqa: ARG002
        file_type: ScannedFileType,  # noqa: ARG002
        handles: HandlesType,  # noqa: ARG002
        parent_folders: list[str],  # noqa: ARG002
        *,
        trailer_names: list[str],  # noqa: ARG002
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


def safe_index[SF](ls: list[SF], item: SF) -> Optional[int]:
    try:
        return ls.index(item)
    except ValueError:
        return None
