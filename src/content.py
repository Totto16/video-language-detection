#!/usr/bin/env python3

import dataclasses as dc
import sys
from dataclasses import dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
from json import JSONDecoder, JSONEncoder
from os import listdir
from pathlib import Path
from typing import (
    Any,
    Generic,
    Optional,
    Self,
    TypedDict,
    TypeVar,
    cast,
    get_type_hints,
)

from classifier import Classifier, FileMetadataError, Language, WAVFile
from enlighten import Manager
from typing_extensions import override


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
                self.__descriptions.append((description, desc[0]))

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
                self.__descriptions.append((description, desc[0], desc[1]))

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
                self.__descriptions.append((description, desc[0], desc[1], desc[2]))

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
        self: Self, json_encoder: Optional[JSONEncoder] = None,
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
        self: Self, language: Language = Language.unknown(),
    ) -> None:
        self.__language = language

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
        characteristic: CT,  # noqa: ARG002
    ) -> None:
        return None

    def get_saved(self: Self) -> RT:
        raise MissingOverrideError


ContentCharacteristic = tuple[Optional[ContentType], ScannedFileType]


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile


class Content:
    __type: ContentType
    __scanned_file: ScannedFile

    def __init__(self: Self, _type: ContentType, scanned_file: ScannedFile) -> None:
        self.__type = _type
        self.__scanned_file = scanned_file

    def summary(self: Self, _detailed: bool = False) -> Summary:
        raise MissingOverrideError

    @property
    def type(self: Self) -> ContentType:  # noqa: A003
        return self.__type

    @property
    def description(self: Self) -> Any:
        raise MissingOverrideError

    @property
    def scanned_file(self: Self) -> ScannedFile:
        return self.__scanned_file

    def generate_checksum(self: Self, manager: Manager) -> None:
        self.__scanned_file.generate_checksum(manager)

    @staticmethod
    def from_scan(
        file_path: Path,
        file_type: ScannedFileType,
        parent_folders: list[str],
        name_parser: NameParser,
    ) -> Optional["Content"]:
        scanned_file: ScannedFile = ScannedFile.from_scan(
            file_path,
            file_type,
            parent_folders,
        )

        name = file_path.name

        # [collection] -> series -> season -> file_name
        parents: list[str] = [*parent_folders, name]

        top_is_collection: bool = (
            (len(parents) == 1 and file_path.is_dir()) or len(parents) != 1
        ) and not SeriesContent.is_valid_name(parents[0], name_parser)

        try:
            if len(parents) == 4:
                if file_type == ScannedFileType.folder:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 4 - {file_path}!",
                    )

                return EpisodeContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 3 and not top_is_collection:
                if file_type == ScannedFileType.folder:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 3 - {file_path}!",
                    )

                return EpisodeContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 3 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 3 - {file_path}!",
                    )

                return SeasonContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 2 and not top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!",
                    )

                return SeasonContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 2 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!",
                    )

                return SeriesContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 1 and not top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!",
                    )

                return SeriesContent.from_path(file_path, scanned_file, name_parser)
            if len(parents) == 1 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!",
                    )
                return CollectionContent.from_path(file_path, scanned_file)

            raise RuntimeError("UNREACHABLE")

        except Exception as e:  # noqa: BLE001
            print(e, file=sys.stderr)
            return None

    def as_dict(
        self: Self, json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = {
            "scanned_file": encode(self.__scanned_file),
            "type": encode(self.__type),
        }
        return as_dict

    def scan(
        self: Self,
        callback: Callback["Content", ContentCharacteristic, Manager],  # noqa: ARG002
        name_parser: NameParser,  # noqa: ARG002
        *,
        parent_folders: list[str],  # noqa: ARG002
        classifier: Classifier,  # noqa: ARG002
        rescan: bool = False,  # noqa: ARG002
    ) -> None:
        raise MissingOverrideError

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


def process_folder(
    directory: Path,
    callback: Callback[Content, ContentCharacteristic, Manager],
    name_parser: NameParser,
    *,
    parent_folders: list[str],
    parent_type: Optional[ContentType] = None,
    rescan: Optional[list[Content]] = None,
) -> list[Content]:
    temp: list[tuple[Path, ScannedFileType, list[str]]] = []
    ignored: int = 0
    for file in listdir(directory):
        file_path: Path = Path(directory) / file

        file_type: ScannedFileType = ScannedFileType.from_path(file_path)
        should_ignore: bool = callback.ignore(file_path, file_type, parent_folders)
        if should_ignore:
            ignored += 1
            continue

        temp.append((file_path, file_type, parent_folders))

    value: ContentCharacteristic = (parent_type, ScannedFileType.folder)

    #  total, processing, ignored
    amount: tuple[int, int, int] = (len(temp) + ignored, len(temp), ignored)

    callback.start(amount, directory.name, parent_folders, value)

    if rescan is None:
        results: list[Content] = []
        for file_path, file_type, parent_folders in temp:
            result: Optional[Content] = callback.process(
                file_path,
                file_type,
                parent_folders,
                name_parser=name_parser,
            )
            value = (
                result.type if result is not None else None,
                ScannedFileType.from_path(file_path),
            )
            callback.progress(directory.name, parent_folders, value)
            if result is not None:
                results.append(result)

        value = (parent_type, ScannedFileType.folder)
        callback.finish(directory.name, parent_folders, value)

        return results

    already_scanned_file_paths: list[Path] = [
        content.scanned_file.path for content in rescan
    ]

    for file_path, file_type, parent_folders in temp:
        is_rescan = (
            None
            if file_path not in already_scanned_file_paths
            else (rescan[already_scanned_file_paths.index(file_path)])
        )

        result = callback.process(
            file_path,
            file_type,
            parent_folders,
            name_parser=name_parser,
            rescan=is_rescan,
        )

        value = (
            result.type if result is not None else None,
            ScannedFileType.from_path(file_path),
        )

        callback.progress(directory.name, parent_folders, value)
        if result is not None and is_rescan is None:
            rescan.append(result)

    value = (parent_type, ScannedFileType.folder)
    callback.finish(directory.name, parent_folders, value)

    return rescan


class EpisodeContentDict(ContentDict):
    description: EpisodeDescription
    language: Language


GLOBAL_ITER_MAX: int = 200
SKIP_ITR: int = 330
itr: int = 0


def itr_print_percent() -> None:
    global itr
    if itr < SKIP_ITR:
        return

    if itr >= GLOBAL_ITER_MAX + SKIP_ITR:
        return

    percent: float = (itr - SKIP_ITR) / GLOBAL_ITER_MAX * 100.0

    print(f"{percent:.02f} %")


class EpisodeContent(Content):
    __description: EpisodeDescription
    __language: Language

    @staticmethod
    def from_path(
        path: Path,
        scanned_file: ScannedFile,
        name_parser: NameParser,
    ) -> "EpisodeContent":
        description: Optional[EpisodeDescription] = EpisodeContent.parse_description(
            path.name,
            name_parser,
        )
        if description is None:
            raise NameError(f"Couldn't get EpisodeDescription from '{path}'")

        return EpisodeContent(scanned_file, description)

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: EpisodeDescription,
        language: Language = Language.unknown(),  # noqa: B008
    ) -> None:
        super().__init__(ContentType.episode, scanned_file)

        self.__description = description
        self.__language = language

    @property
    def description(self: Self) -> EpisodeDescription:
        return self.__description

    def __get_language(
        self: Self,
        classifier: Classifier,
        manager: Optional[Manager] = None,
    ) -> Language:
        try:
            wav_file = WAVFile(self.scanned_file.path)

            best, scanned_percent = classifier.predict(
                wav_file,
                self.scanned_file.path,
                manager,
            )
            return best.language
        except FileMetadataError:
            return Language.unknown()

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
    ) -> Optional[EpisodeDescription]:
        result = name_parser.parse_episode_name(name)
        if result is None:
            return None

        name, season, episode = result

        return EpisodeDescription(name, season, episode)

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        return Summary.from_single(self.__language, self.__description, detailed)

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
        manager: Manager = callback.get_saved()

        characteristic: ContentCharacteristic = (self.type, self.scanned_file.type)

        if rescan:
            is_outdated: bool = self.scanned_file.is_outdated(manager)

            if not is_outdated:
                if self.__language == Language.unknown():
                    callback.start(
                        (1, 1, 0),
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                    # TODO: re-enable
                    global itr
                    if itr < GLOBAL_ITER_MAX + SKIP_ITR:
                        itr_print_percent()
                        itr = itr + 1
                        if itr >= SKIP_ITR:
                            self.__language = self.__get_language(classifier, manager)

                    callback.progress(
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )
                    callback.finish(
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                return

        callback.start(
            (2, 2, 0),
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        self.generate_checksum(manager)
        callback.progress(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        # TODO: re-enable
        if itr < GLOBAL_ITER_MAX + SKIP_ITR:
            itr_print_percent()
            itr = itr + 1
            if itr >= SKIP_ITR:
                self.__language = self.__get_language(classifier, manager)

        callback.progress(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )
        callback.finish(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

    @override
    def as_dict(
        self: Self, json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["language"] = encode(self.__language)
        return as_dict

    @staticmethod
    def from_dict(dct: EpisodeContentDict) -> "EpisodeContent":
        return EpisodeContent(dct["scanned_file"], dct["description"], dct["language"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


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
        self: Self, json_encoder: Optional[JSONEncoder] = None,
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


class SeriesContentDict(ContentDict):
    description: SeriesDescription
    seasons: list[SeasonContent]


class SeriesContent(Content):
    __description: SeriesDescription
    __seasons: list[SeasonContent]

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
            raise NameError(f"Couldn't get SeriesDescription from '{path}'")

        return SeriesContent(scanned_file, description, [])

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: SeriesDescription,
        seasons: list[SeasonContent],
    ) -> None:
        super().__init__(ContentType.series, scanned_file)

        self.__description = description
        self.__seasons = seasons

    @property
    def description(self: Self) -> SeriesDescription:
        return self.__description

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
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for season in self.__seasons:
            summary.combine_seasons(self.description, season.summary(detailed))

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
                if isinstance(content, SeasonContent):
                    self.__seasons.append(content)
                else:
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeriesContent",
                    )

        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
                rescan=cast(list[Content], self.__seasons),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__seasons):
                if not isinstance(content, SeasonContent):
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in SeriesContent",
                    )

    @override
    def as_dict(
        self: Self, json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["seasons"] = self.__seasons
        return as_dict

    @staticmethod
    def from_dict(dct: SeriesContentDict) -> "SeriesContent":
        return SeriesContent(dct["scanned_file"], dct["description"], dct["seasons"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


class CollectionContentDict(ContentDict):
    description: CollectionDescription
    series: list[SeriesContent]


class CollectionContent(Content):
    __description: CollectionDescription
    __series: list[SeriesContent]

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "CollectionContent":
        return CollectionContent(scanned_file, path.name, [])

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: CollectionDescription,
        series: list[SeriesContent],
    ) -> None:
        super().__init__(ContentType.collection, scanned_file)

        self.__description = description
        self.__series = series

    @property
    def description(self: Self) -> str:
        return self.__description

    @override
    def summary(self: Self, detailed: bool = False) -> Summary:
        summary: Summary = Summary.empty(detailed)
        for serie in self.__series:
            summary.combine_series(self.description, serie.summary(detailed))

        return summary

    @override
    def as_dict(
        self: Self, json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = self.__description
        as_dict["series"] = self.__series
        return as_dict

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
                if isinstance(content, SeriesContent):
                    self.__series.append(content)
                else:
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in CollectionContent",
                    )

        else:
            ## no assignment of the return value is needed, it get's added implicitly per appending to the local reference of self
            process_folder(
                self.scanned_file.path,
                callback=callback,
                parent_folders=[*parent_folders, self.scanned_file.path.name],
                parent_type=self.type,
                name_parser=name_parser,
                rescan=cast(list[Content], self.__series),
            )

            # since some are added unchecked, check again now!
            for content in cast(list[Content], self.__series):
                if not isinstance(content, SeriesContent):
                    raise RuntimeError(
                        f"No child with class '{content.__class__.__name__}' is possible in CollectionContent",
                    )

    @staticmethod
    def from_dict(dct: CollectionContentDict) -> "CollectionContent":
        return CollectionContent(dct["scanned_file"], dct["description"], dct["series"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()


VALUE_KEY: str = "__value__"
TYPE_KEY: str = "__type__"


class Encoder(JSONEncoder):
    def default(self: Self, o: Any) -> Any:
        if isinstance(o, Content):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
        if isinstance(o, ScannedFile):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
        if isinstance(o, Path):
            return {TYPE_KEY: "Path", VALUE_KEY: str(o.absolute())}
        if isinstance(o, Enum):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.name}
        if isinstance(o, Stats):
            return {TYPE_KEY: "Stats", VALUE_KEY: o.as_dict()}
        if is_dataclass(o):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: dc.asdict(o)}

        return super().default(o)


class Decoder(JSONDecoder):
    def __init__(self: Self) -> None:
        def object_hook(dct: dict[str, Any] | Any) -> Any:
            if not isinstance(dct, dict):
                return dct

            _type: Optional[str] = cast(Optional[str], dct.get(TYPE_KEY))
            if _type is None:
                return dct

            def pipe(dct: Any, keys: list[str]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key in keys:
                    result[key] = object_hook(dct[key])

                return result

            value: Any = dct[VALUE_KEY]
            match _type:
                # Enums
                case "ScannedFileType":
                    return ScannedFileType(value)
                case "ContentType":
                    return ContentType(value)

                # dataclass
                case "ScannedFile":
                    return ScannedFile(**value)
                case "SeriesDescription":
                    return SeriesDescription(**value)
                case "SeasonDescription":
                    return SeasonDescription(**value)
                case "EpisodeDescription":
                    return EpisodeDescription(**value)
                case "Language":
                    return Language(**value)

                # other classes
                case "Stats":
                    stats_dict: StatsDict = cast(StatsDict, value)
                    return Stats.from_dict(stats_dict)
                case "Path":
                    path_value: str = cast(str, value)
                    return Path(path_value)

                # Content classes
                case "SeriesContent":
                    series_content_dict: SeriesContentDict = cast(
                        SeriesContentDict,
                        pipe(value, list(get_type_hints(SeriesContentDict).keys())),
                    )
                    return SeriesContent.from_dict(series_content_dict)
                case "CollectionContent":
                    collection_content_dict: CollectionContentDict = cast(
                        CollectionContentDict,
                        pipe(value, list(get_type_hints(CollectionContentDict).keys())),
                    )
                    return CollectionContent.from_dict(collection_content_dict)
                case "EpisodeContent":
                    episode_content_dict: EpisodeContentDict = cast(
                        EpisodeContentDict,
                        pipe(value, list(get_type_hints(EpisodeContentDict).keys())),
                    )
                    return EpisodeContent.from_dict(episode_content_dict)
                case "SeasonContent":
                    season_content_dict: SeasonContentDict = cast(
                        SeasonContentDict,
                        pipe(value, list(get_type_hints(SeasonContentDict).keys())),
                    )
                    return SeasonContent.from_dict(season_content_dict)

                # error
                case _:
                    raise TypeError(
                        f"Object of type {_type} is not JSON de-serializable",
                    )

        super().__init__(object_hook=object_hook)
