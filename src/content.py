#!/usr/bin/env python3

from dataclasses import dataclass
import dataclasses
from enum import Enum
import hashlib
from pathlib import Path
from typing import (
    Any,
    Generic,
    Optional,
    Callable,
    TypeVar,
    TypedDict,
    cast,
    get_type_hints,
)
from typing_extensions import override
import re
from classifier import Classifier, Language, WAVFile
from os import listdir
from json import JSONDecoder, JSONEncoder


class ScannedFileType(Enum):
    file = "file"
    folder = "folder"

    def __str__(self) -> str:
        return f"<ScannedFileType: {self.name}>"

    def __repr__(self) -> str:
        return str(self)


class ContentType(Enum):
    series = "series"
    season = "season"
    episode = "episode"
    collection = "collection"

    def __str__(self) -> str:
        return f"<ContentType: {self.name}>"

    def __repr__(self) -> str:
        return str(self)


class MissingOverrideError(RuntimeError):
    pass


def parse_int_safely(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None


class StatsDict(TypedDict):
    checksum: Optional[str]
    mtime: float


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
        checksum: Optional[str] = None
        # TODO: re-enable this
        # (None if file_type == ScannedFileType.folder else Stats.hash_file(file_path))

        mtime: float = Path(file_path).stat().st_mtime

        return Stats(checksum=checksum, mtime=mtime)

    def as_dict(self) -> dict[str, Any]:
        as_dict: dict[str, Any] = {
            "checksum": self.checksum,
            "mtime": self.mtime,
        }
        return as_dict

    @staticmethod
    def from_dict(dct: StatsDict) -> "Stats":
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

    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        encode: Callable[[Any], Any] = lambda x: (
            x if json_encoder is None else json_encoder.default(x)
        )
        as_dict: dict[str, Any] = {
            "path": encode(self.path),
            "parents": self.parents,
            "type": encode(self.type),
            "stats": encode(self.stats),
        }
        return as_dict

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


C = TypeVar("C")


class Callback(Generic[C]):
    def __init__(self) -> None:
        pass

    def process(
        self, file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> Optional[C]:
        return None

    def ignore(
        self, file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> bool:
        return False

    def amount(self, amount: int, name: str, parent_folders: list[str]) -> None:
        return None

    def progress(self, name: str, parent_folders: list[str]) -> None:
        return None

    def finish(self, name: str, parent_folders: list[str]) -> None:
        return None


T = TypeVar("T")


def process_folder(
    directory: Path,
    callback: Callback[T],
    *,
    parent_folders: list[str] = [],
) -> list[T]:
    temp: list[tuple[Path, ScannedFileType, list[str]]] = []
    for file in listdir(directory):
        file_path: Path = Path(directory) / file

        file_type: ScannedFileType = (
            ScannedFileType.folder if file_path.is_dir() else ScannedFileType.file
        )
        should_ignore: bool = callback.ignore(file_path, file_type, parent_folders)
        if should_ignore:
            continue

        temp.append((file_path, file_type, parent_folders))

    callback.amount(len(temp), directory.name, parent_folders)

    results: list[T] = []
    for file_path, file_type, parent_folders in temp:
        result: Optional[T] = callback.process(file_path, file_type, parent_folders)
        callback.progress(directory.name, parent_folders)
        if result is not None:
            results.append(result)

    callback.finish(directory.name, parent_folders)

    return results


class ContentDict(TypedDict):
    type: ContentType
    scanned_file: ScannedFile


class Content:
    __type: ContentType
    __scanned_file: ScannedFile

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

        top_is_collection: bool = (
            (len(parents) == 1 and file_path.is_dir()) or len(parents) != 1
        ) and not SeriesContent.is_valid_name(parents[0])

        try:
            if len(parents) == 4:
                if file_type == ScannedFileType.folder:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 4 - {file_path}!"
                    )

                return EpisodeContent.from_path(file_path, scanned_file)
            elif len(parents) == 3 and not top_is_collection:
                if file_type == ScannedFileType.folder:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 3 - {file_path}!"
                    )

                return EpisodeContent.from_path(file_path, scanned_file)
            elif len(parents) == 3 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting 3 - {file_path}!"
                    )

                return SeasonContent.from_path(file_path, scanned_file)
            elif len(parents) == 2 and not top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!"
                    )

                return SeasonContent.from_path(file_path, scanned_file)
            elif len(parents) == 2 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!"
                    )

                return SeriesContent.from_path(file_path, scanned_file)
            elif len(parents) == 1 and not top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!"
                    )

                return SeriesContent.from_path(file_path, scanned_file)
            elif len(parents) == 1 and top_is_collection:
                if file_type == ScannedFileType.file:
                    raise RuntimeError(
                        f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!"
                    )
                return CollectionContent.from_path(file_path, scanned_file)
            else:
                raise RuntimeError("UNREACHABLE")

        except Exception as e:
            print(e)
            return None

    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        encode: Callable[[Any], Any] = lambda x: (
            x if json_encoder is None else json_encoder.default(x)
        )

        as_dict: dict[str, Any] = {
            "scanned_file": encode(self.__scanned_file),
            "type": encode(self.__type),
        }
        return as_dict

    def scan(
        self,
        callback: Callback["Content"],
        *,
        parent_folders: list[str] = [],
        classifier: Classifier,
    ) -> None:
        raise MissingOverrideError

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class EpisodeDescription:
    name: str
    season: int
    episode: int

    def __str__(self) -> str:
        return (
            f"<Episode season: {self.season} episode: {self.episode} name: {self.name}>"
        )

    def __repr__(self) -> str:
        return str(self)


class EpisodeContentDict(ContentDict):
    description: EpisodeDescription
    language: Language


class EpisodeContent(Content):
    __description: EpisodeDescription
    __language: Language

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "EpisodeContent":
        description: Optional[EpisodeDescription] = EpisodeContent.parse_description(
            path.name
        )
        if description is None:
            raise NameError(f"Couldn't get EpisodeDescription from '{path}'")

        return EpisodeContent(scanned_file, description)

    def __init__(
        self,
        scanned_file: ScannedFile,
        description: EpisodeDescription,
    ) -> None:
        super().__init__(ContentType.episode, scanned_file)

        self.__description = description
        self.__language = Language.Unknown()

    @property
    def description(self) -> EpisodeDescription:
        return self.__description

    def __get_language(self, classifier: Classifier) -> Language:
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

    @override
    def scan(
        self,
        callback: Callback[Content],
        *,
        parent_folders: list[str] = [],
        classifier: Classifier,
    ) -> None:
        pass
        # print("scan episode")

        # TODO: uncomment this
        # self.__language = self.__get_language(classifier)

    @override
    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        encode: Callable[[Any], Any] = lambda x: (
            x if json_encoder is None else json_encoder.default(x)
        )
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["language"] = encode(self.__language)
        return as_dict

    @staticmethod
    def from_dict(dct: EpisodeContentDict) -> "EpisodeContent":
        return EpisodeContent(dct["scanned_file"], dct["description"])

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SeasonDescription:
    season: int

    def __str__(self) -> str:
        return f"<Season season: {self.season}>y"

    def __repr__(self) -> str:
        return str(self)


SPECIAL_NAMES: list[str] = ["Extras", "Specials", "Special"]


class SeasonContentDict(ContentDict):
    description: SeasonDescription
    episodes: list[EpisodeContent]


class SeasonContent(Content):
    __description: SeasonDescription
    __episodes: list[EpisodeContent]

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "SeasonContent":
        description: Optional[SeasonDescription] = SeasonContent.parse_description(
            path.name
        )
        if description is None:
            raise NameError(f"Couldn't get SeasonDescription from '{path}'")

        return SeasonContent(scanned_file, description, [])

    def __init__(
        self,
        scanned_file: ScannedFile,
        description: SeasonDescription,
        episodes: list[EpisodeContent],
    ) -> None:
        super().__init__(ContentType.season, scanned_file)

        self.__description = description
        self.__episodes = episodes

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return SeasonContent.parse_description(name) is not None

    @staticmethod
    def parse_description(name: str) -> Optional[SeasonDescription]:
        match = re.search(r"Staffel (\d{2})", name)
        if match is None:
            if name in SPECIAL_NAMES:
                return SeasonDescription(0)
            else:
                return None

        groups = match.groups()
        if len(groups) != 1:
            return None

        (_season,) = groups
        season = parse_int_safely(_season)
        if season is None:
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

    @override
    def scan(
        self,
        callback: Callback[Content],
        *,
        parent_folders: list[str] = [],
        classifier: Classifier,
    ) -> None:
        contents: list[Content] = process_folder(
            self.scanned_file.path,
            callback=callback,
            parent_folders=[*parent_folders, self.scanned_file.path.name],
        )
        for content in contents:
            if isinstance(content, EpisodeContent):
                self.__episodes.append(content)
            else:
                raise RuntimeError(
                    f"No child with class {type(content)} is possible in SeasonContent"
                )

    @override
    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        encode: Callable[[Any], Any] = lambda x: (
            x if json_encoder is None else json_encoder.default(x)
        )
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["episodes"] = self.__episodes
        return as_dict

    @staticmethod
    def from_dict(dct: SeasonContentDict) -> "SeasonContent":
        return SeasonContent(dct["scanned_file"], dct["description"], dct["episodes"])

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SeriesDescription:
    name: str
    year: int

    def __str__(self) -> str:
        return f"<Series name: {self.name} year: {self.year}>"

    def __repr__(self) -> str:
        return str(self)


class SeriesContentDict(ContentDict):
    description: SeriesDescription
    seasons: list[SeasonContent]


class SeriesContent(Content):
    __description: SeriesDescription
    __seasons: list[SeasonContent]

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "SeriesContent":
        description: Optional[SeriesDescription] = SeriesContent.parse_description(
            path.name
        )
        if description is None:
            raise NameError(f"Couldn't get SeriesDescription from '{path}'")

        return SeriesContent(scanned_file, description, [])

    def __init__(
        self,
        scanned_file: ScannedFile,
        description: SeriesDescription,
        seasons: list[SeasonContent],
    ) -> None:
        super().__init__(ContentType.series, scanned_file)

        self.__description = description
        self.__seasons = seasons

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

    @override
    def scan(
        self,
        callback: Callback[Content],
        *,
        parent_folders: list[str] = [],
        classifier: Classifier,
    ) -> None:
        contents: list[Content] = process_folder(
            self.scanned_file.path,
            callback=callback,
            parent_folders=[*parent_folders, self.scanned_file.path.name],
        )
        for content in contents:
            if isinstance(content, SeasonContent):
                self.__seasons.append(content)
            else:
                raise RuntimeError(
                    f"No child with class {type(content)} is possible in SeriesContent"
                )

    @override
    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        encode: Callable[[Any], Any] = lambda x: (
            x if json_encoder is None else json_encoder.default(x)
        )
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["seasons"] = self.__seasons
        return as_dict

    @staticmethod
    def from_dict(dct: SeriesContentDict) -> "SeriesContent":
        return SeriesContent(dct["scanned_file"], dct["description"], dct["seasons"])

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


class CollectionContentDict(ContentDict):
    description: str
    series: list[SeriesContent]


class CollectionContent(Content):
    __description: str
    __series: list[SeriesContent]

    @staticmethod
    def from_path(path: Path, scanned_file: ScannedFile) -> "CollectionContent":
        return CollectionContent(scanned_file, path.name, [])

    def __init__(
        self,
        scanned_file: ScannedFile,
        description: str,
        series: list[SeriesContent],
    ) -> None:
        super().__init__(ContentType.collection, scanned_file)

        self.__description = description
        self.__series = series

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

    @override
    def as_dict(self, json_encoder: Optional[JSONEncoder] = None) -> dict[str, Any]:
        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = self.__description
        as_dict["series"] = self.__series
        return as_dict

    @override
    def scan(
        self,
        callback: Callback[Content],
        *,
        parent_folders: list[str] = [],
        classifier: Classifier,
    ) -> None:
        contents: list[Content] = process_folder(
            self.scanned_file.path,
            callback=callback,
            parent_folders=[*parent_folders, self.scanned_file.path.name],
        )
        for content in contents:
            if isinstance(content, SeriesContent):
                self.__series.append(content)
            else:
                raise RuntimeError(
                    f"No child with class {type(content)} is possible in CollectionContent"
                )

    @staticmethod
    def from_dict(dct: CollectionContentDict) -> "CollectionContent":
        return CollectionContent(dct["scanned_file"], dct["description"], dct["series"])

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return self.__str__()


VALUE_KEY: str = "__value__"
TYPE_KEY: str = "__type__"


class Encoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Content):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
        elif isinstance(o, ScannedFile):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
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
        def object_hook(dct: dict[str, Any] | Any) -> Any:
            if not isinstance(dct, dict):
                return dct

            type: Optional[str] = cast(Optional[str], dct.get(TYPE_KEY))
            if type is None:
                return dct

            def pipe(dct: Any, keys: list[str]) -> dict[str, Any]:
                result: dict[str, Any] = dict()
                for key in keys:
                    result[key] = object_hook(dct[key])

                return result

            value: Any = dct[VALUE_KEY]
            match type:
                # Enums
                case "ScannedFileType":
                    return ScannedFileType(value)
                case "ContentType":
                    return ContentType(value)

                # dataclass
                case "SeriesDescription":
                    return SeriesDescription(**value)
                case "ScannedFile":
                    return ScannedFile(**value)
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
                        f"Object of type {type} is not JSON de-serializable"
                    )

        super().__init__(object_hook=object_hook)
