#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
import hashlib
from os import listdir, makedirs, path, remove
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, cast
from typing_extensions import override
from ffprobe import FFProbe
from ffmpeg import FFmpeg, Progress
from speechbrain.pretrained import EncoderClassifier
import torch
import gc
import pynvml
import humanize
import shutil
import re


class FileType(Enum):
    wav = "wav"
    video = "video"
    audio = "audio"


class Status(Enum):
    ready = "ready"
    raw = "raw"


class WAVOptions(TypedDict):
    bitrate: int
    amount: Optional[timedelta]


class WAVFile:
    __tmp_file: Optional[str]
    __file: str
    __type: FileType
    __status: Status

    def __init__(self, file: str) -> None:
        self.__tmp_file = None
        if not path.exists(file):
            raise FileNotFoundError(file)
        self.__file = file
        type, status = self.__get_info()
        self.__type = type
        self.__status = status

    def __get_info(self) -> tuple[FileType, Status]:
        try:
            metadata = FFProbe(self.__file)
            for stream in metadata.streams:
                if stream.is_video():
                    return (FileType.video, Status.raw)

            for stream in metadata.streams:
                if stream.is_audio() and stream.codec() == "pcm_s16le":
                    return (FileType.wav, Status.ready)

            return (FileType.audio, Status.raw)
        except Exception:
            return (FileType.video, Status.raw)

    def create_wav_file(
        self,
        options: WAVOptions = {"bitrate": 16000, "amount": timedelta(minutes=10)},
        force_recreation: bool = False,
    ) -> bool:
        match (self.__status, self.__type, force_recreation):
            case (Status.ready, _, False):
                return False
            case (_, FileType.wav, _):
                return False
            case (Status.raw, _, False):
                self.__convert_to_wav(options)
                return True
            case (_, _, True):
                self.__convert_to_wav(options)
                return True
            case _:  # stupid mypy
                raise RuntimeError("UNREACHABLE")

    def __convert_to_wav(self, options: WAVOptions) -> None:
        temp_dir: str = path.join("/tmp", "video_lang_detect")
        if not path.exists(temp_dir):
            makedirs(temp_dir)
        self.__tmp_file = path.join(temp_dir, path.basename(self.__file) + ".wav")

        if path.exists(self.__tmp_file):
            remove(self.__tmp_file)

        ffmpeg_options: dict[str, Any] = {
            "acodec": "pcm_s16le",
            "ar": options["bitrate"],
            "ac": 1,
        }

        if options["amount"] is not None:
            ffmpeg_options["to"] = (str(options["amount"]),)

        # to use the same format as: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
        ffmpeg: FFmpeg = (
            FFmpeg()
            .option("y")
            .input(self.__file)
            .output(self.__tmp_file, **ffmpeg_options)
        )

        @ffmpeg.on("progress")  # type: ignore
        def progress_report(progress: Progress) -> None:
            # print(progress)
            pass

        ffmpeg.execute()

        self.__status = Status.ready

    def wav_path(self) -> str:
        match (self.__status, self.__type):
            case (_, FileType.wav):
                return self.__file
            case (Status.raw, _):
                raise RuntimeError("Not converted")
            case (Status.ready, _):
                if self.__tmp_file is None:
                    raise RuntimeError("Not converted correctly, temp file is missing")
                return self.__tmp_file
            case _:  # stupid mypy
                raise RuntimeError("UNREACHABLE")

    def __del__(self) -> None:
        if self.__tmp_file is not None:
            remove(self.__tmp_file)


class LanguageDict(TypedDict):
    language: str
    score: float


@dataclass
class Language:
    short: str
    long: str

    @staticmethod
    def from_str(input: str) -> Optional["Language"]:
        arr: list[str] = [a.strip() for a in input.split(":")]
        if len(arr) != 2:
            return None

        return Language(arr[0], arr[1])

    @staticmethod
    def from_str_unsafe(input: str) -> "Language":
        lan: Optional["Language"] = Language.from_str(input)
        if lan is None:
            raise RuntimeError(f"Couldn't get the Language from str {input}")

        return lan


class Classifier:
    __classifier: EncoderClassifier
    __save_dir: str

    def __init__(self) -> None:
        self.__save_dir = path.join(path.dirname(__file__), "tmp")

        self.__init_classifier()

    def __init_classifier(self, force_cpu: bool = False) -> None:
        run_opts: Optional[dict[str, Any]] = None
        if not force_cpu:
            run_opts = self.__get_run_opts()

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="model",
            run_opts=run_opts,
        )
        if classifier is None:
            raise RuntimeError("Couldn't initialize Classifier")

        self.__classifier = classifier

    @staticmethod
    def print_gpu_stat() -> None:
        if not torch.cuda.is_available():
            return None

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print("GPU stats:")
        print(f"total    : {humanize.naturalsize(info.total, binary=True)}")
        print(f"free     : {humanize.naturalsize(info.free, binary=True)}")
        print(f"used     : {humanize.naturalsize(info.used, binary=True)}")

    def predict(self, wav_file: WAVFile) -> tuple[Language, float]:
        minutes: list[Optional[int]] = [1, 2, 4, 5, 10, 20, None]

        for minute in minutes:
            try:
                delta = None if minute is None else timedelta(minutes=minute)
                wav_file.create_wav_file({"bitrate": 16000, "amount": delta}, True)

                # from: https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
                signal = self.__classifier.load_audio(
                    wav_file.wav_path(), savedir=self.__save_dir
                )
                prediction = self.__classifier.classify_batch(signal)

                accuracy = cast(float, prediction[1].exp().item())
                # The identified language ISO code is given in prediction[3]
                language = Language.from_str_unsafe(cast(str, prediction[3][0]))

                if accuracy < 0.95:
                    continue

                return (language, accuracy)
            except RuntimeError as exception:
                if isinstance(exception, torch.cuda.OutOfMemoryError):
                    self.__init_classifier(True)
                else:
                    raise exception

        raise RuntimeError("No language with enough accuracy could be found :(")

    def __get_run_opts(self) -> Optional[dict[str, Any]]:
        if not torch.cuda.is_available():
            return None

        gc.collect()
        torch.cuda.empty_cache()
        return {
            "device": "cuda",
            "data_parallel_count": -1,
            "data_parallel_backend": True,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }

    def __del__(self) -> None:
        if path.exists(self.__save_dir):
            if path.isfile(self.__save_dir):
                remove(self.__save_dir)
            else:
                shutil.rmtree(self.__save_dir, ignore_errors=True)


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


class Content:
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
    content: Optional[Content]

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

        content = Content.from_scan(file_path, file_type, parent_folders)

        return ScannedFile(
            parents=parent_folders,
            name=name,
            type=file_type,
            stats=stats,
            content=content,
        )


def process_folder_recursively(
    directory: Path,
    *,
    callback_function: Optional[
        Callable[[Path, ScannedFileType, list[str]], None]
    ] = None,
    ignore_folder_fn: Optional[Callable[[Path, str, list[str]], bool]] = None,
    parent_folders: list[str] = [],
) -> None:
    for file in listdir(directory):
        file_path: Path = Path(path.join(directory, file))
        if file_path.is_dir():
            if ignore_folder_fn is not None:
                should_ignore = ignore_folder_fn(file_path, file, parent_folders)
                if should_ignore:
                    continue

            if callback_function is not None:
                callback_function(file_path, ScannedFileType.folder, parent_folders)

            process_folder_recursively(
                file_path,
                callback_function=callback_function,
                parent_folders=[*parent_folders, file],
                ignore_folder_fn=ignore_folder_fn,
            )
        else:
            if callback_function is not None:
                callback_function(file_path, ScannedFileType.file, parent_folders)


def main() -> None:
    classifier = Classifier()

    ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")

    video_formats: list[str] = ["mp4", "mkv", "avi"]

    def process_file(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> None:
        needs_scan: bool = file_type == ScannedFileType.folder
        if file_type == ScannedFileType.file:
            extension: str = file_path.suffix[1:]
            if extension not in video_formats:
                needs_scan = False

        if not needs_scan:
            return

        file: ScannedFile = ScannedFile.from_scan(file_path, file_type, parent_folders)
        print(file)

        if file_type == ScannedFileType.folder:
            return

        return

        wav_file = WAVFile(file_path)

        language, accuracy = classifier.predict(wav_file)
        print(language, accuracy)

    ignore_files: list[str] = ["metadata"]

    def ignore_folders(file_path: Path, file: str, parent_folders: list[str]) -> bool:
        if file.startswith("."):
            return True

        if file in ignore_files:
            return True

        return False

    process_folder_recursively(
        ROOT_FOLDER, callback_function=process_file, ignore_folder_fn=ignore_folders
    )


if __name__ == "__main__":
    main()
