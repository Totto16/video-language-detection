#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from math import floor
from os import makedirs, path, remove
from typing import Any, Optional, TypedDict, cast
from pathlib import Path
from enlighten import Manager
from ffprobe import FFProbe
from ffmpeg import FFmpeg, Progress

from warnings import filterwarnings

filterwarnings("ignore")

from speechbrain.pretrained import EncoderClassifier
from torch import cuda
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from humanize import naturalsize
from shutil import rmtree


WAV_FILE_BAR_FMT = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}.1f}/{total:.1f} "
    + "[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
)


def timedelta_from_minutes(minutes: float) -> timedelta:
    m: int = floor(minutes)
    s: int = floor((minutes % 1) * 60)
    return timedelta(minutes=m, seconds=s)


class FileType(Enum):
    wav = "wav"
    video = "video"
    audio = "audio"

    def __str__(self) -> str:
        return f"<FileType: {self.name}>"

    def __repr__(self) -> str:
        return str(self)


class Status(Enum):
    ready = "ready"
    raw = "raw"

    def __str__(self) -> str:
        return f"<Status: {self.name}>"

    def __repr__(self) -> str:
        return str(self)


class WAVOptions(TypedDict):
    bitrate: int
    amount: Optional[timedelta]


class WAVFile:
    __tmp_file: Optional[Path]
    __file: Path
    __type: FileType
    __status: Status
    __runtime: Optional[float]

    def __init__(self, file: Path) -> None:
        self.__tmp_file = None
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        type, status, runtime = self.__get_info()
        self.__type = type
        self.__status = status
        self.__runtime = runtime

    def __get_info(self) -> tuple[FileType, Status, Optional[float]]:
        try:
            metadata = FFProbe(str(self.__file.absolute()))
            for stream in metadata.streams:
                if stream.is_video():
                    return (
                        FileType.video,
                        Status.raw,
                        stream.duration_seconds() / 60.0,
                    )

            for stream in metadata.streams:
                if stream.is_audio() and stream.codec() == "pcm_s16le":
                    return (
                        FileType.wav,
                        Status.ready,
                        stream.duration_seconds() / 60.0,
                    )

            for stream in metadata.streams:
                if stream.is_audio():
                    return (
                        FileType.audio,
                        Status.raw,
                        stream.duration_seconds() / 60.0,
                    )

            return (FileType.audio, Status.raw, None)
        except Exception:
            return (FileType.video, Status.raw, None)

    @property
    def runtime(self) -> Optional[float]:
        return self.__runtime

    def create_wav_file(
        self,
        options: WAVOptions = {"bitrate": 16000, "amount": timedelta(minutes=10)},
        *,
        force_recreation: bool = False,
        manager: Optional[Manager] = None,
    ) -> bool:
        match (self.__status, self.__type, force_recreation):
            case (Status.ready, _, False):
                return False
            case (_, FileType.wav, _):
                return False
            case (Status.raw, _, False):
                self.__convert_to_wav(options, manager)
                return True
            case (_, _, True):
                self.__convert_to_wav(options, manager)
                return True
            case _:  # stupid mypy
                raise RuntimeError("UNREACHABLE")

    def __convert_to_wav(
        self, options: WAVOptions, manager: Optional[Manager] = None
    ) -> None:
        bar: Optional[Any] = None
        elapsed_time: timedelta = timedelta(seconds=0)
        if manager is not None and self.runtime is not None:
            bar = manager.counter(
                total=timedelta_from_minutes(self.runtime),
                desc="generating wav",
                unit="minutes",
                leave=False,
                bar_format=WAV_FILE_BAR_FMT,
                color="red",
            )
            bar.update(0, force=True)

        temp_dir: Path = Path("/tmp") / "video_lang_detect"
        if not temp_dir.exists():
            makedirs(temp_dir)

        self.__tmp_file = temp_dir / (self.__file.stem + ".wav")

        if self.__tmp_file.exists():
            remove(str(self.__tmp_file.absolute()))

        ffmpeg_options: dict[str, Any] = {
            "acodec": "pcm_s16le",
            "ar": options["bitrate"],
            "ac": 1,
            "stats_period": 1,
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

        def progress_report(progress: Progress) -> None:
            if bar is not None:
                delta_time = progress.time - elapsed_time
                bar.update(delta_time)

            elapsed_time += progress.time

        ffmpeg.on("progress", progress_report)
        ffmpeg.execute()

        self.__status = Status.ready

        if bar is not None:
            bar.close(clear=True)

    def wav_path(self) -> Path:
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
            remove(str(self.__tmp_file.absolute()))


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
            raise RuntimeError(f"Couldn't get the Language from str '{input}'")

        return lan

    @staticmethod
    def Unknown() -> "Language":
        return Language("un", "Unknown")

    def __str__(self) -> str:
        return f"<Language short: {self.short} long: {self.long}>"

    def __repr__(self) -> str:
        return str(self)


class Classifier:
    __classifier: EncoderClassifier
    __save_dir: Path

    def __init__(self) -> None:
        self.__save_dir = Path(path.dirname(__file__)) / "tmp"

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
        if not cuda.is_available():
            return None

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print("GPU stats:")
        print(f"total    : {naturalsize(info.total, binary=True)}")
        print(f"free     : {naturalsize(info.free, binary=True)}")
        print(f"used     : {naturalsize(info.used, binary=True)}")

    def predict(
        self, wav_file: WAVFile, manager: Optional[Manager] = None
    ) -> tuple[Language, float]:
        def get_minutes(runtime: float) -> list[Optional[float]]:
            index: float = 1.0
            result: list[Optional[float]] = [index]
            steps: float = 1.0
            while True:
                index += steps
                if index > runtime:
                    result.append(None)
                    break

                result.append(index)
                steps += 1.0

            return result

        minutes: list[Optional[float]] = (
            [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, None]
            if wav_file.runtime is None
            else get_minutes(wav_file.runtime)
        )

        bar: Optional[Any] = None
        if manager is not None:
            bar = manager.counter(
                total=len(minutes),
                desc="detecting language",
                unit="attempts",
                leave=False,
                color="red",
            )
            bar.update(0, force=True)
        for minute in minutes:
            try:
                delta: Optional[timedelta] = None
                if minute is not None:
                    delta = timedelta_from_minutes(minute)
                wav_file.create_wav_file(
                    {"bitrate": 16000, "amount": delta},
                    force_recreation=True,
                    manager=manager,
                )

                # from: https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
                signal = self.__classifier.load_audio(
                    str(wav_file.wav_path().absolute()),
                    savedir=str(self.__save_dir.absolute()),
                )
                prediction = self.__classifier.classify_batch(signal)

                accuracy = cast(float, prediction[1].exp().item())
                # The identified language ISO code is given in prediction[3]
                language = Language.from_str_unsafe(cast(str, prediction[3][0]))

                if bar is not None:
                    bar.update()

                if accuracy < 0.95:
                    continue

                if bar is not None:
                    bar.close(clear=True)

                return (language, accuracy)
            except RuntimeError as exception:
                if isinstance(exception, cuda.OutOfMemoryError):
                    self.__init_classifier(True)
                else:
                    raise exception

        if bar is not None:
            bar.close(clear=True)

        return (Language.Unknown(), 0.0)

    def __get_run_opts(self) -> Optional[dict[str, Any]]:
        if not cuda.is_available():
            return None

        gc.collect()
        cuda.empty_cache()
        return {
            "device": "cuda",
            "data_parallel_count": -1,
            "data_parallel_backend": True,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }

    def __del__(self) -> None:
        if self.__save_dir.exists():
            if self.__save_dir.is_file():
                remove(str(self.__save_dir.absolute()))
            else:
                rmtree(str(self.__save_dir.absolute()), ignore_errors=True)
