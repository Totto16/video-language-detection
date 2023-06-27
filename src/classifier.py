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
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from humanize import naturalsize
from shutil import rmtree


WAV_FILE_BAR_FMT = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:2n}/{total:2n} "
    + "[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
)


def parse_int_safely(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None


class Timestamp:
    __delta: timedelta

    def __init__(self, delta: timedelta) -> None:
        self.__delta = delta

    @property
    def delta(self) -> timedelta:
        return self.__delta

    @property
    def minutes(self) -> float:
        return self.__delta.total_seconds() / 60.0

    @staticmethod
    def zero() -> "Timestamp":
        return Timestamp(timedelta(seconds=0))

    @staticmethod
    def from_minutes(minutes: float) -> "Timestamp":
        m: int = floor(minutes)
        s: int = floor((minutes % 1) * 60)
        return Timestamp(timedelta(minutes=m, seconds=s))

    @staticmethod
    def from_seconds(seconds: float) -> "Timestamp":
        return Timestamp.from_minutes(seconds / 60)

    def __str__(self) -> str:
        return str(self.__delta)

    def __repr__(self) -> str:
        return str(self)

    def __format__(self, spec: str) -> str:
        """This is responsible for formatting the Timestamp

        Args:
            spec (str): a formst string, if empty or "d" no microseconds will be printed
                        otherwise you can provide an int from 1-5 inclusive, to determine how much you wan't to round
                        the microseconds, you can provide a "n" afterwards, to not have ".00". e.g , it ignores 0's with e.g. "2n"

        Examples:
            f"{timestamp:2}" => "00:01:00.20"
            f"{timestamp:4}" => "00:01:00.2010"

            f"{timestamp2:2}" => "00:05:00.00"
            f"{timestamp2:4}" => "00:05:00"
        """

        delta: timedelta = timedelta(
            seconds=int(self.delta.total_seconds()), microseconds=0
        )
        ms: int = self.delta.microseconds

        if spec == "" or spec == "d":
            return str(delta)
        else:

            def round_to_tens(value: int, tens: int) -> int:
                return int(round(value / (10**tens)))

            try:
                ignore_zero: bool = False
                if spec.endswith("n"):
                    ignore_zero = True
                    spec = spec[:-1]

                if ignore_zero and ms == 0:
                    return str(delta)

                val = parse_int_safely(spec)
                if val is None:
                    raise Exception

                if val > 5 or val <= 0:
                    raise Exception

                # val is between 1 and 5 inclusive
                ms = round_to_tens(ms, 5 - val)

                return "{delta}.{ms:0{val}d}".format(delta=str(delta), ms=ms, val=val)
            except Exception:
                raise ValueError(
                    f"Invalid format specifier '{spec}' for object of type 'Timestamp'"
                )

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.delta == value.delta

        return False

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __lt__(self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.delta < value.delta
        else:
            raise TypeError(
                f"'<' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __le__(self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.delta <= value.delta
        else:
            raise TypeError(
                f"'<=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __gt__(self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.delta > value.delta
        else:
            raise TypeError(
                f"'>' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __ge__(self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.delta >= value.delta
        else:
            raise TypeError(
                f"'>=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __iadd__(self, value: object) -> "Timestamp":
        if isinstance(value, Timestamp):
            self.__delta += value.delta
            return self
        elif isinstance(value, timedelta):
            self.__delta += value
            return self
        else:
            raise TypeError(
                f"'+=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __sub__(self, value: object) -> "Timestamp":
        if isinstance(value, Timestamp):
            result: timedelta = self.__delta - value.delta
            return Timestamp(result)
        else:
            raise TypeError(
                f"'-' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __abs__(self) -> "Timestamp":
        return Timestamp(abs(self.delta))

    def __float__(self) -> float:
        return self.minutes

    def __truediv__(self, value: object) -> float:
        if isinstance(value, Timestamp):
            return self.__delta / value.delta
        elif isinstance(value, float):
            return self / Timestamp.from_minutes(value)
        else:
            raise TypeError(
                f"'/' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )


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
    amount: Optional[Timestamp]


class WAVFile:
    __tmp_file: Optional[Path]
    __file: Path
    __type: FileType
    __status: Status
    __runtime: Optional[Timestamp]

    def __init__(self, file: Path) -> None:
        self.__tmp_file = None
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        type, status, runtime = self.__get_info()
        self.__type = type
        self.__status = status
        self.__runtime = runtime

    def __get_info(self) -> tuple[FileType, Status, Optional[Timestamp]]:
        try:
            metadata = FFProbe(str(self.__file.absolute()))
            for stream in metadata.streams:
                if stream.is_video():
                    return (
                        FileType.video,
                        Status.raw,
                        Timestamp.from_seconds(stream.duration_seconds()),
                    )

            for stream in metadata.streams:
                if stream.is_audio() and stream.codec() == "pcm_s16le":
                    return (
                        FileType.wav,
                        Status.ready,
                        Timestamp.from_seconds(stream.duration_seconds()),
                    )

            for stream in metadata.streams:
                if stream.is_audio():
                    return (
                        FileType.audio,
                        Status.raw,
                        Timestamp.from_seconds(stream.duration_seconds()),
                    )

            return (FileType.audio, Status.raw, None)
        except Exception:
            return (FileType.video, Status.raw, None)

    @property
    def runtime(self) -> Optional[Timestamp]:
        return self.__runtime

    def create_wav_file(
        self,
        options: WAVOptions = {
            "bitrate": 16000,
            "amount": Timestamp.from_minutes(10.0),
        },
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
        elapsed_time: Timestamp = Timestamp.zero()

        runtime: Optional[Timestamp] = options["amount"]
        if runtime is None:
            runtime = self.runtime

        if manager is not None and runtime is not None:
            bar = manager.counter(
                total=runtime,
                count=elapsed_time,
                desc="generating wav",
                unit="minutes",
                leave=False,
                bar_format=WAV_FILE_BAR_FMT,
                color="red",
            )
            bar.update(elapsed_time, force=True)

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
            "stats_period": 0.5,  # in seconds
        }

        if options["amount"] is not None:
            if self.runtime is None or self.runtime >= options["amount"]:
                ffmpeg_options["to"] = str(options["amount"])

        # to use the same format as: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
        ffmpeg: FFmpeg = (
            FFmpeg()
            .option("y")
            .input(self.__file)
            .output(self.__tmp_file, **ffmpeg_options)
        )

        def progress_report(progress: Progress) -> None:
            nonlocal elapsed_time
            if bar is not None:
                delta_time: Timestamp = Timestamp(progress.time) - elapsed_time
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


EXPECTED_LANGUAGES: list[str] = ["de", "en", "it"]


def has_enough_memory() -> tuple[bool, float, float]:
    memory = psutil.virtual_memory()
    percent = memory.free / memory.total

    swap_memory = psutil.swap_memory()
    swap_percent = swap_memory.free / swap_memory.total

    if percent <= 0.05 and swap_percent <= 0.10:
        return (False, percent, swap_percent)

    return (True, percent, swap_percent)


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
    def clear_gpu_cache() -> None:
        gc.collect()
        cuda.empty_cache()

    @staticmethod
    def print_gpu_stat() -> None:
        if not cuda.is_available():
            return None

        Classifier.clear_gpu_cache()

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print("GPU stats:")
        print(f"total    : {naturalsize(info.total, binary=True)}")
        print(f"free     : {naturalsize(info.free, binary=True)}")
        print(f"used     : {naturalsize(info.used, binary=True)}")

    def predict(
        self, wav_file: WAVFile, path: Path, manager: Optional[Manager] = None
    ) -> tuple[Language, float]:
        def get_timestamps(runtime: Optional[Timestamp]) -> list[Optional[Timestamp]]:
            if runtime is None:
                return [
                    Timestamp.from_minutes(x) if x is not None else None
                    for x in [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, None]
                ]

            index: float = 1.0
            result: list[Optional[float]] = [index]
            steps: float = 1.0
            while True:
                index += steps
                if index > runtime.minutes:
                    result.append(None)
                    break

                result.append(index)
                steps += 1.0

            return [
                Timestamp.from_minutes(x) if x is not None else None for x in result
            ]

        timestamps: list[Optional[Timestamp]] = get_timestamps(wav_file.runtime)

        bar: Optional[Any] = None
        if manager is not None:
            bar = manager.counter(
                total=len(timestamps),
                desc="detecting language",
                unit="attempts",
                leave=False,
                color="red",
            )
            bar.update(0, force=True)

        err: str = "after scanning the whole file"
        for i, timestamp in enumerate(timestamps):
            try:
                Classifier.clear_gpu_cache()

                enough_memory, mem, swap = has_enough_memory()
                if not enough_memory:
                    raise MemoryError(f"Not enough memory: {mem} - {swap}")

                delta: Optional[Timestamp] = None
                if timestamp is not None:
                    delta = timestamp

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

                accuracy: float = cast(float, prediction[1].exp().item())
                # The identified language ISO code is given in prediction[3]
                language: Language = Language.from_str_unsafe(
                    cast(str, prediction[3][0])
                )
                print(language, accuracy)

                accuracy_to_reach: float = 0.95 - (0.05 * i)
                if (
                    accuracy > accuracy_to_reach
                    and language.short not in EXPECTED_LANGUAGES
                ):
                    if i < 5:
                        print(
                            f"unexpected language {language} with accuracy {accuracy}, trying re-scan: {i} for file '{path}'"
                        )
                        accuracy = 0.0
                    else:
                        raise ValueError(
                            f"unexpected language {language} with accuracy {accuracy} for file '{path}'"
                        )

                if bar is not None:
                    bar.update()

                if accuracy > accuracy_to_reach:
                    Classifier.clear_gpu_cache()
                    if accuracy_to_reach <= 0.55:
                        raise ValueError(
                            f"Accuracy to reach to low: {accuracy_to_reach}!"
                        )

                    continue

                if bar is not None:
                    bar.close(clear=True)

                Classifier.clear_gpu_cache()

                return (language, accuracy)
            except Exception as exception:
                if isinstance(exception, cuda.OutOfMemoryError):
                    self.__init_classifier(True)
                    continue
                elif isinstance(exception, MemoryError) or isinstance(
                    exception, ValueError
                ):
                    err = str(exception)
                    break
                else:
                    raise exception

        if bar is not None:
            bar.close(clear=True)

        Classifier.clear_gpu_cache()

        print(f"Couldn't get Language of '{path}': {err}")

        return (Language.Unknown(), 0.0)

    def __get_run_opts(self) -> Optional[dict[str, Any]]:
        if not cuda.is_available():
            return None

        Classifier.clear_gpu_cache()

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
