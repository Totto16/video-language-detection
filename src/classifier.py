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
import torch
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
            spec (str): a format string, if empty or "d" no microseconds will be printed
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
        elif isinstance(value, int):
            self += Timestamp.from_seconds(value)
            return self
        else:
            raise TypeError(
                f"'+=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
            )

    def __add__(self, value: object) -> "Timestamp":
        if isinstance(value, Timestamp):
            result: timedelta = self.__delta + value.delta
            return Timestamp(result)
        if isinstance(value, int):
            result2: Timestamp = self + Timestamp.from_seconds(value)
            return result2
        else:
            raise TypeError(
                f"'+' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
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


@dataclass
class Segment:
    start: Optional[Timestamp]
    end: Optional[Timestamp]

    def timediff(self, runtime: Timestamp) -> Timestamp:
        if self.start is None and self.end is None:
            return runtime
        elif self.start is None and self.end is not None:
            return self.end
        elif self.start is not None and self.end is None:
            return runtime - self.start
        elif self.start is not None and self.end is not None:
            return self.end - self.start
        else:
            raise RuntimeError("UNREACHABLE")


@dataclass
class WAVOptions:
    bitrate: int
    segment: Segment


class WAVFile:
    __tmp_file: Optional[Path]
    __file: Path
    __type: FileType
    __status: Status
    __runtime: Timestamp

    def __init__(self, file: Path) -> None:
        self.__tmp_file = None
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        type, status, runtime = self.__get_info()
        self.__type = type
        self.__status = status
        self.__runtime = runtime

    def __get_info(self) -> tuple[FileType, Status, Timestamp]:
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

        raise Exception(f"Unable to get a valid stream from file {self.__file}")

    @property
    def runtime(self) -> Timestamp:
        return self.__runtime

    def create_wav_file(
        self,
        options: WAVOptions = WAVOptions(
            bitrate=16000,
            segment=Segment(
                start=None,
                end=Timestamp.from_minutes(10.0),
            ),
        ),
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

        total_time: Timestamp = options.segment.timediff(self.runtime)
        elapsed_time: Timestamp = (
            Timestamp.zero() if options.segment.start is None else options.segment.start
        )

        if manager is not None:
            bar = manager.counter(
                total=total_time,
                count=elapsed_time,
                desc="generating wav",
                unit="minutes",
                leave=False,
                bar_format=WAV_FILE_BAR_FMT,
                color="red",
            )
            bar.update(Timestamp.zero(), force=True)

        temp_dir: Path = Path("/tmp") / "video_lang_detect"
        if not temp_dir.exists():
            makedirs(temp_dir)

        self.__tmp_file = temp_dir / (self.__file.stem + ".wav")

        if self.__tmp_file.exists():
            remove(str(self.__tmp_file.absolute()))

        ffmpeg_options: dict[str, Any] = {
            "acodec": "pcm_s16le",
            "ar": options.bitrate,
            "ac": 1,
            "stats_period": 0.5,  # in seconds
        }

        if options.segment.end is not None:
            if self.runtime >= options.segment.end:
                ffmpeg_options["to"] = str(options.segment.end)

        input_options: dict[str, Any] = dict()

        if options.segment.start is not None:
            if self.runtime >= options.segment.start:
                input_options["ss"] = str(options.segment.start)

        # to use the same format as: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
        ffmpeg: FFmpeg = (
            FFmpeg()
            .option("y")
            .input(self.__file, **input_options)
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

    def __hash__(self) -> int:
        return hash((self.short, self.long))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Language):
            return self.short == other.short and self.long == other.long

        return False


def has_enough_memory() -> tuple[bool, float, float]:
    memory = psutil.virtual_memory()
    percent = memory.free / memory.total

    swap_memory = psutil.swap_memory()
    swap_percent = swap_memory.free / swap_memory.total

    if percent <= 0.05 and swap_percent <= 0.10:
        return (False, percent, swap_percent)

    return (True, percent, swap_percent)


SEGMENT_LENGTH_IN_SECONDS: int = 30
LANGUAGE_ACCURACY_THRESHOLD: float = 0.95
LANGUAGE_MINIMUM_SCANNED: float = 0.2


PredictionType = list[tuple[float, Language]]


@dataclass
class PredictionBest:
    accuracy: float
    language: Language


class Prediction:
    __data: list[PredictionType]

    def __init__(self, data: Optional[PredictionType] = None) -> None:
        self.__data = []
        if data is not None:
            self.__data.append(data)

    def get_best(self) -> PredictionBest:
        prob_dict: dict[Language, float] = dict()
        for data in self.__data:
            for acc, language in data:
                if prob_dict.get(language) is None:
                    prob_dict[language] = 0.0

                prob_dict[language] += acc

        amount: int = len(self.__data)
        prob: PredictionType = []
        for lan, acc in prob_dict.items():
            prob.append((acc / amount, lan))

        sorted_prob: PredictionType = sorted(prob, key=lambda x: -x[0])

        return PredictionBest(*sorted_prob[0])

    @property
    def data(self) -> list[PredictionType]:
        return self.__data

    def append(self, data: PredictionType) -> None:
        self.__data.append(data)

    def __iadd__(self, value: object) -> "Prediction":
        if isinstance(value, Prediction):
            self.__data.extend(value.data)
            return self
        else:
            raise TypeError(
                f"'+=' not supported between instances of 'Prediction' and '{value.__class__.__name__}'"
            )


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

    def __classify(
        self, wav_file: WAVFile, segment: Segment, manager: Optional[Manager] = None
    ) -> Prediction:
        wav_file.create_wav_file(
            WAVOptions(bitrate=16000, segment=segment),
            force_recreation=True,
            manager=manager,
        )

        # from: https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
        wavs: Any = self.__classifier.load_audio(
            str(wav_file.wav_path().absolute()),
            savedir=str(self.__save_dir.absolute()),
        )

        try:
            Classifier.clear_gpu_cache()

            emb = self.__classifier.encode_batch(wavs, wav_lens=None)
            out_prob = self.__classifier.mods.classifier(emb).squeeze(1)

            # ATTENTION: unsorted
            prob: list[tuple[float, Language]] = [
                (
                    p,
                    Language.from_str_unsafe(
                        self.__classifier.hparams.label_encoder.decode_ndim(index)
                    ),
                )
                for index, p in enumerate(out_prob.exp().tolist()[0])
            ]

            Classifier.clear_gpu_cache()

            return Prediction(prob)

        except Exception as exception:
            if isinstance(exception, cuda.OutOfMemoryError):
                self.__init_classifier(True)
                return self.__classify(wav_file, segment, manager)
            else:
                raise exception

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
    ) -> tuple[PredictionBest, float]:
        def get_segments(runtime: Timestamp) -> list[Segment]:
            result: list[Segment] = []
            current_timestamp: Timestamp = Timestamp.zero()

            while current_timestamp <= runtime:
                end: Optional[Timestamp] = (
                    None
                    if current_timestamp + SEGMENT_LENGTH_IN_SECONDS > runtime
                    else current_timestamp + SEGMENT_LENGTH_IN_SECONDS
                )
                segment: Segment = Segment(current_timestamp, end)
                result.append(segment)
                current_timestamp += SEGMENT_LENGTH_IN_SECONDS

            return result

        segments: list[Segment] = get_segments(wav_file.runtime)

        bar: Optional[Any] = None
        if manager is not None:
            bar = manager.counter(
                total=len(segments),
                desc="detecting language",
                unit="attempts",
                leave=False,
                color="red",
            )
            bar.update(0, force=True)

        prediction: Prediction = Prediction()
        for i, segment in enumerate(segments):
            local_prediction = self.__classify(wav_file, segment, manager)
            prediction += local_prediction

            if bar is not None:
                bar.update()

            amount_scanned: float = 0.0

            if amount_scanned < LANGUAGE_MINIMUM_SCANNED:
                continue

            best = prediction.get_best()
            if best.accuracy < LANGUAGE_ACCURACY_THRESHOLD:
                continue

            if bar is not None:
                bar.close(clear=True)

            return (best, amount_scanned)

        if bar is not None:
            bar.close(clear=True)

        print(f"Couldn't get Language of '{path}'")

        return (PredictionBest(0.0, Language.Unknown()), 0.0)

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
