#!/usr/bin/env python3

import gc
import math
import sys
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import reduce
from math import floor
from pathlib import Path
from shutil import rmtree
from typing import Any, Never, Optional, Self, TypedDict
from warnings import filterwarnings

import psutil
from enlighten import Manager
from helper.ffprobe import ffprobe
from humanize import naturalsize
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from speechbrain.pretrained import EncoderClassifier
from torch import cuda

from ffmpeg import FFmpeg, FFmpegError, Progress  # type: ignore[attr-defined]

filterwarnings("ignore")

WAV_FILE_BAR_FMT = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:2n}/{total:2n} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"

TEMP_MAX_LENGTH = 1


def parse_int_safely(inp: str) -> Optional[int]:
    try:
        return int(inp)
    except ValueError:
        return None


class Timestamp:
    __delta: timedelta

    def __init__(self: Self, delta: timedelta) -> None:
        self.__delta = delta

    @property
    def delta(self: Self) -> timedelta:
        return self.__delta

    @property
    def minutes(self: Self) -> float:
        return self.__delta.total_seconds() / 60.0 + (
            self.__delta.microseconds / (60.0 * 10**6)
        )

    @staticmethod
    def zero() -> "Timestamp":
        return Timestamp(timedelta(seconds=0))

    @staticmethod
    def from_minutes(minutes: float) -> "Timestamp":
        return Timestamp(timedelta(minutes=minutes))

    @staticmethod
    def from_seconds(seconds: float) -> "Timestamp":
        return Timestamp(timedelta(seconds=seconds))

    def __str__(self: Self) -> str:
        return str(self.__delta)

    def __repr__(self: Self) -> str:
        return str(self)

    def __format__(self: Self, spec: str) -> str:
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
            seconds=int(self.__delta.total_seconds()),
            microseconds=0,
        )
        ms: int = self.__delta.microseconds

        if spec in ("", "d"):
            return str(delta)

        def round_to_tens(value: int, tens: int) -> int:
            return int(round(value / (10**tens)))

        def emit_error(reason: str) -> Never:
            msg = f"Invalid format specifier '{spec}' for object of type 'Timestamp': reason {reason}"
            raise ValueError(msg)

        ignore_zero: bool = False
        if spec.endswith("n"):
            ignore_zero = True
            spec = spec[:-1]

        if ignore_zero and ms == 0:
            return str(delta)

        val: Optional[int] = parse_int_safely(spec)
        if val is None:
            msg = f"Couldn't parse int: '{spec}'"
            emit_error(msg)

        if val > 5 or val <= 0:
            msg = f"{val} is out of allowed range 0 < value <= 5"
            emit_error(msg)

        # val is between 1 and 5 inclusive
        ms = round_to_tens(ms, 5 - val)

        return "{delta}.{ms:0{val}d}".format(  # noqa: PLE1300
            delta=str(delta),
            ms=ms,
            val=val,
        )

    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta == value.delta

        return False

    def __ne__(self: Self, value: object) -> bool:
        return not self.__eq__(value)

    def __lt__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta < value.delta

        msg = f"'<' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __le__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta <= value.delta

        msg = f"'<=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __gt__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta > value.delta

        msg = f"'>' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __ge__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta >= value.delta

        msg = f"'>=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __iadd__(self: Self, value: object) -> Self:
        if isinstance(value, Timestamp):
            self.__delta += value.delta
            return self
        if isinstance(value, timedelta):
            self.__delta += value
            return self

        msg = f"'+=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __add__(self: Self, value: object) -> "Timestamp":
        new_value: Timestamp = Timestamp(self.__delta)
        new_value += value
        return new_value

    def __sub__(self: Self, value: object) -> "Timestamp":
        if isinstance(value, Timestamp):
            result: timedelta = self.__delta - value.delta
            return Timestamp(result)
        if isinstance(value, float):
            result2: Timestamp = self - Timestamp.from_minutes(value)
            return result2

        msg = f"'-' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __abs__(self: Self) -> "Timestamp":
        return Timestamp(abs(self.__delta))

    def __float__(self: Self) -> float:
        return self.minutes

    def __truediv__(self: Self, value: object) -> float:
        if isinstance(value, Timestamp):
            return self.__delta / value.delta
        if isinstance(value, float):
            return self / Timestamp.from_minutes(value)

        msg = f"'/' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)


class FileType(Enum):
    wav = "wav"
    video = "video"
    audio = "audio"

    def __str__(self: Self) -> str:
        return f"<FileType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class Status(Enum):
    ready = "ready"
    raw = "raw"

    def __str__(self: Self) -> str:
        return f"<Status: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class Segment:
    start: Optional[Timestamp]
    end: Optional[Timestamp]

    @property
    def is_valid(self: Self) -> bool:
        return not (
            self.start is not None and self.end is not None and self.start > self.end
        )

    def timediff(self: Self, runtime: Timestamp) -> Timestamp:
        if self.start is None and self.end is None:
            return runtime
        if self.start is None and self.end is not None:
            return self.end
        if self.start is not None and self.end is None:
            return runtime - self.start
        if self.start is not None and self.end is not None:
            return self.end - self.start

        msg = "UNREACHABLE"
        raise RuntimeError(msg)


@dataclass
class WAVOptions:
    bitrate: int
    segment: Segment


class FileMetadataError(ValueError):
    pass


class WAVFile:
    __tmp_file: Optional[Path]
    __file: Path
    __type: FileType
    __status: Status
    __runtime: Timestamp

    def __init__(self: Self, file: Path) -> None:
        self.__tmp_file = None
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        info, err = self.__get_info()
        if info is None:
            raise FileMetadataError(err)
        _type, status, runtime = info
        self.__type = _type
        self.__status = status
        self.__runtime = runtime

    def __get_info(
        self: Self,
    ) -> tuple[Optional[tuple[FileType, Status, Timestamp]], str]:
        metadata, err = ffprobe(self.__file.absolute())
        if err is not None or metadata is None:
            with Path("error.log").open(mode="a") as f:
                print(f'"{self.__file}",', file=f)

            err_msg: str = (
                f"Unable to get a valid stream from file '{self.__file}':\n{err}"
            )
            print(
                err_msg,
                file=sys.stderr,
            )

            return None, err_msg

        file_duration: Optional[float] = metadata.file_info.duration_seconds()

        if metadata.is_video():
            video_streams = metadata.video_streams()
            # multiple video makes no sense
            if len(video_streams) > 1:
                msg = "Multiple Video Streams are not supported"
                raise RuntimeError(msg)

            duration = video_streams[0].duration_seconds()
            if duration is None:
                if file_duration is None:
                    return None, "No video duration was found"

                duration = file_duration

            return (
                FileType.video,
                Status.raw,
                Timestamp.from_seconds(duration),
            ), ""

        if metadata.is_audio():
            audio_streams = metadata.audio_streams()
            # TODO: multiple audio streams can happen and shouldn't be a problem ! also every episode should hav an array of streams, with language etc!
            if len(audio_streams) > 1:
                return None, "multiple audio streams are not supported atm"

            duration = audio_streams[0].duration_seconds()
            if duration is None:
                if file_duration is None:
                    return None, "No audio duration was found"

                duration = file_duration

            if audio_streams[0].codec() == "pcm_s16le":
                return (
                    FileType.wav,
                    Status.ready,
                    Timestamp.from_seconds(duration),
                ), ""

            return (
                FileType.audio,
                Status.raw,
                Timestamp.from_seconds(duration),
            ), ""

        return None, "Unknown media type, not video or audio"

    @property
    def runtime(self: Self) -> Timestamp:
        return self.__runtime

    def create_wav_file(
        self: Self,
        _options: Optional[WAVOptions] = None,
        *,
        force_recreation: bool = False,
        manager: Optional[Manager] = None,
    ) -> bool:
        options: WAVOptions = (
            WAVOptions(
                bitrate=16000,
                segment=Segment(
                    start=None,
                    end=Timestamp.from_minutes(10.0),
                ),
            )
            if _options is None
            else _options
        )

        match (self.__status, self.__type, force_recreation):
            case (Status.ready, _, False):
                return False
            case (_, FileType.wav, _):
                return False
            case (Status.raw, _, False):
                return self.__convert_to_wav(options, manager)
            case (_, _, True):
                return self.__convert_to_wav(options, manager)
            case _:  # stupid mypy
                msg = "UNREACHABLE"
                raise RuntimeError(msg)

    @property
    def status(self: Self) -> Status:
        return self.__status

    def __convert_to_wav(
        self: Self,
        options: WAVOptions,
        manager: Optional[Manager] = None,
    ) -> bool:
        bar: Optional[Any] = None

        if not options.segment.is_valid:
            msg = f"Segment is not valid: start > end: {options.segment.start:3n} > {options.segment.end:3n}"
            raise RuntimeError(msg)

        total_time: Timestamp = options.segment.timediff(self.runtime)
        elapsed_time: Timestamp = Timestamp.zero()

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

        temp_dir: Path = Path("/tmp") / "video_lang_detect"  # noqa: S108
        if not temp_dir.exists():
           temp_dir.mkdir(parents=True)

        self.__tmp_file = temp_dir / (self.__file.stem + ".wav")

        if self.__tmp_file.exists():
            self.__tmp_file.unlink(missing_ok=True)

        ffmpeg_options: dict[str, Any] = {}

        if options.segment.start is not None and self.runtime >= options.segment.start:
            ffmpeg_options["ss"] = str(options.segment.start)

        # to use the same format as: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
        ffmpeg_options = {
            **ffmpeg_options,
            "acodec": "pcm_s16le",
            "ar": options.bitrate,
            "ac": 1,
            "stats_period": 0.1,  # in seconds
        }

        if options.segment.end is not None and self.runtime >= options.segment.end:
            ffmpeg_options["to"] = str(options.segment.end)

        input_options: dict[str, Any] = {}

        ffmpeg_proc: FFmpeg = (
            FFmpeg()
            .option("y")
            .input(str(self.__file.absolute()), input_options)
            .output(str(self.__tmp_file.absolute()), ffmpeg_options)
        )

        def progress_report(progress: Progress) -> None:
            nonlocal elapsed_time
            if bar is not None:
                delta_time: Timestamp = Timestamp(progress.time) - elapsed_time
                bar.update(delta_time)
                elapsed_time += progress.time

        ffmpeg_proc.on("progress", progress_report)
        try:
            ffmpeg_proc.execute()
        except FFmpegError as e:
            print(e, file=sys.stderr)
            if bar is not None:
                bar.close(clear=True)
            return False

        self.__status = Status.ready

        if bar is not None:
            bar.close(clear=True)

        return True

    def wav_path(self: Self) -> Path:
        match (self.__status, self.__type):
            case (_, FileType.wav):
                return self.__file
            case (Status.raw, _):
                msg = "Not converted"
                raise RuntimeError(msg)
            case (Status.ready, _):
                if self.__tmp_file is None:
                    msg = "Not converted correctly, temp file is missing"
                    raise RuntimeError(msg)
                return self.__tmp_file
            case _:  # stupid mypy
                msg = "UNREACHABLE"
                raise RuntimeError(msg)

    def __del__(self: Self) -> None:
        if self.__tmp_file is not None and self.__tmp_file.exists():
            self.__tmp_file.unlink(missing_ok=True)


class LanguagePercentageDict(TypedDict):
    language: str
    score: float


@dataclass
class Language:
    short: str
    long: str

    @staticmethod
    def from_str(inp: str) -> Optional["Language"]:
        arr: list[str] = [a.strip() for a in inp.split(":")]
        if len(arr) != 2:
            return None

        return Language(arr[0], arr[1])

    @staticmethod
    def from_str_unsafe(inp: str) -> "Language":
        lan: Optional[Language] = Language.from_str(inp)
        if lan is None:
            msg = f"Couldn't get the Language from str '{inp}'"
            raise RuntimeError(msg)

        return lan

    @staticmethod
    def unknown() -> "Language":
        return Language("un", "Unknown")

    def __str__(self: Self) -> str:
        return f"<Language short: {self.short} long: {self.long}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash((self.short, self.long))

    def __eq__(self: Self, other: object) -> bool:
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


SEGMENT_LENGTH: Timestamp = Timestamp.from_seconds(30)
LANGUAGE_ACCURACY_THRESHOLD: float = 0.95
LANGUAGE_MINIMUM_SCANNED: float = 0.2
LANGUAGE_MINIMUM_AMOUNT: int = 5

PredictionType = list[tuple[float, Language]]


@dataclass
class PredictionBest:
    accuracy: float
    language: Language

    def __str__(self: Self) -> str:
        return f"<PredictionBest accuracy: {self.accuracy} language: {self.language}>"

    def __repr__(self: Self) -> str:
        return str(self)


TRUNCATED_PERCENTILE: float = 0.2


# see: https://en.wikipedia.org/wiki/Mean
class MeanType(Enum):
    arithmetic = "arithmetic"
    geometric = "geometric"
    harmonic = "harmonic"
    truncated = "truncated"


def get_mean(
    mean_type: MeanType,
    values: list[float],
    *,
    normalize_percents: bool = False,
) -> float:
    if normalize_percents:
        percent_mean: float = get_mean(
            mean_type,
            [value * 100.0 for value in values],
            normalize_percents=False,
        )
        return percent_mean / 100.0

    match mean_type:
        case MeanType.arithmetic:
            sum_value: float = sum(values)
            return sum_value / len(values)
        case MeanType.geometric:
            sum_value = reduce(lambda x, y: x * y, values, 1.0)
            return math.pow(sum_value, (1 / len(values)))
        case MeanType.harmonic:
            sum_value = sum(1 / value for value in values)
            return len(values) / sum_value
        case MeanType.truncated:
            start: int = floor(len(values) * TRUNCATED_PERCENTILE)
            end: int = len(values) - start
            new_values: list[float] = [
                value
                for i, value in enumerate(sorted(values))
                if i >= start and i < end
            ]
            return get_mean(
                MeanType.arithmetic,
                new_values,
                normalize_percents=normalize_percents,
            )
        case _:  # stupid mypy
            msg = "UNREACHABLE"
            raise RuntimeError(msg)


class Prediction:
    __data: list[PredictionType]

    def __init__(self: Self, data: Optional[PredictionType] = None) -> None:
        self.__data = []
        if data is not None:
            self.__data.append(data)

    def get_best_list(
        self: Self,
        mean_type: MeanType = MeanType.arithmetic,
    ) -> list[PredictionBest]:
        prob_dict: dict[Language, list[float]] = {}
        for data in self.__data:
            for acc, language in data:
                if prob_dict.get(language) is None:
                    prob_dict[language] = []

                prob_dict[language].append(acc)

        prob: PredictionType = []
        for lan, acc2 in prob_dict.items():
            prob.append((get_mean(mean_type, acc2), lan))

        sorted_prob: PredictionType = sorted(prob, key=lambda x: -x[0])

        return [PredictionBest(*sorted_prob_item) for sorted_prob_item in sorted_prob]

    def get_best(
        self: Self,
        mean_type: MeanType = MeanType.arithmetic,
    ) -> PredictionBest:
        best_list: list[PredictionBest] = self.get_best_list(mean_type)
        if len(best_list) == 0:
            return PredictionBest(0.0, Language.unknown())

        return best_list[0]

    @property
    def data(self: Self) -> list[PredictionType]:
        return self.__data

    def append(self: Self, data: PredictionType) -> None:
        self.__data.append(data)

    def append_other(self: Self, pred: "Prediction") -> None:
        self.__data.extend(pred.data)

    def __iadd__(self: Self, value: object) -> Self:
        if isinstance(value, Prediction):
            self.append_other(value)
            return self

        msg = f"'+=' not supported between instances of 'Prediction' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __add__(self: Self, value: object) -> "Prediction":
        new_value = Prediction()
        new_value += self
        new_value += value
        return new_value


class Classifier:
    __classifier: EncoderClassifier
    __save_dir: Path

    def __init__(self: Self) -> None:
        self.__save_dir = Path(__file__).parent / "tmp"

        self.__init_classifier()

    def __init_classifier(self: Self, *, force_cpu: bool = False) -> None:
        run_opts: Optional[dict[str, Any]] = None
        if not force_cpu:
            run_opts = self.__get_run_opts()

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="model",
            run_opts=run_opts,
        )
        if classifier is None:
            msg = "Couldn't initialize Classifier"
            raise RuntimeError(msg)

        self.__classifier = classifier

    def __classify(
        self: Self,
        wav_file: WAVFile,
        segment: Segment,
        manager: Optional[Manager] = None,
    ) -> Optional[Prediction]:
        result: bool = wav_file.create_wav_file(
            WAVOptions(bitrate=16000, segment=segment),
            force_recreation=True,
            manager=manager,
        )

        if not result and wav_file.status != Status.ready:
            return None

        # from: https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
        wavs: Any = self.__classifier.load_audio(
            str(wav_file.wav_path().absolute()),
            savedir=str(self.__save_dir.absolute()),
        )

        try:
            Classifier.clear_gpu_cache()

            emb = self.__classifier.encode_batch(wavs, wav_lens=None)
            out_prob = self.__classifier.mods.classifier(emb).squeeze(1)

            # NOTE:  ATTENTION - unsorted list
            prob: list[tuple[float, Language]] = [
                (
                    p,
                    Language.from_str_unsafe(
                        self.__classifier.hparams.label_encoder.decode_ndim(index),
                    ),
                )
                for index, p in enumerate(out_prob.exp().tolist()[0])
            ]

            Classifier.clear_gpu_cache()

            return Prediction(prob)

        except RuntimeError as ex:
            if isinstance(ex, cuda.OutOfMemoryError):
                self.__init_classifier(force_cpu=True)
                return self.__classify(wav_file, segment, manager)

            print(ex, file=sys.stderr)
            return None

    @staticmethod
    def clear_gpu_cache() -> None:
        gc.collect()
        cuda.empty_cache()

    @staticmethod
    def print_gpu_stat() -> None:
        if not cuda.is_available():
            return

        Classifier.clear_gpu_cache()

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print("GPU stats:")
        print(f"total    : {naturalsize(info.total, binary=True)}")
        print(f"free     : {naturalsize(info.free, binary=True)}")
        print(f"used     : {naturalsize(info.used, binary=True)}")

    def predict(
        self: Self,
        wav_file: WAVFile,
        path: Path,
        manager: Optional[Manager] = None,
    ) -> tuple[PredictionBest, float]:
        def get_segments(runtime: Timestamp) -> list[Segment]:
            result: list[Segment] = []
            current_timestamp: Timestamp = Timestamp.zero()

            while current_timestamp <= runtime:
                end: Optional[Timestamp] = (
                    None
                    if current_timestamp + SEGMENT_LENGTH > runtime
                    else current_timestamp + SEGMENT_LENGTH
                )
                segment: Segment = Segment(current_timestamp, end)
                result.append(segment)
                # ATTENTION: don't use +=, since that doesn't create a new object!
                current_timestamp = current_timestamp + SEGMENT_LENGTH

            return result

        segments: list[Segment] = get_segments(wav_file.runtime)

        bar: Optional[Any] = None
        if manager is not None:
            bar = manager.counter(
                total=len(segments),
                desc="detecting language",
                unit="fragments",
                leave=False,
                color="red",
            )
            bar.update(0, force=True)

        prediction: Prediction = Prediction()
        for i, segment in enumerate(segments):
            local_prediction = self.__classify(wav_file, segment, manager)
            if local_prediction is not None:
                prediction += local_prediction

            if bar is not None:
                bar.update()

            amount_scanned: float = 0.0  # TODO: calculate that

            if TEMP_MAX_LENGTH < LANGUAGE_MINIMUM_AMOUNT:
                break

            if (
                # amount_scanned < LANGUAGE_MINIMUM_SCANNED
                #  and
                i
                < LANGUAGE_MINIMUM_AMOUNT
            ):
                continue

            # TODO temporary to scan fast scannable first!
            if i + 1 > TEMP_MAX_LENGTH:
                break

            best: PredictionBest = prediction.get_best(MeanType.truncated)
            if best.accuracy < LANGUAGE_ACCURACY_THRESHOLD:
                if i + 1 == len(segments):
                    if best.accuracy < 0.55:
                        continue
                else:
                    continue

            if best.language == Language.unknown():
                continue

            if bar is not None:
                bar.close(clear=True)

            return (best, amount_scanned)

        # END OF FOR LOOP

        if bar is not None:
            bar.close(clear=True)

        best = prediction.get_best(MeanType.truncated)
        print(f"Couldn't get Language of '{path}': {best}", file=sys.stderr)

        return (PredictionBest(0.0, Language.unknown()), 0.0)

    def __get_run_opts(self: Self) -> Optional[dict[str, Any]]:
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

    def __del__(self: Self) -> None:
        if self.__save_dir.exists():
            if self.__save_dir.is_file():
                self.__save_dir.rmdir()
            else:
                rmtree(str(self.__save_dir.absolute()), ignore_errors=True)
