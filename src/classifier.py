import gc
import math
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from logging import Logger
from math import floor
from pathlib import Path
from shutil import rmtree
from typing import Any, Optional, Self, TypedDict

import psutil
import torchaudio
from apischema import schema
from enlighten import Manager
from ffmpeg.ffmpeg import FFmpeg, FFmpegError
from ffmpeg.progress import Progress
from humanize import naturalsize
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from torch import cuda

from helper.ffprobe import ffprobe, ffprobe_check
from helper.log import get_logger, setup_global_logger
from helper.timestamp import Timestamp
from helper.translation import get_translator

setup_global_logger()

from speechbrain.inference.classifiers import EncoderClassifier  # noqa: E402

WAV_FILE_BAR_FMT = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:2n}/{total:2n} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"

logger: Logger = get_logger()

_ = get_translator()


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
            logger.error(err_msg)

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
        except FFmpegError:
            logger.exception("FFmpeg exception")
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
        return self.long

    def __repr__(self: Self) -> str:
        return f"<Language short: {self.short!r} long: {self.long!r}>"

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


PredictionType = list[tuple[float, Language]]


@dataclass
class PredictionBest:
    accuracy: float
    language: Language

    def __str__(self: Self) -> str:
        return f"{self.language!s} ({self.accuracy:.2%})"

    def __repr__(self: Self) -> str:
        return (
            f"<PredictionBest accuracy: {self.accuracy!r} language: {self.language!r}>"
        )


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


# TODO: relativate to the root path
def relative_path_str(path: Path) -> str:
    return str(path)


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


@dataclass()
class ClassifierOptions:
    segment_length: Optional[Timestamp]
    accuracy_threshold: Optional[float]
    final_accuracy_threshold: Optional[float]
    minimum_scanned: Optional[float]
    scan_until: Optional[float]


@dataclass()
class ClassifierOptionsParsed:
    segment_length: Timestamp = field(metadata=schema())
    accuracy_threshold: float = field(metadata=schema(min=0.0, max=1.0))
    final_accuracy_threshold: float = field(metadata=schema(min=0.0, max=1.0))
    minimum_scanned: float
    scan_until: Optional[float]

    @staticmethod
    def default() -> "ClassifierOptionsParsed":
        return ClassifierOptionsParsed(
            segment_length=Timestamp.from_seconds(30),
            accuracy_threshold=0.95,
            final_accuracy_threshold=0.55,
            minimum_scanned=0.2,
            scan_until=None,
        )


def is_percentage(value: float) -> bool:
    return value >= 0.0 and value <= 1.0


class Classifier:
    __save_dir: Path
    __options: ClassifierOptionsParsed
    __classifier: EncoderClassifier

    def __init__(
        self: Self,
        options: Optional[ClassifierOptions] | ClassifierOptionsParsed = None,
    ) -> None:
        self.__save_dir = Path(__file__).parent / "tmp"
        self.__options = self.__parse_options(options)
        self.__classifier = self.__init_classifier()

    def __parse_options(
        self: Self,
        options: Optional[ClassifierOptions] | ClassifierOptionsParsed,
    ) -> ClassifierOptionsParsed:
        total_options: ClassifierOptionsParsed = ClassifierOptionsParsed.default()

        if options is not None:
            total_options.segment_length = (
                options.segment_length
                if options.segment_length is not None
                else total_options.segment_length
            )
            total_options.accuracy_threshold = (
                options.accuracy_threshold
                if options.accuracy_threshold is not None
                else total_options.accuracy_threshold
            )
            total_options.final_accuracy_threshold = (
                options.final_accuracy_threshold
                if options.final_accuracy_threshold is not None
                else total_options.final_accuracy_threshold
            )
            total_options.minimum_scanned = (
                options.minimum_scanned
                if options.minimum_scanned is not None
                else total_options.minimum_scanned
            )
            total_options.scan_until = (
                options.scan_until
                if options.scan_until is not None
                else total_options.scan_until
            )

        if not is_percentage(total_options.accuracy_threshold):
            msg = f"Option 'accuracy_threshold' has to be in percentage (0.0 - 1.0) but was: {total_options.accuracy_threshold}"
            raise RuntimeError(msg)

        if not is_percentage(total_options.final_accuracy_threshold):
            msg = f"Option 'final_accuracy_threshold' has to be in percentage (0.0 - 1.0) but was: {total_options.final_accuracy_threshold}"
            raise RuntimeError(msg)

        if not is_percentage(total_options.minimum_scanned):
            msg = f"Option 'minimum_scanned' has to be in percentage (0.0 - 1.0) but was: {total_options.minimum_scanned}"
            raise RuntimeError(msg)

        if total_options.scan_until is not None and not is_percentage(
            total_options.scan_until,
        ):
            msg = f"Option 'scan_until' has to be in percentage (0.0 - 1.0) but was: {total_options.scan_until}"
            raise RuntimeError(msg)

        return total_options

    def __init_classifier(self: Self, *, force_cpu: bool = False) -> EncoderClassifier:
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

        self.__check_audio_backends()
        self.__check_ffprobe()

        return classifier

    def __check_audio_backends(self: Self) -> None:
        backends = torchaudio.list_audio_backends()
        if len(backends) == 0:
            msg = "Couldn't find any audio backends for torchaudio"
            raise RuntimeError(msg)

        logger.debug("Found audio backends: %s", str(backends))

    def __check_ffprobe(self: Self) -> None:
        is_ffprobe_present = ffprobe_check()
        if not is_ffprobe_present:
            msg = "FFProbe not installed"
            raise RuntimeError(msg)

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

            logger.exception("Classify file")
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
        logger.info("GPU stats:")
        logger.info("total : %s", naturalsize(info.total, binary=True))
        logger.info("free : %s", naturalsize(info.free, binary=True))
        logger.info("used : %s", naturalsize(info.used, binary=True))

    def predict(
        self: Self,
        wav_file: WAVFile,
        path: Path,
        manager: Optional[Manager] = None,
    ) -> tuple[PredictionBest, float]:
        def get_segments(runtime: Timestamp) -> list[tuple[Segment, Timestamp]]:
            result: list[tuple[Segment, Timestamp]] = []
            current_timestamp: Timestamp = Timestamp.zero()

            while current_timestamp <= runtime:
                end: Optional[Timestamp] = (
                    None
                    if current_timestamp + self.__options.segment_length > runtime
                    else current_timestamp + self.__options.segment_length
                )
                segment: Segment = Segment(current_timestamp, end)
                result.append((segment, end if end is not None else runtime))
                # ATTENTION: don't use +=, since that doesn't create a new object!
                current_timestamp = current_timestamp + self.__options.segment_length

            return result

        # This guards for cases, where scanning is useless, e.g. when you want 20 % to be scanned, for a valid result, but scan until 10 %
        scan_nothing = (
            self.__options.scan_until is not None
            and self.__options.scan_until < self.__options.minimum_scanned
        )

        segments: list[tuple[Segment, Timestamp]] = (
            [] if scan_nothing else get_segments(wav_file.runtime)
        )

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
        for i, (segment, scanned_length) in enumerate(segments):
            local_prediction = self.__classify(wav_file, segment, manager)
            if local_prediction is not None:
                prediction += local_prediction

            if bar is not None:
                bar.update()

            amount_scanned: float = scanned_length / wav_file.runtime

            if amount_scanned < self.__options.minimum_scanned:
                continue

            if (
                self.__options.scan_until is not None
                and amount_scanned >= self.__options.scan_until
            ):
                break

            best: PredictionBest = prediction.get_best(MeanType.truncated)
            if best.accuracy < self.__options.accuracy_threshold:
                if i + 1 == len(segments):
                    if best.accuracy < self.__options.final_accuracy_threshold:
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

        msg = _("Couldn't get Language of '{path}': Best was {best}").format(
            path=relative_path_str(path), best=str(best),
        )

        logger.error(msg)

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
