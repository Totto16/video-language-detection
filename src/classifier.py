import gc
import math
import re
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from shutil import rmtree
from types import TracebackType
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Self,
    TypedDict,
    assert_never,
    cast,
    override,
)

import numpy as np
import torch
from apischema import deserializer, schema, serializer
from enlighten import Manager
from ffmpeg.errors import FFmpegError
from ffmpeg.ffmpeg import FFmpeg
from ffmpeg.progress import Progress
from numpy.polynomial.polynomial import Polynomial
from speechbrain.dataio import audio_io

from content.general import MissingOverrideError
from content.language import Language
from content.language_picker import LanguagePicker
from content.prediction import MeanType, Prediction, PredictionBest
from helper.apischema import OneOf
from helper.devices import AllocatorType, DeviceManager
from helper.ffprobe import ffprobe, ffprobe_check
from helper.log import get_logger, setup_global_logger
from helper.result import Result
from helper.timestamp import (
    ConfigTimeStamp,
    Timestamp,
    TimestampCompat,
    parse_int_safely,
)
from helper.translation import get_translator

setup_global_logger()

from speechbrain.inference.classifiers import EncoderClassifier  # noqa: E402

WAV_FILE_BAR_FMT = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:2n}/{total:2n} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"

logger: Logger = get_logger()

_ = get_translator()


class MemoryPatternType(Enum):
    linear = "linear"
    quadratic = "quadratic"


class MemoryPattern:
    pattern_type: MemoryPatternType

    def __init__(self: Self, pattern_type: MemoryPatternType) -> None:
        self.typepattern_type = pattern_type

    def get_seconds_for_memory_amount(
        self: Self,
        memory_amount: int,  # noqa: ARG002
    ) -> Optional[float]:
        raise MissingOverrideError

    def to_constructor_str(self: Self) -> str:
        raise MissingOverrideError


@dataclass
class LinearCoeffs:
    c: float  # x ^ 0
    m: float  # x ^ 1


class MemoryPatternLinear(MemoryPattern):
    """
    linear  =>
    y = m * x + c
    """

    __coeffs: LinearCoeffs

    def __init__(self: Self, coeffs: LinearCoeffs) -> None:
        super().__init__(MemoryPatternType.linear)
        self.__coeffs = coeffs

    @staticmethod
    def from_poly_coefs(
        coeffs: tuple[float, float],
    ) -> "MemoryPatternLinear":
        l_coeffs = LinearCoeffs(c=coeffs[0], m=coeffs[1])
        return MemoryPatternLinear(coeffs=l_coeffs)

    @override
    def get_seconds_for_memory_amount(
        self: Self,
        memory_amount: int,
    ) -> Optional[float]:
        """
        x = seconds,
        y = memory_used

        y = m * x + c
        x = (y - c) / m
        """

        result = (memory_amount - self.__coeffs.c) / self.__coeffs.m

        if result < 0:
            return None

        return result

    def to_constructor_str(self: Self) -> str:
        return f"MemoryPatternLinear(coeffs=LinearCoeffs(c={self.__coeffs.c}, m={self.__coeffs.m}))"


@dataclass
class QuadraticCoeffs:
    c: float  # x ^ 0
    b: float  # x ^ 1
    a: float  # x ^ 2


class MemoryPatternQuadratic(MemoryPattern):
    """
    quadratic  =>
    y = a * x^2 + b * x + c
    """

    __coeffs: QuadraticCoeffs

    def __init__(self: Self, coeffs: QuadraticCoeffs) -> None:
        super().__init__(MemoryPatternType.quadratic)
        self.__coeffs = coeffs

    @staticmethod
    def from_poly_coefs(
        coeffs: tuple[float, float, float],
    ) -> "MemoryPatternQuadratic":
        q_coeffs = QuadraticCoeffs(c=coeffs[0], b=coeffs[1], a=coeffs[2])
        return MemoryPatternQuadratic(coeffs=q_coeffs)

    @override
    def get_seconds_for_memory_amount(
        self: Self,
        memory_amount: int,
    ) -> Optional[float]:
        """
        x = seconds,
        y = memory_used

        y = a * x^2 + b * x + c
        x = D1/2 / 2 a
        D1/2 = -b +- sq(DISC)
        DISC = b^2 - 4ac
        """

        disc = (self.__coeffs.b**2) - (4 * self.__coeffs.a * self.__coeffs.c)

        if disc < 0:
            return None

        if disc == 0.0:
            # only one solution
            d1 = -self.__coeffs.b
            res = d1 / (2 * self.__coeffs.a)

            if res < 0:
                return None

            return res

        # two solutions
        sq_res = math.sqrt(disc)
        min_b = -self.__coeffs.b
        d1, d2 = (min_b + sq_res, min_b - sq_res)

        _2a = 2 * self.__coeffs.a

        sol1, sol2 = (d1 / _2a, d2 / _2a)

        max_sol = max(sol1, sol2)

        if max_sol < 0:
            return None

        return max_sol

    def to_constructor_str(self: Self) -> str:
        return f"MemoryPatternQuadratic(coeffs=QuadraticCoeffs(c={self.__coeffs.c}, b={self.__coeffs.b}, a={self.__coeffs.a}))"


@dataclass
class Model:
    name: str
    sample_count: int
    source: str
    bitrate: int
    memory_pattern: Optional[MemoryPattern] = (
        None  # if this is None, it is inferred and printed, so that you can hardcode it!
    )


class RunOpts(TypedDict, total=False):
    device: torch.device
    data_parallel_count: int
    data_parallel_backend: bool
    distributed_launch: bool
    distributed_backend: str
    jit: str
    jit_module_keys: Optional[str]
    compule: str
    compile_module_keys: str
    compile_mode: str
    compile_using_fullgraph: str
    compile_using_dynamic_shape_tracing: str


voxlingua107_ecapa_model: Model = Model(
    name="voxlingua107",
    sample_count=107,
    source="speechbrain/lang-id-voxlingua107-ecapa",
    bitrate=16000,
    memory_pattern=MemoryPatternLinear(
        coeffs=LinearCoeffs(c=121287679.99999952, m=12845499.313230773),
    ),
)


MODEL_SAVEDIR: str = "model"


def get_model_run_opts(device_manager: DeviceManager) -> Optional[RunOpts]:
    device = device_manager.get_torch_device()

    if device is None:
        msg = "GPU found, but not usable in torch"
        raise RuntimeError(msg)

    run_ops: RunOpts
    if device.type == "cpu":
        run_ops = {
            "device": device,
        }
    elif device.type == "cuda":
        run_ops = {
            "device": device,
            "data_parallel_count": -1,
            "data_parallel_backend": True,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }
    else:
        msg = f"Not supported torch device: '{device}'"
        raise RuntimeError(msg)

    return run_ops


def get_classifier_from_model(
    model: Model,
    run_opts: Optional[RunOpts],
) -> EncoderClassifier:
    classifier: Optional[EncoderClassifier] = EncoderClassifier.from_hparams(
        source=model.source,
        savedir=MODEL_SAVEDIR,
        run_opts=run_opts,
    )
    if classifier is None:
        msg = "Couldn't initialize Classifier"
        raise RuntimeError(msg)

    classifier.hparams.label_encoder.expect_len(model.sample_count)

    return classifier


MAX_MEAN_VARIANCE: float = 0.05


def get_memory_pattern_for_model(model: Model) -> Optional[MemoryPattern]:
    @dataclass
    class CalibrateResult:
        peak_bytes: int
        probe_sec: int

    @dataclass
    class SolvedPolynomial:
        p: Polynomial
        deg: int
        coefs: list[float]
        mean: float

    def calibrate_ecapa(
        classifier: EncoderClassifier,
        sample_rate: int,
        probe_sec: int,
        device: Optional[torch.device],
    ) -> CalibrateResult:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

        waveform = torch.randn(1, probe_sec * sample_rate, device=device).cuda(
            device=device,
        )

        with torch.no_grad():
            classifier.encode_batch(waveform)

        peak_bytes: int = torch.cuda.max_memory_allocated(device=device)
        return CalibrateResult(peak_bytes=peak_bytes, probe_sec=probe_sec)

    device_manager: DeviceManager = DeviceManager()

    run_opts: Optional[RunOpts] = get_model_run_opts(device_manager=device_manager)

    device = run_opts["device"] if run_opts is not None else None

    classifier: Optional[EncoderClassifier] = EncoderClassifier.from_hparams(
        source=model.source,
        savedir=MODEL_SAVEDIR,
        run_opts=run_opts,
    )
    if classifier is None:
        msg = "Couldn't initialize Classifier"
        raise RuntimeError(msg)

    classifier.hparams.label_encoder.expect_len(model.sample_count)

    results = [
        calibrate_ecapa(
            classifier=classifier,
            sample_rate=model.bitrate,
            probe_sec=5 * (i + 1),
            device=device,
        )
        for i in range(25)
    ]

    torch.cuda.empty_cache()
    x = np.array(list(r.probe_sec for r in results))
    y = np.array(list(r.peak_bytes for r in results))

    polynomials: list[SolvedPolynomial] = []
    for deg in [1, 2]:
        p = Polynomial.fit(x=x, y=y, deg=deg)

        p_std = p.convert()
        coefs = p_std.coef

        y_fit = p_std(x)
        residuals = y - y_fit
        residuals_pct = residuals / y

        mean_res = residuals_pct.mean()

        polynomial: SolvedPolynomial = SolvedPolynomial(
            p=p,
            deg=deg,
            coefs=coefs.tolist(),
            mean=float(mean_res),
        )

        polynomials.append(polynomial)

    def sort_by_mean(p: SolvedPolynomial) -> float:
        return p.mean

    polynomials.sort(key=sort_by_mean)

    best_p = polynomials[0]

    if abs(best_p.mean) > MAX_MEAN_VARIANCE:
        return None

    memory_pattern: MemoryPattern

    match best_p.deg:
        case 1:
            memory_pattern = MemoryPatternLinear.from_poly_coefs(
                cast(tuple[float, float], best_p.coefs),
            )
        case 2:
            memory_pattern = MemoryPatternQuadratic.from_poly_coefs(
                cast(tuple[float, float, float], best_p.coefs),
            )
        case _:
            return None

    return memory_pattern


class WavFile:
    pass


class FileType(Enum):
    video = "video"
    audio = "audio"

    def __str__(self: Self) -> str:
        return f"<FileType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


class ConversionStatus(Enum):
    ready = "ready"
    raw = "raw"

    def __str__(self: Self) -> str:
        return f"<ConversionStatus: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class FileAnnotation:
    type: FileType
    status: ConversionStatus


type FileStatus = WavFile | FileAnnotation


@dataclass
class Segment:
    index: int
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


class OriginalWavFileManager(AbstractContextManager[Path]):
    __file: Path

    def __init__(self: Self, file: Path) -> None:
        self.__file = file

    @override
    def __enter__(self: Self) -> Path:
        return self.__file

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        return False


class GeneratedWavFileManager(AbstractContextManager[Path]):
    __file: Path
    __released: bool

    def __init__(self: Self, file: Path) -> None:
        self.__file = file
        self.__released = False

    @override
    def __enter__(self: Self) -> Path:
        return self.__file

    def release(self: Self) -> "GeneratedWavFileManager":
        self.__released = True
        return GeneratedWavFileManager(self.__file)

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        if self.__released:
            return False

        if self.__file.exists():
            self.__file.unlink(missing_ok=True)
        return False


WAVFile__InfoResult = Result[tuple[FileStatus, Timestamp], str]
WavFile__WavFileResult = Result[AbstractContextManager[Path], tuple[()]]


class WAVFile:
    __file: Path
    __status: FileStatus
    __runtime: Timestamp

    def __init__(self: Self, file: Path) -> None:
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        info = self.__get_info()
        if info.is_err():
            raise FileMetadataError(info.get_err())
        status, runtime = info.get_ok()
        self.__status = status
        self.__runtime = runtime

    def __get_info(
        self: Self,
    ) -> WAVFile__InfoResult:
        metadata, err = ffprobe(self.__file.absolute())
        if err is not None or metadata is None:
            with Path("error.log").open(mode="a") as f:
                print(f'"{self.__file}",', file=f)

            err_msg: str = (
                f"Unable to get a valid stream from file '{self.__file}':\n{err}"
            )
            logger.error(err_msg)

            return WAVFile__InfoResult.err(err_msg)

        file_duration: Optional[float] = metadata.file_info.duration_seconds()

        if metadata.is_video():
            video_streams = metadata.video_streams()
            # only one video stream supported
            if len(video_streams) != 1:
                msg = f"Only One Video Stream supported, but got {len(video_streams)}"
                raise RuntimeError(msg)

            duration = video_streams[0].duration_seconds()
            if duration is None:
                if file_duration is None:
                    return WAVFile__InfoResult.err("No video duration was found")

                duration = file_duration

            # check if we have enough audio streams
            audio_streams = metadata.audio_streams()

            # only one audio stream supported atm
            if len(audio_streams) == 1:
                return WAVFile__InfoResult.ok(
                    (
                        FileAnnotation(
                            type=FileType.video,
                            status=ConversionStatus.raw,
                        ),
                        Timestamp.from_seconds(duration),
                    ),
                )

            if len(audio_streams) == 0:
                err_msg = f"Got a Video with no Audio Stream, aborting: '{self.__file}'"
                return WAVFile__InfoResult.err(err_msg)

            msg = f"Got a Video with {len(audio_streams)} Audio Streams, aborting: '{self.__file}'"
            raise RuntimeError(msg)

        if metadata.is_audio():
            audio_streams = metadata.audio_streams()

            # only one audio stream supported atm
            if len(audio_streams) != 1:
                msg = f"Only One Audio Stream supported, but got {len(audio_streams)}: '{self.__file}'"
                raise RuntimeError(msg)

            duration = audio_streams[0].duration_seconds()
            if duration is None:
                if file_duration is None:
                    return WAVFile__InfoResult.err("No audio duration was found")

                duration = file_duration

            if audio_streams[0].codec() == "pcm_s16le":
                return WAVFile__InfoResult.ok(
                    (
                        WavFile(),
                        Timestamp.from_seconds(duration),
                    ),
                )

            return WAVFile__InfoResult.ok(
                (
                    FileAnnotation(type=FileType.audio, status=ConversionStatus.raw),
                    Timestamp.from_seconds(duration),
                ),
            )

        return WAVFile__InfoResult.err("Unknown media type, not video or audio")

    @property
    def runtime(self: Self) -> Timestamp:
        return self.__runtime

    def create_wav_file(
        self: Self,
        options: WAVOptions,
        *,
        force_recreation: bool = False,
        manager: Optional[Manager] = None,
    ) -> WavFile__WavFileResult:

        if force_recreation:
            return self.__convert_to_wav(options, manager)

        match self.__status:
            case WavFile():
                return WavFile__WavFileResult.ok(OriginalWavFileManager(self.__file))
            case FileAnnotation(_, status):
                match status:
                    case ConversionStatus.ready:
                        msg = "File already converted"
                        raise RuntimeError(msg)
                    case ConversionStatus.raw:
                        return self.__convert_to_wav(options, manager)
            case _:
                assert_never(self.__status)

    def __get_temp_file_manager(self: Self, index: int) -> GeneratedWavFileManager:
        tmp_file_managed = tempfile.NamedTemporaryFile(  # noqa: SIM115
            delete=False,
            suffix=f".{self.__file.stem}_{index}.wav",
        )

        tmp_file_path = Path(tmp_file_managed.name)
        tmp_file_managed.close()

        return GeneratedWavFileManager(tmp_file_path)

    def __convert_to_wav(
        self: Self,
        options: WAVOptions,
        manager: Optional[Manager] = None,
    ) -> WavFile__WavFileResult:
        if isinstance(self.__status, WavFile):
            return WavFile__WavFileResult.ok(OriginalWavFileManager(self.__file))

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

        wav_manager = self.__get_temp_file_manager(options.segment.index)

        with wav_manager as tmp_file:

            output_options: dict[str, Any] = {}

            if (
                options.segment.start is not None
                and self.runtime >= options.segment.start
            ):
                output_options["ss"] = str(options.segment.start)

            # to use the same format as: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
            # TODO. also depend on the model: Model somehow
            output_options = {
                **output_options,
                # mapping options
                "map": "0:a:0",  # first input stream -> audio -> first (audio) stream
                # audio options
                "codec:a": "pcm_s16le",
                "ar": options.bitrate,
                "ac": 1,  # number of output channels
            }

            if options.segment.end is not None and self.runtime >= options.segment.end:
                output_options["to"] = str(options.segment.end)

            input_options: dict[str, Any] = {}

            ffmpeg_proc: FFmpeg = (
                FFmpeg()
                .option("y")
                .option("stats_period", 0.1)  # in seconds
                .input(str(self.__file.absolute()), input_options)
                .output(str(tmp_file.absolute()), output_options)
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
                msg = f"FFmpeg exception in file {self.__file.absolute()}"
                logger.exception(msg)
                if bar is not None:
                    bar.close(clear=True)
                return WavFile__WavFileResult.err(error=())

            match self.__status:
                case FileAnnotation(type_, _):
                    self.__status = FileAnnotation(
                        type=type_,
                        status=ConversionStatus.ready,
                    )
                case _:
                    assert_never(self.__status)

            if bar is not None:
                bar.close(clear=True)

            return WavFile__WavFileResult.ok(wav_manager.release())


# TODO: relativate to the root path
def relative_path_str(path: Path) -> str:
    return str(object=path)


def is_percentage(value: float) -> bool:
    return value >= 0.0 and value <= 1.0


PERCENTAGE_PATTERN = r"^(\d{1,3})(?:\.(\d+))?%$"


@schema(pattern=PERCENTAGE_PATTERN)
class AdvancedPercentage:
    __value: float

    def __init__(self: Self, value: float, description: str = "") -> None:

        option_name = "" if description == "" else f" '{description}'"

        if not is_percentage(value):
            msg = f"Option {option_name} has to be in percentage (0.0 - 1.0) but was: {value}"
            raise RuntimeError(msg)
        self.__value = value

    @staticmethod
    def safe_from_value(value: float) -> Optional["AdvancedPercentage"]:

        if not is_percentage(value):
            return None

        return AdvancedPercentage(value)

    @property
    def value(self: Self) -> float:
        return self.__value

    @serializer
    def serialize(self: Self) -> str:
        return str(self)

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "AdvancedPercentage":
        match = re.match(PERCENTAGE_PATTERN, inp)

        if match is None:
            msg = f"Invalid pattern for AdvancedPercentage: got '{inp}', this didn't match the pattern {PERCENTAGE_PATTERN}"
            raise TypeError(msg)

        match_args = match.groups(default=None)

        if len(match_args) != 2:
            msg = "Implementation error for AdvancedPercentage: case 1"
            raise TypeError(msg)

        (match_arg1, match_arg2) = match_args

        if match_arg1 is None:
            msg = "Implementation error for AdvancedPercentage: case 2"
            raise TypeError(msg)

        match_int1 = parse_int_safely(match_arg1)
        if match_int1 is None:
            msg = "Implementation error for AdvancedPercentage: case 3"
            raise TypeError(msg)

        match_int2 = 0 if match_arg2 is None else parse_int_safely(match_arg2)
        if match_int2 is None:
            msg = "Implementation error for AdvancedPercentage: case 4"
            raise TypeError(msg)

        value: float = ((float(match_int2) / 100.0) + float(match_int1)) / 100.0

        final_value = AdvancedPercentage.safe_from_value(value)

        if final_value is None:
            msg = f"Option has to be in percentage (0.0 - 1.0) but was: {value}"
            raise TypeError(msg)

        return final_value

    def __str__(self: Self) -> str:
        return f"{self.__value:.2%}"

    def __repr__(self: Self) -> str:
        return repr(self.__value)

    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, AdvancedPercentage):
            return self.__value == value.value

        return False

    def __ne__(self: Self, value: object) -> bool:
        return not self.__eq__(value)

    @staticmethod
    def __get_value_for_comparions(value: object, comparions_desc: str) -> float:
        if isinstance(value, AdvancedPercentage):
            return value.value

        if isinstance(value, float):
            return value

        msg = f"'{comparions_desc}' not supported between instances of 'AdvancedPercentage' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __lt__(self: Self, value: object) -> bool:
        return self.__value < AdvancedPercentage.__get_value_for_comparions(value, "<")

    def __le__(self: Self, value: object) -> bool:
        return self.__value <= AdvancedPercentage.__get_value_for_comparions(
            value,
            "<=",
        )

    def __gt__(self: Self, value: object) -> bool:
        return self.__value > AdvancedPercentage.__get_value_for_comparions(value, ">")

    def __ge__(self: Self, value: object) -> bool:
        return self.__value >= AdvancedPercentage.__get_value_for_comparions(
            value,
            ">=",
        )


SimplePercentage = Annotated[float, schema(min=0.0, max=1.0)]

Percentage = Annotated[SimplePercentage | AdvancedPercentage, OneOf]


def to_advanced_percentage(
    percentage: Percentage,
    description: str,
) -> AdvancedPercentage:
    if isinstance(percentage, AdvancedPercentage):
        return percentage

    return AdvancedPercentage(percentage, description)


# TODO: is there a better way?
class AccuracySettingsDict(TypedDict, total=False):
    normal_threshold: Percentage
    final_threshold: Percentage
    use_picker_at_end: bool


class AccuracySettingsDictTotal(TypedDict, total=True):
    normal_threshold: AdvancedPercentage
    final_threshold: AdvancedPercentage
    use_picker_at_end: bool


# TODO: is there a better way?
class ScanConfigDict(TypedDict, total=False):
    minimum: Percentage
    maximum: Percentage


class ScanConfigDictTotal(TypedDict, total=True):
    minimum: AdvancedPercentage
    maximum: Optional[AdvancedPercentage]


@dataclass
class ManualBatchSettings:
    batch_type: Literal["manual"]
    amount: Timestamp


@dataclass
class AutoBatchSettings:
    batch_type: Literal["auto"]
    keep_free: Percentage

    @staticmethod
    def default() -> "AutoBatchSettings":
        return AutoBatchSettings(
            batch_type="auto",
            keep_free=0.1,
        )


type BatchSettings = ManualBatchSettings | AutoBatchSettings


@dataclass()
class ClassifierOptions:
    batch_settings: BatchSettings
    accuracy: AccuracySettingsDictTotal
    scan_config: ScanConfigDictTotal

    @staticmethod
    def default() -> "ClassifierOptions":
        default_accuracy = AccuracySettingsDictTotal(
            normal_threshold=AdvancedPercentage(0.95, "normal_threshold"),
            final_threshold=AdvancedPercentage(0.75, "final_threshold"),
            use_picker_at_end=True,
        )

        default_scan_config = ScanConfigDictTotal(
            minimum=AdvancedPercentage(0.4, "minimum"),
            maximum=None,
        )

        return ClassifierOptions(
            batch_settings=AutoBatchSettings.default(),
            accuracy=default_accuracy,
            scan_config=default_scan_config,
        )


@dataclass()
class ClassifierOptionsConfig:
    batch_settings: Annotated[
        Optional[ManualBatchSettings | AutoBatchSettings | ConfigTimeStamp],
        OneOf,
    ]
    accuracy: Annotated[Optional[AccuracySettingsDict], OneOf]
    scan_config: Annotated[Optional[ScanConfigDict], OneOf]

    @staticmethod
    def default() -> "ClassifierOptionsConfig":
        config_defaults: ClassifierOptions = ClassifierOptions.default()

        default_accuracy = AccuracySettingsDict(
            normal_threshold=config_defaults.accuracy["normal_threshold"],
            final_threshold=config_defaults.accuracy["final_threshold"],
            use_picker_at_end=config_defaults.accuracy["use_picker_at_end"],
        )

        default_scan_config = ScanConfigDict(
            minimum=config_defaults.scan_config["minimum"],
        )

        maximum = config_defaults.scan_config["maximum"]
        if maximum is not None:
            default_scan_config["maximum"] = maximum

        return ClassifierOptionsConfig(
            batch_settings=config_defaults.batch_settings,
            accuracy=default_accuracy,
            scan_config=default_scan_config,
        )


MAX_RETRY_COUNT_FOR_GPU: int = 5
MAX_RETRY_COUNT: int = 10


class ClassifierManager(AbstractContextManager[None]):
    __device_manager: DeviceManager
    __model: Model
    __batch_settings: BatchSettings
    __segment_length: Timestamp
    __classifier: EncoderClassifier
    __retry_count: int
    __failed_too_often: bool

    def __init__(
        self: Self,
        device_manager: DeviceManager,
        model: Model,
        batch_settings: BatchSettings,
    ) -> None:
        self.__device_manager = device_manager
        self.__model = model
        self.__batch_settings = batch_settings
        self.__init_classification()
        self.__retry_count = 0
        self.__failed_too_often = False

    def __init_classifier(self: Self) -> None:
        run_opts: Optional[RunOpts] = self.__get_run_opts()

        self.__classifier = get_classifier_from_model(
            model=self.__model,
            run_opts=run_opts,
        )

    def __init_classification(self: Self) -> None:
        self.__check_audio_backends()
        self.__check_ffprobe()
        self.__init_segment_length()
        self.__init_classifier()

    def __check_audio_backends(self: Self) -> None:
        backends = audio_io.list_audio_backends()
        if len(backends) == 0:
            msg = "Couldn't find any audio backends for torchaudio"
            raise RuntimeError(msg)

        logger.debug("Found audio backends: %s", str(backends))

    def __check_ffprobe(self: Self) -> None:
        is_ffprobe_present = ffprobe_check()
        if not is_ffprobe_present:
            msg = "FFProbe not installed"
            raise RuntimeError(msg)

    @property
    def failed_too_often(self: Self) -> bool:
        return self.__failed_too_often

    def clear_cache(self: Self) -> None:
        gc.collect()

        self.__device_manager.clear_device_cache()

    def __get_run_opts(self: Self) -> Optional[RunOpts]:

        self.clear_cache()

        return get_model_run_opts(device_manager=self.__device_manager)

    def __get_segment_length(self: Self, batch_settings: BatchSettings) -> Timestamp:
        match batch_settings.batch_type:
            case "manual":
                return batch_settings.amount
            case "auto":
                available_memory: int = self.__device_manager.get_available_memory()

                memory_pattern = self.__model.memory_pattern
                if memory_pattern is None:

                    memory_pattern = get_memory_pattern_for_model(self.__model)

                    if memory_pattern is None:
                        msg = f"failed to derive the memory pattern for the model {self.__model.name}"
                        raise RuntimeError(msg)

                    msg = f"No memory_pattern for model {self.__model.name} defined, use the following derived:\n{memory_pattern.to_constructor_str()}"
                    raise RuntimeError(msg)

                keep_perc = to_advanced_percentage(batch_settings.keep_free, "")

                usable_memory = round(available_memory * (1.0 - keep_perc.value))

                seconds = memory_pattern.get_seconds_for_memory_amount(
                    memory_amount=usable_memory,
                )

                if seconds is None:
                    msg = "failed to calculate the seconds we can use with this available memory, this is likely an implementation error, or the memory_pattern was set incorrectly"
                    raise RuntimeError(msg)

                return Timestamp.from_seconds(seconds=seconds)
            case _:
                assert_never(batch_settings.batch_type)

    def __init_segment_length(self: Self) -> None:
        self.__segment_length = self.__get_segment_length(self.__batch_settings)

    @override
    def __enter__(self: Self) -> None:
        return None

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> bool:
        if exc_val is not None:

            if isinstance(exc_val, torch.cuda.OutOfMemoryError):
                if self.__retry_count >= MAX_RETRY_COUNT:
                    msg = f"Exceeded retry amount of {MAX_RETRY_COUNT} for classify"
                    logger.error(msg)
                    self.__failed_too_often = True
                    return True

                if (
                    self.__retry_count >= MAX_RETRY_COUNT_FOR_GPU
                    and self.__device_manager.type == AllocatorType.gpu
                ):
                    msg = "Switching the classifier to the cpu"
                    logger.debug(msg)

                    # reinitialize the classifier to use the cpu
                    del self.__classifier
                    self.__device_manager.force_cpu()
                    self.__init_classifier()

                self.__retry_count = self.__retry_count + 1
                self.__decrease_batch_size()
                return True

            logger.exception("Classify file")
            return False

        self.__retry_count = 0
        return False

    def retry_manager(self: Self) -> Self:
        return self

    def perform_classification(
        self: Self,
        audio_path: Path,
        savedir: Optional[Path],
    ) -> list[tuple[str, float]]:
        savedir_arg = str(savedir.absolute()) if savedir is not None else None

        # from: https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
        wavs: Any = self.__classifier.load_audio(
            str(audio_path.absolute()),
            savedir=savedir_arg,
        )

        embeddings = self.__classifier.encode_batch(wavs, wav_lens=None)
        out_prob = self.__classifier.mods.classifier(embeddings).squeeze(1)  # type: ignore[operator]

        return [
            (self.__classifier.hparams.label_encoder.decode_ndim(index), p)
            for index, p in enumerate(out_prob.exp().tolist()[0])
        ]

    def __del__(self: Self) -> None:
        self.clear_cache()

    def __decrease_batch_size(self: Self) -> None:
        # TODO
        pass

    @property
    def model(self: Self) -> Model:
        return self.__model

    @property
    def segment_length(self: Self) -> Timestamp:
        # TODO
        return Timestamp.from_seconds(30)


class PredictionFailReason(Enum):
    failed_too_often = "failed_too_often"
    pick_failed = "pick_failed"
    normal_threshold_failed = "normal_threshold_failed"
    final_threshold_failed = "final_threshold_failed"
    no_best_available = "no_best_available"


@dataclass
class PredictionFail:
    reason: PredictionFailReason
    best: Optional[PredictionBest]


class Classifier:
    __save_dir: Path
    __options: ClassifierOptions
    __manager: ClassifierManager

    def __init__(
        self: Self,
        device_manager: DeviceManager,
        model: Model,
        options: Optional[ClassifierOptionsConfig] | ClassifierOptions = None,
    ) -> None:
        self.__save_dir = Path(__file__).parent / "tmp"
        self.__options = self.__parse_options(options)
        self.__manager = ClassifierManager(
            device_manager=device_manager,
            model=model,
            batch_settings=self.__options.batch_settings,
        )

    def __parse_options(
        self: Self,
        options: Optional[ClassifierOptionsConfig] | ClassifierOptions,
    ) -> ClassifierOptions:
        total_options: ClassifierOptions = ClassifierOptions.default()

        if options is not None:
            total_options.batch_settings = (
                (
                    (
                        (
                            ManualBatchSettings(
                                batch_type="manual",
                                amount=(
                                    options.batch_settings
                                    if isinstance(options.batch_settings, Timestamp)
                                    else options.batch_settings.value
                                ),
                            )
                        )
                        if isinstance(
                            options.batch_settings,
                            (Timestamp, TimestampCompat),
                        )
                        else options.batch_settings
                    )
                )
                if options.batch_settings is not None
                else total_options.batch_settings
            )

            if options.accuracy is not None:
                normal_threshold = options.accuracy.get("normal_threshold", None)
                if normal_threshold is not None:
                    total_options.accuracy["normal_threshold"] = to_advanced_percentage(
                        normal_threshold,
                        "normal_threshold",
                    )

                final_threshold = options.accuracy.get("final_threshold", None)
                if final_threshold is not None:
                    total_options.accuracy["final_threshold"] = to_advanced_percentage(
                        final_threshold,
                        "final_threshold",
                    )

                use_picker_at_end = options.accuracy.get("use_picker_at_end", None)
                if use_picker_at_end is not None:
                    total_options.accuracy["use_picker_at_end"] = use_picker_at_end

            if options.scan_config is not None:
                minimum = options.scan_config.get("minimum", None)
                if minimum is not None:
                    total_options.scan_config["minimum"] = to_advanced_percentage(
                        minimum,
                        "minimum",
                    )

                maximum = options.scan_config.get("maximum", None)
                if maximum is not None:
                    total_options.scan_config["maximum"] = to_advanced_percentage(
                        maximum,
                        "maximum",
                    )

        return total_options

    def __classify(
        self: Self,
        wav_file: WAVFile,
        segment: Segment,
        manager: Optional[Manager] = None,
    ) -> Optional[Prediction]:
        result: WavFile__WavFileResult = wav_file.create_wav_file(
            WAVOptions(bitrate=self.__manager.model.bitrate, segment=segment),
            force_recreation=True,
            manager=manager,
        )

        if result.is_err():
            return None

        with result.get_ok() as wav_path:

            # TODO: say to the manager, that we try to use the gpu, so that if we switched to the cpu in the previous run, it maybe now has enough gpu memory to switch back

            while not self.__manager.failed_too_often:
                with self.__manager.retry_manager():
                    classifier_result = self.__manager.perform_classification(
                        wav_path,
                        self.__save_dir,
                    )

                    # NOTE:  ATTENTION - unsorted list
                    prob: list[tuple[float, Language]] = [
                        (
                            p,
                            Language.from_str_unsafe(
                                language_str,
                            ),
                        )
                        for language_str, p in classifier_result
                    ]

                    return Prediction(prob)

            return None

    def predict(
        self: Self,
        wav_file: WAVFile,
        path: Path,
        language_picker: LanguagePicker,
        manager: Optional[Manager],
    ) -> PredictionBest | PredictionFail:
        if self.__manager.failed_too_often:
            return PredictionFail(PredictionFailReason.failed_too_often, None)

        def get_segments(runtime: Timestamp) -> list[tuple[Segment, Timestamp]]:
            result: list[tuple[Segment, Timestamp]] = []
            current_timestamp: Timestamp = Timestamp.zero()

            # this is cached here, so that it doesn't change in the middle, as the manager can change it (in case of oom e.g.)
            segment_length: Timestamp = self.__manager.segment_length

            index: int = 0
            while current_timestamp <= runtime:
                end: Optional[Timestamp] = (
                    None
                    if current_timestamp + segment_length > runtime
                    else current_timestamp + segment_length
                )
                segment: Segment = Segment(
                    index=index,
                    start=current_timestamp,
                    end=end,
                )
                index = index + 1
                result.append((segment, end if end is not None else runtime))
                # ATTENTION: don't use +=, since that doesn't create a new object!
                current_timestamp = current_timestamp + segment_length

            return result

        # This guards for cases, where scanning is useless, e.g. when you want 20 % to be scanned, for a valid result, but scan until 10 %
        scan_nothing: bool = (
            self.__options.scan_config["maximum"] is not None
            and self.__options.scan_config["maximum"]
            < self.__options.scan_config["minimum"]
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

        amount_scanned: float = 0.0

        for segment, scanned_length in segments:
            local_prediction = self.__classify(wav_file, segment, manager)

            if local_prediction is None:
                return PredictionFail(PredictionFailReason.no_best_available, best=None)

            prediction += local_prediction

            if bar is not None:
                bar.update()

            amount_scanned = scanned_length / wav_file.runtime

            if amount_scanned < self.__options.scan_config["minimum"]:
                # scan the next segment
                continue

            if (
                self.__options.scan_config["maximum"] is not None
                and amount_scanned >= self.__options.scan_config["maximum"]
            ):
                # go to the end, to check if the result is good enough
                break

            best_local: Optional[PredictionBest] = prediction.get_best(
                MeanType.truncated,
            )

            if best_local is None or Language.is_default_value(best_local.language):
                # scan the next segment
                continue

            if best_local.accuracy >= self.__options.accuracy["normal_threshold"]:
                # go to evaluation, if the result is good enough
                break

            # otherwise fall trough and
            # scan the next segment

        # END OF FOR LOOP (I hate python and significant whitespace, {} would be better)

        if bar is not None:
            bar.close(clear=True)

        best: Optional[PredictionBest] = prediction.get_best(MeanType.truncated)

        if best is None:
            return PredictionFail(PredictionFailReason.no_best_available, best)

        maximum_to_scan: float = (
            1.0
            if self.__options.scan_config["maximum"] is None
            else self.__options.scan_config["maximum"].value
        )

        if amount_scanned >= maximum_to_scan:
            # use final treshold
            if best.accuracy >= self.__options.accuracy["final_threshold"]:
                return best

            if self.__options.accuracy["use_picker_at_end"]:
                picked_language = language_picker.pick_language(path, prediction)

                if picked_language is not None:
                    return PredictionBest(1.0, picked_language)

                return PredictionFail(PredictionFailReason.pick_failed, best)

            return PredictionFail(PredictionFailReason.final_threshold_failed, best)

        # use normal treshold
        if best.accuracy >= self.__options.accuracy["normal_threshold"]:
            return best

        msg = _("Couldn't get Language of '{path}': Best was {best}").format(
            path=relative_path_str(path),
            best=str(best),
        )
        logger.error(msg)

        return PredictionFail(PredictionFailReason.normal_threshold_failed, best)

    def __del__(self: Self) -> None:
        if self.__save_dir.exists():
            if self.__save_dir.is_file():
                self.__save_dir.rmdir()
            else:
                rmtree(str(self.__save_dir.absolute()), ignore_errors=True)
