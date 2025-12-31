import gc
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
    override,
)

import psutil
import torchaudio
from apischema import deserializer, schema, serializer
from enlighten import Manager
from ffmpeg.errors import FFmpegError
from ffmpeg.ffmpeg import FFmpeg
from ffmpeg.progress import Progress
from humanize import naturalsize


from content.language import Language
from content.language_picker import LanguagePicker
from content.prediction import MeanType, Prediction, PredictionBest
from helper.apischema import OneOf
from helper.ffprobe import ffprobe, ffprobe_check
from helper.gpu import GPU, GPUErrors
from helper.log import get_logger, setup_global_logger
from helper.result import Result
from helper.timestamp import ConfigTimeStamp, Timestamp, parse_int_safely
from helper.translation import get_translator

setup_global_logger()

from speechbrain.inference.classifiers import EncoderClassifier  # noqa: E402

WAV_FILE_BAR_FMT = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:2n}/{total:2n} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"

logger: Logger = get_logger()

_ = get_translator()

VOXLINGUA_SAMPLE_COUNT: int = 107


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


def has_enough_memory() -> tuple[bool, float, float]:
    memory = psutil.virtual_memory()
    percent = memory.free / memory.total

    swap_memory = psutil.swap_memory()
    swap_percent = swap_memory.free / swap_memory.total

    if percent <= 0.05 and swap_percent <= 0.10:
        return (False, percent, swap_percent)

    return (True, percent, swap_percent)


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
            msg = f"Option{option_name} has to be in percentage (0.0 - 1.0) but was: {value}"
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


@dataclass()
class ClassifierOptions:
    ##TODO: allow detection, detect gpu gddr ram and select best length based on kbit/s
    segment_length: Timestamp
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
            segment_length=Timestamp.from_seconds(30),
            accuracy=default_accuracy,
            scan_config=default_scan_config,
        )


@dataclass()
class ClassifierOptionsConfig:
    segment_length: Annotated[Optional[ConfigTimeStamp], OneOf]
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
            segment_length=config_defaults.segment_length,
            accuracy=default_accuracy,
            scan_config=default_scan_config,
        )


class RunOpts(TypedDict, total=False):
    device: Literal["cpu"] | str  ## can be "cuda:<index>"
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


class AllocatorType(Enum):
    cpu = "cpu"
    gpu = " gpu"


class Allocator:
    type: AllocatorType

    def __init__(self: Self, type_: AllocatorType) -> None:
        self.type = type_


class CPUAllocator(Allocator):
    def __init__(self) -> None:
        super().__init__(type_=AllocatorType.cpu)


class GPUAllocator(Allocator):
    gpu: GPU

    def __init__(self, gpu: GPU) -> None:
        super().__init__(type_=AllocatorType.gpu)
        self.gpu = gpu


MAX_RETRY_COUNT_FOR_GPU: int = 5
MAX_RETRY_COUNT: int = 10


class ClassifierManager(AbstractContextManager[None]):
    __allocator: CPUAllocator | GPUAllocator
    __classifier: EncoderClassifier
    __retry_count: int
    __failed_too_often: bool

    def __init__(self: Self) -> None:
        self.__init_type()
        self.__init_classification()
        self.__retry_count = 0
        self.__failed_too_often = False

    def __force_cpu(self: Self) -> None:
        self.__type = CPUAllocator()

    def __init_type(self: Self) -> None:

        gpu_result = GPU.get_best(use_integrated=False)

        if gpu_result.is_err():
            logger.warning("Got GPU error: %s", gpu_result.get_err())
            self.__type = CPUAllocator()
        else:
            self.__type = GPUAllocator(gpu_result.get_ok())

    def __init_classifier(self: Self) -> None:
        run_opts: Optional[RunOpts] = self.__get_run_opts()

        classifier: Optional[EncoderClassifier] = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="model",
            run_opts=run_opts,
        )
        if classifier is None:
            msg = "Couldn't initialize Classifier"
            raise RuntimeError(msg)

        classifier.hparams.label_encoder.expect_len(VOXLINGUA_SAMPLE_COUNT)

        self.__classifier = classifier

    def __init_classification(self: Self) -> None:
        self.__check_audio_backends()
        self.__check_ffprobe()
        self.__init_classifier()

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

    @property
    def failed_too_often(self: Self) -> bool:
        return self.__failed_too_often

    def clear_cache(self: Self) -> None:
        gc.collect()
        self.gpu.empty_cache()

    def __get_run_opts(self: Self) -> Optional[RunOpts]:

        if self.__type == AllocatorType.cpu:
            return {"device": "cpu"}

        ClassifierManager.clear_gpu_cache()

        return {
            "device": self.gpu.device_name_for_torch(),
            "data_parallel_count": -1,
            "data_parallel_backend": True,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }

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
            for GPUError in GPUErrors:
                if isinstance(exc_val, GPUError):
                    if self.__retry_count >= MAX_RETRY_COUNT:
                        msg = f"Exceeded retry amount of {MAX_RETRY_COUNT} for classify"
                        logger.error(msg)
                        self.__failed_too_often = True
                        return True

                    if (
                        self.__retry_count >= MAX_RETRY_COUNT_FOR_GPU
                        and self.__type == AllocatorType.gpu
                    ):
                        msg = "Switching the classifier to the cpu"
                        logger.debug(msg)

                        # reinitialize the classifier to use the cpu
                        del self.__classifier
                        self.__force_cpu()
                        self.__init_classifier()

                    self.__retry_count = self.__retry_count + 1
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
        if self.__type == AllocatorType.gpu:
            ClassifierManager.clear_gpu_cache()


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
        options: Optional[ClassifierOptionsConfig] | ClassifierOptions = None,
    ) -> None:
        self.__save_dir = Path(__file__).parent / "tmp"
        self.__options = self.__parse_options(options)
        self.__manager = ClassifierManager()

    def __parse_options(
        self: Self,
        options: Optional[ClassifierOptionsConfig] | ClassifierOptions,
    ) -> ClassifierOptions:
        total_options: ClassifierOptions = ClassifierOptions.default()

        if options is not None:
            total_options.segment_length = (
                (
                    options.segment_length
                    if isinstance(options.segment_length, Timestamp)
                    else options.segment_length.value
                )
                if options.segment_length is not None
                else total_options.segment_length
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
            WAVOptions(bitrate=16000, segment=segment),
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

            index: int = 0
            while current_timestamp <= runtime:
                end: Optional[Timestamp] = (
                    None
                    if current_timestamp + self.__options.segment_length > runtime
                    else current_timestamp + self.__options.segment_length
                )
                segment: Segment = Segment(
                    index=index,
                    start=current_timestamp,
                    end=end,
                )
                index = index + 1
                result.append((segment, end if end is not None else runtime))
                # ATTENTION: don't use +=, since that doesn't create a new object!
                current_timestamp = current_timestamp + self.__options.segment_length

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
