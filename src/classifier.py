#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from os import makedirs, path, remove
from typing import Any, Optional, TypedDict, cast
from pathlib import Path
from ffprobe import FFProbe
from ffmpeg import FFmpeg, Progress
from speechbrain.pretrained import EncoderClassifier
import torch
import gc
import pynvml
import humanize
import shutil


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
    __tmp_file: Optional[Path]
    __file: Path
    __type: FileType
    __status: Status

    def __init__(self, file: Path) -> None:
        self.__tmp_file = None
        if not file.exists():
            raise FileNotFoundError(file)
        self.__file = file
        type, status = self.__get_info()
        self.__type = type
        self.__status = status

    def __get_info(self) -> tuple[FileType, Status]:
        try:
            metadata = FFProbe(str(self.__file.absolute()))
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
            raise RuntimeError(f"Couldn't get the Language from str {input}")

        return lan

    @staticmethod
    def Unknown() -> "Language":
        return Language("un", "Unknown")


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
                    wav_file.wav_path(), savedir=str(self.__save_dir.absolute())
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

        return (Language.Unknown(), 0.0)

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
        if self.__save_dir.exists():
            if self.__save_dir.is_file():
                remove(str(self.__save_dir.absolute()))
            else:
                shutil.rmtree(str(self.__save_dir.absolute()), ignore_errors=True)
