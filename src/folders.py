#!/usr/bin/env python3

from enum import Enum
from os import makedirs, path, remove
from typing import Optional, TypedDict, cast
from lid import Model, KaldiRecognizer
import wave
import json
from ffprobe import FFProbe
from ffmpeg import FFmpeg, Progress


class FileType(Enum):
    wav = "wav"
    video = "video"
    audio = "audio"


class Status(Enum):
    ready = "ready"
    raw = "raw"


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

    def create_wav_file(self) -> bool:
        match (self.__status, self.__type):
            case (Status.ready, _):
                return False
            case (_, FileType.wav):
                return False
            case (Status.raw, _):
                self.__convert_to_wav()
                return True
            case _:  # stupid mypy
                raise RuntimeError("UNREACHABLE")

    def __convert_to_wav(self) -> None:
        temp_dir: str = path.join("/tmp", "video_lang_detect")
        if not path.exists(temp_dir):
            makedirs(temp_dir)
        self.__tmp_file = path.join(temp_dir, path.basename(self.__file) + ".wav")

        ffmpeg: FFmpeg = (
            FFmpeg()
            .option("y")
            .input(self.__file)
            .output(
                self.__tmp_file,
                acodec="pcm_s16le",
                #    ar=8000,
                ac=1,
            )
        )

        @ffmpeg.on("progress")  # type: ignore
        def progress_report(progress: Progress) -> None:
            print(progress)

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


MODEL: str = "lid-model"


class LanguageDict(TypedDict):
    language: str
    score: float


def main() -> None:
    files: list[str] = [
        "Citadel.S01E03.Infinite.Shadows.1080p.AMZN.WEB-DL.DDP5.1.H.264-NTb.mkv",
        "Black.Mirror.S06E01.GERMAN.DL.1080p.WEB.h264-SAUERKRAUT.mkv",
    ]

    ROOT_FOLDER: str = "/media/totto/Totto_1"

    for file in files:
        file_path = path.join(ROOT_FOLDER, file)
        wav_file = WAVFile(file_path)
        wav_file.create_wav_file()

        wf: wave.Wave_read = wave.open(wav_file.wav_path(), "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            print("Audio file must be WAV format mono PCM.")
            exit(1)

        model = Model(f"models/{MODEL}")
        rec = KaldiRecognizer(model, wf.getframerate())
        data: bytes = wf.readframes(-1)
        rec.AcceptWaveform(data)

        results = rec.Result()
        results_dict: list[LanguageDict] = cast(list[LanguageDict], json.loads(results))
        result = max(results_dict, key=lambda ev: ev["score"])

        language = result["language"]
        print(result)


if __name__ == "__main__":
    main()
