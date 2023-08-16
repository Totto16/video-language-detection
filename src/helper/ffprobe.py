import json
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional, Self, TypedDict


class FFprobeRawStream(TypedDict):
    pass


def parse_float_safely(inp: str) -> Optional[float]:
    try:
        return float(inp)
    except ValueError:
        return None


# some things here were copied and modified from the original ffprobe-python repo:
# https://github.com/gbstack/ffprobe-python/blob/master/ffprobe/ffprobe.py
class FFprobeStream:
    __stream: FFprobeRawStream

    def __init__(self: Self, stream: FFprobeRawStream) -> None:
        self.__stream = stream

    def is_audio(self: Self) -> bool:
        """
        Is this stream labelled as an audio stream?
        """
        return self.__stream.get("codec_type", None) == "audio"

    def is_video(self: Self) -> bool:
        """
        Is the stream labelled as a video stream.
        """
        return self.__stream.get("codec_type", None) == "video"

    def is_subtitle(self: Self) -> bool:
        """
        Is the stream labelled as a subtitle stream.
        """
        return self.__stream.get("codec_type", None) == "subtitle"

    def is_attachment(self: Self) -> bool:
        """
        Is the stream labelled as a attachment stream.
        """
        return self.__stream.get("codec_type", None) == "attachment"

    def codec(self: Self) -> Optional[str]:
        """
        Returns a string representation of the stream codec.
        """
        val: Optional[Any] = self.__stream.get("codec_name", None)
        return val if isinstance(val, str) else None

    def duration_seconds(self: Self) -> Optional[float]:
        """
        Returns the runtime duration of the video stream as a floating point number of seconds.
        Returns None not a video or audio stream.
        """
        if self.is_video() or self.is_audio():
            val: Optional[Any] = self.__stream.get("duration", None)
            return parse_float_safely(val) if isinstance(val, str) else None

        return None

    def __repr__(self: Self) -> str:
        return json.dumps(self.__stream)


class FormatInfo:
    __raw: dict[str, Any]

    def __init__(self: Self, raw: dict[str, Any]) -> None:
        self.__raw = raw

    def duration_seconds(self: Self) -> Optional[float]:
        """
        Returns the runtime duration of the file as a floating point number of seconds.
        Returns None if the information is not present
        """
        val: Optional[Any] = self.__raw.get("duration", None)
        return parse_float_safely(val) if isinstance(val, str) else None


class FFProbeRawResult(TypedDict):
    streams: list[FFprobeRawStream]
    format: dict[str, Any]


class FFProbeResult:
    __raw: FFProbeRawResult

    def __init__(self: Self, raw: FFProbeRawResult) -> None:
        self.__raw = raw

    @property
    def streams(self: Self) -> list[FFprobeStream]:
        return [FFprobeStream(stream) for stream in self.__raw["streams"]]

    @property
    def file_info(self: Self) -> FormatInfo:
        return FormatInfo(self.__raw["format"])

    def video_streams(self: Self) -> list[FFprobeStream]:
        """
        Get all video streams
        """
        return [stream for stream in self.streams if stream.is_video()]

    def is_video(self: Self) -> bool:
        """
        Is the file a video alias has it at least one video stream
        """
        return len(self.video_streams()) != 0

    def audio_streams(self: Self) -> list[FFprobeStream]:
        """
        Get all audio streams
        """
        return [stream for stream in self.streams if stream.is_audio()]

    def is_audio(self: Self) -> bool:
        """
        Is the file a audio alias has it at least one audio stream
        """
        return len(self.audio_streams()) != 0


def ffprobe(file_path: Path) -> tuple[Optional[FFProbeResult], Optional[str]]:
    # some things here were copied and modified from the original ffprobe-python repo:
    # https://github.com/gbstack/ffprobe-python/blob/master/ffprobe/ffprobe.py
    try:
        with Path(os.devnull).open(mode="w") as temp_file:
            subprocess.check_call(
                ["ffprobe", "-h"],  # noqa: S607, S603
                stdout=temp_file,
                stderr=temp_file,
            )
    except FileNotFoundError as err:
        msg = "ffprobe not found."
        raise OSError(msg) from err

    commands: list[str] = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        shlex.quote(str(file_path.absolute())),
    ]

    if platform.system() != "Windows":
        commands = [" ".join(commands)]

    if not file_path.exists():
        return None, "File doesn't exist"

    result = subprocess.run(commands, capture_output=True, shell=True)  # noqa: S602
    if result.returncode == 0:
        return FFProbeResult(json.loads(result.stdout)), None

    return None, f"FFProbe failed for {file_path}, output:\n{result.stderr.decode()}"
