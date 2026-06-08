import json
import os
import platform
import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Self, TypedDict

from helper.result import Err, Ok, Result
from helper.timestamp import parse_int_safely


class FFprobeRawStream(TypedDict):
    pass


def parse_float_safely(inp: str) -> Optional[float]:
    try:
        return float(inp)
    except ValueError:
        return None


class StreamType(Enum):
    video = "video"
    audio = "audio"
    subtitle = "subtitle"
    attachment = "attachment"
    unknown = "unknown"


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

    def type(self: Self) -> StreamType:
        if self.is_video():
            return StreamType.video
        if self.is_audio():
            return StreamType.audio

        if self.is_subtitle():
            return StreamType.subtitle

        if self.is_attachment():
            return StreamType.attachment

        return StreamType.unknown

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
            return optional_float(val)

        return None

    def video_dimensions(self: Self) -> Optional[tuple[int, int]]:
        if self.is_video():
            width: Optional[Any] = self.__stream.get("width", None)
            height: Optional[Any] = self.__stream.get("height", None)

            width_val = optional_int(width)
            height_val = optional_int(height)

            if width_val is None or height_val is None:
                return None

            return (width_val, height_val)

        return None

    def __repr__(self: Self) -> str:
        return json.dumps(self.__stream)


def optional_int(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val

    if isinstance(val, str):
        return parse_int_safely(val)

    return None


def optional_float(val: Any) -> Optional[float]:
    if isinstance(val, int):
        return float(val)

    if isinstance(val, str):
        return parse_float_safely(val)

    return None


class FormatInfo:
    __raw: dict[str, Any]

    def __init__(self: Self, raw: dict[str, Any]) -> None:
        self.__raw = raw

    @property
    def raw(self: Self) -> dict[str, Any]:
        return self.__raw

    def duration_seconds(self: Self) -> Optional[float]:
        """
        Returns the runtime duration of the file as a floating point number of seconds.
        Returns None if the information is not present
        """
        val: Optional[Any] = self.__raw.get("duration", None)
        return optional_float(val)

    def size(self: Self) -> Optional[int]:
        val: Optional[Any] = self.__raw.get("size", None)
        return optional_int(val)

    def bit_rate(self: Self) -> Optional[float]:
        val: Optional[Any] = self.__raw.get("bit_rate", None)
        return optional_float(val)


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


def ffprobe_check() -> bool:
    # some things here were copied and modified from the original ffprobe-python repo:
    # https://github.com/gbstack/ffprobe-python/blob/master/ffprobe/ffprobe.py
    try:
        with Path(os.devnull).open(mode="w") as temp_file:
            subprocess.check_call(
                ["ffprobe", "-h"],  # noqa: S607
                stdout=temp_file,
                stderr=temp_file,
            )
    except FileNotFoundError:
        return False
    return True


def ffprobe(file_path: Path) -> Result[FFProbeResult, str]:
    # some things here were copied and modified from the original ffprobe-python repo:
    # https://github.com/gbstack/ffprobe-python/blob/master/ffprobe/ffprobe.py
    try:
        with Path(os.devnull).open(mode="w") as temp_file:
            subprocess.check_call(
                ["ffprobe", "-h"],  # noqa: S607
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
        return Err("File doesn't exist")

    result = subprocess.run(  # noqa: PLW1510, S602
        commands,
        capture_output=True,
        shell=True,
    )
    if result.returncode == 0:
        return Ok(FFProbeResult(json.loads(result.stdout)))

    return Err(f"FFProbe failed for {file_path}, output:\n{result.stderr.decode()}")
