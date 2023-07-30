import json
import pipes
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


class FFProbeRawResult(TypedDict):
    streams: list[FFprobeRawStream]


class FFProbeResult:
    __raw: FFProbeRawResult

    def __init__(self: Self, raw: FFProbeRawResult) -> None:
        self.__raw = raw

    @property
    def streams(self: Self) -> list[FFprobeStream]:
        return [FFprobeStream(stream) for stream in self.__raw["streams"]]

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
    commands = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        pipes.quote(str(file_path.absolute())),
    ]

    if not file_path.exists():
        return None, "File doesn't exist"

    result = subprocess.run(commands, capture_output=True)  # noqa: S603
    if result.returncode == 0:
        return FFProbeResult(json.loads(result.stdout)), None

    return None, f"FFProbe failed for {file_path}, output: {result.stderr!s}"
