from dataclasses import dataclass, field
from enum import StrEnum
from logging import Logger
from pathlib import Path
from typing import Annotated, Literal, Optional, Self

from apischema import alias, schema
from apischema.metadata import none_as_undefined

from helper.apischema import OneOf, narrow_type
from helper.error import ErrorMode
from helper.ffprobe import FFprobeStream, StreamType, ffprobe, ffprobe_check
from helper.log import get_logger
from helper.result import Err, Ok, Result
from helper.translation import get_translator

logger: Logger = get_logger()
_ = get_translator()


class VideoStreamType(StrEnum):
    video = "video"
    audio = "audio"
    subtitle = "subtitle"
    attachment = "attachment"
    unknown = "unknown"

    def __str__(self: Self) -> str:
        return f"<VideoStreamType: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class VideoStreamInterface:
    __type: VideoStreamType = field(metadata=alias("type"))

    def __init__(self: Self, type_: VideoStreamType) -> None:
        super().__init__()
        self.__type = type_

    @property
    def type(self: Self) -> VideoStreamType:
        return self.__type


@schema(extra=narrow_type(("type", Literal[VideoStreamType.video])))
@dataclass(slots=True, repr=True)
class VideoStreamVideo(VideoStreamInterface):
    duration: float


@schema(extra=narrow_type(("type", Literal[VideoStreamType.audio])))
@dataclass(slots=True, repr=True)
class VideoStreamAudio(VideoStreamInterface):
    duration: float


@schema(extra=narrow_type(("type", Literal[VideoStreamType.subtitle])))
@dataclass(slots=True, repr=True)
class VideoStreamSubtitle(VideoStreamInterface):
    pass


@schema(extra=narrow_type(("type", Literal[VideoStreamType.attachment])))
@dataclass(slots=True, repr=True)
class VideoStreamAttachment(VideoStreamInterface):
    pass


@schema(extra=narrow_type(("type", Literal[VideoStreamType.unknown])))
@dataclass(slots=True, repr=True)
class VideoStreamUnknown(VideoStreamInterface):
    pass


VideoStream = Annotated[
    VideoStreamVideo
    | VideoStreamAudio
    | VideoStreamSubtitle
    | VideoStreamAttachment
    | VideoStreamUnknown,
    OneOf,
]


@dataclass(slots=True, repr=True)
class VideoDimension:
    width: int
    height: int


@dataclass(slots=True, repr=True)
class VideoMetadata:
    __duration: float = field(metadata=alias("duration"))
    __streams: list[VideoStream] = field(metadata=alias("streams"))
    __dimensions: VideoDimension = field(metadata=alias("dimensions"))
    __size: int = field(
        metadata=alias("size"),
    )
    __bit_rate: Optional[float] = field(
        default=None,
        metadata=alias("bit_rate") | none_as_undefined,
    )

    @property
    def duration(self: Self) -> float:
        return self.__duration

    @property
    def streams(self: Self) -> list[VideoStream]:
        return self.__streams

    @property
    def dimensions(self: Self) -> VideoDimension:
        return self.__dimensions

    @property
    def size(self: Self) -> int:
        return self.__size

    @property
    def bit_rate(self: Self) -> Optional[float]:
        return self.__bit_rate

    @staticmethod
    def __read_metadata(
        file: Path,
        error_mode: ErrorMode,
    ) -> Result["VideoMetadata", str]:
        metadata_res = ffprobe(file.absolute())

        if metadata_res.err():
            error_mode.write_error(f'"{file}",')

            err_msg: str = _(
                "Unable to get a valid stream from file:\n{err}"  # noqa: COM812
            ).format(err=metadata_res.as_err())
            return Err(err_msg)

        metadata = metadata_res.as_ok()

        if not metadata.is_video():
            return Err(
                _("File is not a video"),
            )

        video_streams = metadata.video_streams()
        # only one video stream supported
        if len(video_streams) != 1:
            return Err(
                _("Only one video stream supported, but got {video_streams}").format(
                    video_streams=len(video_streams),
                ),
            )

        video_dimensions = video_streams[0].video_dimensions()

        if video_dimensions is None:
            return Err(
                _("Video file has no dimensions"),
            )

        # check if we have enough audio streams
        audio_streams = metadata.audio_streams()

        # only one audio stream supported atm
        if len(audio_streams) != 1:
            return Err(
                _("Only one audio stream supported, but got {audio_streams}").format(
                    audio_streams=len(audio_streams),
                ),
            )

        file_duration: Optional[float] = metadata.file_info.duration_seconds()

        if file_duration is None:
            return Err(_("No video duration was found"))

        def map_stream(stream: FFprobeStream) -> VideoStream:
            match stream.type():
                case StreamType.video:
                    duration = stream.duration_seconds() or file_duration
                    return VideoStreamVideo(VideoStreamType.video, duration)
                case StreamType.audio:
                    duration = stream.duration_seconds() or file_duration
                    return VideoStreamAudio(VideoStreamType.audio, duration)
                case StreamType.subtitle:
                    return VideoStreamSubtitle(VideoStreamType.subtitle)
                case StreamType.attachment:
                    return VideoStreamAttachment(VideoStreamType.attachment)
                case StreamType.unknown:
                    return VideoStreamUnknown(VideoStreamType.unknown)

        try:
            width, height = video_dimensions
            dimensions = VideoDimension(width=width, height=height)

            streams: list[VideoStream] = [
                map_stream(stream) for stream in metadata.streams
            ]

            size: int = metadata.file_info.size() or file.stat().st_size

            bit_rate: Optional[float] = metadata.file_info.bit_rate()

            return Ok(
                VideoMetadata(
                    file_duration,
                    streams,
                    dimensions,
                    size,
                    bit_rate,
                ),
            )
        except RuntimeError as err:
            return Err(str(err))

    @staticmethod
    def __check_ffprobe() -> None:
        is_ffprobe_present = ffprobe_check()
        if not is_ffprobe_present:
            msg = "FFProbe not installed"
            raise RuntimeError(msg)

    @staticmethod
    def from_file(file: Path, error_mode: ErrorMode) -> Result["VideoMetadata", str]:
        VideoMetadata.__check_ffprobe()

        return VideoMetadata.__read_metadata(file=file, error_mode=error_mode)

    def __str__(self: Self) -> str:
        return f"<VideoMetadata duration: {self.__duration} streams: {self.__streams} dimensions: {self.__dimensions} size: {self.__size}>"

    def __repr__(self: Self) -> str:
        return str(self)
