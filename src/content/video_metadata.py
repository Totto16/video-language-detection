from dataclasses import dataclass, field
from enum import StrEnum
from logging import Logger
from pathlib import Path
from typing import Annotated, Literal, Optional, Self

from apischema import alias, schema

from helper.apischema import OneOf, narrow_type
from helper.error import ErrorMode
from helper.ffprobe import FFprobeStream, StreamType, ffprobe, ffprobe_check
from helper.log import get_logger
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
    __dimensions: VideoDimension = field(metadata=alias("dimensions"))
    __streams: list[VideoStream] = field(metadata=alias("streams"))

    @staticmethod
    def __read_metadata(file: Path, error_mode: ErrorMode) -> Optional["VideoMetadata"]:
        metadata, err = ffprobe(file.absolute())

        if err is not None or metadata is None:
            error_mode.write_error(f'"{file}",')

            err_msg: str = f"Unable to get a valid stream from file '{file}':\n{err}"
            logger.error(err_msg)
            return None

        if not metadata.is_video():
            logger.error(_("File is not a video: {file}").format(file=str(file)))
            return None

        video_streams = metadata.video_streams()
        # only one video stream supported
        if len(video_streams) != 1:
            logger.error(
                _("Only one video stream supported, but got {video_streams}").format(
                    video_streams=len(video_streams),
                ),
            )
            return None

        video_dimensions = video_streams[0].video_dimensions()

        if video_dimensions is None:
            logger.error(
                _("Video file has no dimensions: {file}").format(file=str(file)),
            )
            return None

        # check if we have enough audio streams
        audio_streams = metadata.audio_streams()

        # only one audio stream supported atm
        if len(audio_streams) != 1:
            logger.error(
                _("Only one audio stream supported, but got {audio_streams}").format(
                    audio_streams=len(audio_streams),
                ),
            )
            return None

        file_duration: Optional[float] = metadata.file_info.duration_seconds()

        if file_duration is None:
            logger.error(_("No video duration was found"))
            return None

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

            return VideoMetadata(
                file_duration,
                dimensions,
                streams,
            )
        except RuntimeError as err:
            logger.error(err)  # noqa: TRY400
            return None

    @staticmethod
    def __check_ffprobe() -> None:
        is_ffprobe_present = ffprobe_check()
        if not is_ffprobe_present:
            msg = "FFProbe not installed"
            raise RuntimeError(msg)

    @staticmethod
    def from_file(file: Path, error_mode: ErrorMode) -> Optional["VideoMetadata"]:
        VideoMetadata.__check_ffprobe()

        return VideoMetadata.__read_metadata(file=file, error_mode=error_mode)
