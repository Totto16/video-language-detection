from pathlib import Path

from content.tagger.mp4_tagger import VideoTaggerMp4
from content.tagger.mutagen_tagger import VideoTaggerMutagen
from content.tagger.video_tagger import (
    VideoTagger,
    VideoTagger__HandleResult,
    VideoTaggerMultiple,
)
from helper.translation import get_translator

_ = get_translator()


def get_tagger_for_mp4_file(file: Path) -> VideoTagger__HandleResult:

    tagger: list[VideoTagger] = []

    mutagen_handle = VideoTaggerMutagen.get_handle(file)

    if mutagen_handle.is_err():
        return VideoTagger__HandleResult.err(mutagen_handle.get_err())

    tagger.append(mutagen_handle.get_ok())

    tagger.append(VideoTaggerMp4(file))

    result = VideoTaggerMultiple(file, tagger)

    return VideoTagger__HandleResult.ok(result)


def get_tagger_for_file(file: Path) -> VideoTagger__HandleResult:

    ext = file.suffix

    match ext:
        case ".mp4":
            return get_tagger_for_mp4_file(file)
        case ".mkv":
            return VideoTagger__HandleResult.err("TODO")
        case "avi":
            return VideoTagger__HandleResult.err("TODO")
        case _:
            return VideoTagger__HandleResult.err(
                _("Unsupported file extension {ext}").format(ext=ext),
            )
