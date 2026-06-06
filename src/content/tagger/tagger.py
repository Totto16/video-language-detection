from pathlib import Path

from content.tagger.mp4_tagger import VideoTaggerMP4
from content.tagger.mutagen_tagger import VideoTaggerMutagen
from content.tagger.video_tagger import (
    VideoTagger,
    VideoTagger__HandleResult,
    VideoTaggerMultiple,
)
from helper.translation import get_translator

_ = get_translator()


def get_tagger_for_mp4_file(file: Path) -> VideoTagger__HandleResult:

    try:

        tagger: list[VideoTagger] = []

        mutagen_handle = VideoTaggerMutagen.get_handle(file)

        # NOTE: we use mutagen instead of our custom tagger, as writing metadata is quite complicated
        # it lives inside "moov" -> "udta" -> "meta" -> "ilst" boxes
        # so to add new things, some complicated logic is needed,
        # wee need to create a udta box under moov, if it is not present, in both cases some complicated size logic need to be made, as all parent need to be updated, that means all sizes, sometimes some "free" padding is present, but just implementing using that is complicated, and mutagen already does that
        # additionally, when we resize the "moov" box, we need to also relocate global offset in the boxes "stco" and "co64", which is rather complicated, mutagen already does all that, so no need to complicate things

        if mutagen_handle.is_err():
            return VideoTagger__HandleResult.err(mutagen_handle.get_err())

        tagger.append(mutagen_handle.get_ok())

        mp4_handle = VideoTaggerMP4.get_handle(file)

        if mp4_handle.is_err():
            return VideoTagger__HandleResult.err(mp4_handle.get_err())

        tagger.append(mp4_handle.get_ok())

        result = VideoTaggerMultiple(file, tagger)

        return VideoTagger__HandleResult.ok(result)

    except RuntimeError as err:
        return VideoTagger__HandleResult.err(
            _("get tagger {err}").format(err=err),
        )


def get_tagger_for_file(file: Path) -> VideoTagger__HandleResult:

    ext = file.suffix

    match ext:
        case ".mp4":
            return get_tagger_for_mp4_file(file)
        case ".mkv":
            return VideoTagger__HandleResult.err("TODO")
        case ".avi":
            return VideoTagger__HandleResult.err("TODO")
        case _:
            return VideoTagger__HandleResult.err(
                _("Unsupported file extension {ext}").format(ext=ext),
            )
