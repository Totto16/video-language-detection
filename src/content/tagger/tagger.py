from pathlib import Path

from content.tagger.mp4_tagger import VideoTaggerMP4
from content.tagger.mutagen_tagger import VideoTaggerMutagen
from content.tagger.video_tagger import (
    VideoTagger,
    VideoTaggerMultiple,
)
from helper.result import Err, Ok, Result
from helper.translation import get_translator

_ = get_translator()


def get_tagger_for_mp4_file_deprecated(file: Path) -> Result["VideoTagger", str]:

    try:

        tagger: list[VideoTagger] = []

        mutagen_handle = VideoTaggerMutagen.get_handle(file)

        # NOTE: we use mutagen instead of our custom tagger, as writing metadata is quite complicated
        # it lives inside "moov" -> "udta" -> "meta" -> "ilst" boxes
        # so to add new things, some complicated logic is needed,
        # wee need to create a udta box under moov, if it is not present, in both cases some complicated size logic need to be made, as all parent need to be updated, that means all sizes, sometimes some "free" padding is present, but just implementing using that is complicated, and mutagen already does that
        # additionally, when we resize the "moov" box, we need to also relocate global offset in the boxes "stco" and "co64", which is rather complicated, mutagen already does all that, so no need to complicate things

        if mutagen_handle.err():
            return Err(mutagen_handle.as_err())

        tagger.append(mutagen_handle.as_ok())

        mp4_handle = VideoTaggerMP4.get_handle(file)

        if mp4_handle.err():
            return Err(mp4_handle.as_err())

        tagger.append(mp4_handle.as_ok())

        result = VideoTaggerMultiple(file, tagger)

        return Ok(result)

    except RuntimeError as err:
        return Err(
            _("get tagger {err}").format(err=err),
        )


def get_tagger_for_mp4_file(file: Path) -> Result["VideoTagger", str]:

    try:

        mp4_handle = VideoTaggerMP4.get_handle(file)

        if mp4_handle.err():
            return Err(mp4_handle.as_err())

        return Ok(mp4_handle.as_ok())

    except RuntimeError as err:
        return Err(
            _("get tagger {err}").format(err=err),
        )


def get_tagger_for_file(file: Path) -> Result["VideoTagger", str]:

    ext = file.suffix

    match ext:
        case ".mp4":
            return get_tagger_for_mp4_file(file)
        case ".mkv":
            return Err("TODO")
        case ".avi":
            return Err("TODO")
        case _:
            return Err(
                _("Unsupported file extension {ext}").format(ext=ext),
            )
