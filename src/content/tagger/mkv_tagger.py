from pymkv import (
    MKVFile,
    get_iso639_2,
)

from content.language import Language


def set_mkv_language(language: Language) -> None:

    lang = get_iso639_2(language.short)

    # MKVTrack setter is now lenient — any recognized form is accepted and
    # canonicalized to /B on store.
    mkv = MKVFile("path/to/file.mkv")

    for track in mkv.tracks:
        track.language = lang
