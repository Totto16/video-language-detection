def mkv() -> None:
    from pymkv import (
        MKVFile,
        get_iso639_2,
        languages_match,
        language_equivalents,
        normalize_language,
    )

    get_iso639_2("English")  # "eng"
    get_iso639_2("fra")  # "fre"  (canonical /B)
    normalize_language("zh-Hans")  # "chi"  (BCP 47 subtag stripped)
    languages_match("zho", "zh")  # True
    language_equivalents("eng")  # frozenset({"eng", "en"})

    # MKVTrack setter is now lenient — any recognized form is accepted and
    # canonicalized to /B on store.
    mkv = MKVFile("path/to/file.mkv")
    track = mkv.tracks[1]
    track.language = "Chinese"  # stored as "chi"
    track.matches_language("zh")  # True (works against language_ietf too)
    track.effective_language  # "chi" — normalized /B
