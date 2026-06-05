import os
from math import isnan
from pathlib import Path

import pytest
from pytest_subtests import SubTests

from helper.ffprobe import ffprobe, parse_float_safely

from fixtures import ffprobe_dummy_files, mark_as_used, ffprobe_temp_mp4_files, TempVideoFiles, DummyFiles

mark_as_used(ffprobe_dummy_files)
mark_as_used(ffprobe_temp_mp4_files)


def test_float_parsing_correct(subtests: SubTests) -> None:
    raw_ints: list[str] = [
        "1",
        "112411414",
        "358013251367513513515",
        "-1212",
        "0",
        "+1212",
        "1.9",
        "1e10",
        "-1e-3",
        "Infinity",
        "NaN",
        "-12131.2121212e10",
    ]
    for float_num in raw_ints:
        with subtests.test():
            parsed_float = parse_float_safely(float_num)
            assert parsed_float == float(
                float_num,
            ) or (
                parsed_float is not None and isnan(parsed_float)
            ), f"{float_num} is parsable as float"


def test_float_parsing_wrong(subtests: SubTests) -> None:
    raw_ints: list[str] = ["-+112411414", "hks2", "test", "1t", "z0"]

    for int_num in raw_ints:
        with subtests.test():
            assert (
                parse_float_safely(int_num) is None
            ), f"{int_num} isn't parsable as float"


def test_raw_int_parse(subtests: SubTests) -> None:
    raw_ints: list[str] = [
        "iunfsaf",
        "sadsada",
        "dafdsa",
        "ds",
        "1saa",
        "ds1",
        "-1.4242jdwsa",
    ]

    for int_num in raw_ints:
        with subtests.test(), pytest.raises(
            ValueError,
            match=r"^could not convert string to float: '.*'$",
        ):
            float(int_num)


def test_ffprobe_with_intact_videos(
    subtests: SubTests,
    ffprobe_temp_mp4_files: TempVideoFiles,
) -> None:
    for video in ffprobe_temp_mp4_files.data:
        with subtests.test("video get's parsed correctly"):
            result, err = ffprobe(video)
            assert err is None, "No error occurred"
            assert result is not None, "a result was returned"

            assert (
                result.file_info.duration_seconds() is not None
            ), "duration is defined"

            assert len(result.streams) > 0, "at least one stream was detected"
            assert len(result.streams) == 1, "correct amount of streams"

            for stream in result.streams:
                assert stream.codec() == "h264", "codec is correct"
                assert stream.duration_seconds() == 10.0, "duration is correct"

                assert stream.is_attachment() is False, "has no attachments"
                assert stream.is_subtitle() is False, "has no subtitles"

                assert stream.is_video(), "stream is video"
                assert stream.is_audio() is False, "stream is not audio"

            assert len(result.video_streams()) == 1, "correct amount of video streams"
            assert result.is_video(), "result is video"

            assert len(result.audio_streams()) == 0, "correct amount of audio streams"
            assert result.is_audio() is False, "result is no audio"


def test_ffprobe_errors() -> None:
    assert ffprobe(Path("/zt/e.mp4")) == (
        None,
        "File doesn't exist",
    ), "file doesn't exist"

    path = os.environ["PATH"]
    os.environ["PATH"] = ""

    with pytest.raises(
        OSError,
        match=r"^ffprobe not found\.$",
    ):
        ffprobe(Path("/dev/null"))

    os.environ["PATH"] = path


def test_ffprobe_errors_with_files(
    subtests: SubTests,
    ffprobe_dummy_files: DummyFiles,
) -> None:
    with subtests.test("dummy wrong file fails "):
        for file, should_pass in ffprobe_dummy_files.data:
            result, _err = ffprobe(file)
            assert (result is not None) == should_pass, "pass status is correct"
            if should_pass:
                assert (
                    result is not None
                ), "pass status is correct"  # only for type checking!
                for stream in result.streams:
                    assert (
                        stream.duration_seconds() is None
                    ), "dummy files have no duration"
