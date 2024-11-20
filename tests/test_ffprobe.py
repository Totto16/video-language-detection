import os
import tempfile
from math import isnan
from pathlib import Path

import pytest
import requests
from pytest_subtests import SubTests

from helper.ffprobe import ffprobe, parse_float_safely


@pytest.fixture(scope="module")
def temp_mp4_files() -> list[Path]:
    ## from: https://test-videos.co.uk/bigbuckbunny/mp4-h264
    video_urls = [
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4",
    ]
    results: list[Path] = []
    for url in video_urls:
        response = requests.get(url, timeout=10)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            results.append(Path(f.file.name))

    return results


@pytest.fixture(scope="module")
def dummy_files() -> list[tuple[Path, bool]]:
    content_description = [
        (
            ".srt",
            """1
00:00:00,498 --> 00:00:02,827
- Here's what I love most
about food and diet.

2
00:00:02,827 --> 00:00:06,383
We all eat several times a day,
and we're totally in charge

3
00:00:06,383 --> 00:00:09,427
of what goes on our plate
and what stays off.""",
            True,
        ),
        (".txt", "", False),
    ]
    results: list[tuple[Path, bool]] = []
    for suffix, content, res in content_description:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(bytes(content, encoding="utf-8"))
            results.append((Path(f.file.name), res))

    return results


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
    temp_mp4_files: list[Path],
) -> None:
    for video in temp_mp4_files:
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

        video.unlink(missing_ok=True)


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
    dummy_files: list[tuple[Path, bool]],
) -> None:
    with subtests.test("dummy wrong file fails "):
        for file, should_pass in dummy_files:
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
