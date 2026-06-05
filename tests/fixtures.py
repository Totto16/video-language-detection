import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest
import requests


@dataclass
class Finalizer[A]:
    data: A
    drop: Callable[[A], None]


class FinalizerFixture[A]:
    __data: Finalizer[A]

    def __init__(self: Self, data: Finalizer[A]) -> None:
        self.__data = data

    @property
    def data(self: Self) -> A:
        return self.__data.data

    def __del__(self: Self) -> None:
        print("DEL called with", self, self.__data)
        self.__data.drop(self.__data.data)


TempMp4Files = FinalizerFixture[list[Path]]


@pytest.fixture(scope="package")
def temp_mp4_files() -> TempMp4Files:
    # from: https://test-videos.co.uk/bigbuckbunny/mp4-h264
    # and https://file-examples.com/index.php/sample-video-files/
    video_urls = [
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/av1/360/Big_Buck_Bunny_360_10s_1MB.mp4",
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4",
        "https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm",
        "https://test-videos.co.uk/vids/bigbuckbunny/mkv/360/Big_Buck_Bunny_360_10s_1MB.mkv",
        #
        "https://file-examples.com/wp-content/storage/2017/04/file_example_MP4_480_1_5MG.mp4",
        "https://file-examples.com/wp-content/storage/2020/03/file_example_WEBM_480_900KB.webm",
        "https://file-examples.com/wp-content/storage/2018/04/file_example_AVI_480_750kB.avi",
        "https://file-examples.com/wp-content/storage/2018/04/file_example_MOV_480_700kB.mov",
        "https://file-examples.com/wp-content/storage/2018/04/file_example_WMV_480_1_2MB.wmv",
    ]
    results: list[Path] = []
    raise RuntimeError("TODO refactor the usage of this!")
    for url in video_urls:
        response = requests.get(url, timeout=10)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            results.append(Path(f.file.name))

    def delete_results(files: list[Path]) -> None:
        for f in files:
            f.unlink(missing_ok=True)

    return TempMp4Files(Finalizer[list[Path]](results, delete_results))


DummyFiles = FinalizerFixture[list[tuple[Path, bool]]]


@pytest.fixture(scope="package")
def dummy_files() -> DummyFiles:
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

    def delete_results(files: list[tuple[Path, bool]]) -> None:
        for f, _ in files:
            f.unlink(missing_ok=True)

    return DummyFiles(
        Finalizer[list[tuple[Path, bool]]](results, delete_results),
    )


def mark_as_used(value: Any) -> None:
    pass
