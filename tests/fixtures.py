import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self

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
        self.__data.drop(self.__data.data)


TempVideoFiles = FinalizerFixture[list[Path]]


@pytest.fixture(scope="package")
def video_file_dict() -> dict[str, str]:
    # from: https://test-videos.co.uk/bigbuckbunny/mp4-h264
    # and https://file-examples.com/index.php/sample-video-files/
    video: dict[str, str] = {
        "Big_Buck_Bunny_360_10s_1MB.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/av1/360/Big_Buck_Bunny_360_10s_1MB.mp4",
        "Big_Buck_Bunny_1080_10s_1MB.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
        "Big_Buck_Bunny_1080_10s_30MB.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4",
        "Big_Buck_Bunny_360_10s_1MB.webm": "https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm",
        "Big_Buck_Bunny_360_10s_1MB.mkv": "https://test-videos.co.uk/vids/bigbuckbunny/mkv/360/Big_Buck_Bunny_360_10s_1MB.mkv",
        #
        "file_example_MP4_480_1_5MG.mp4": "https://file-examples.com/storage/fe63e83a686a22f89a101a3/2017/04/file_example_MP4_480_1_5MG.mp4",
        "file_example_WEBM_480_900KB.webm": "https://file-examples.com/wp-content/storage/2020/03/file_example_WEBM_480_900KB.webm",
        "file_example_AVI_480_750kB.avi": "https://file-examples.com/storage/fe2c4db53a6a2330da0d375/2018/04/file_example_AVI_480_750kB.avi",
        "file_example_MOV_480_700kB.mov": "https://file-examples.com/wp-content/storage/2018/04/file_example_MOV_480_700kB.mov",
        "file_example_WMV_480_1_2MB.wmv": "https://file-examples.com/wp-content/storage/2018/04/file_example_WMV_480_1_2MB.wmv",
    }

    return video


def at_video_dict(dct: dict[str, str], name: str) -> tuple[str, str]:
    return (name, dct[name])


class CachedFileManager:
    __cache_folder: Path

    def __init__(self: Self, cache_folder: Path) -> None:
        self.__cache_folder = cache_folder
        if not self.__cache_folder.exists():
            self.__cache_folder.mkdir(parents=True, exist_ok=True)

    def get(self: Self, name: str, url: str) -> bytes:
        cached_path = self.__cache_folder / name
        if cached_path.exists():
            return cached_path.read_bytes()

        result = requests.get(url, timeout=10)

        data = result.content
        with cached_path.open("wb") as f:
            f.write(data)

        return data


@pytest.fixture(scope="package")
def cached_file_manager() -> CachedFileManager:

    cache_folder = Path(__file__).parent / "files" / "cache"

    return CachedFileManager(cache_folder)


def temp_video_files(
    videos: list[tuple[str, str]],
    cached_manager: CachedFileManager,
) -> Finalizer[list[Path]]:
    results: list[Path] = []
    for name, url in videos:
        file_data = cached_manager.get(name, url)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(file_data)
            results.append(Path(f.file.name))

    def delete_results(files: list[Path]) -> None:
        for f in files:
            f.unlink(missing_ok=True)

    return Finalizer[list[Path]](results, delete_results)


@dataclass
class FFprobeData:
    codec: str
    duration: Optional[float]


TempFFProbeVideoFiles = FinalizerFixture[list[tuple[Path, FFprobeData]]]


@pytest.fixture(scope="package")
def ffprobe_temp_mp4_files(
    video_file_dict: dict[str, str],
    cached_file_manager: CachedFileManager,
) -> TempFFProbeVideoFiles:

    video_urls = [
        at_video_dict(video_file_dict, "Big_Buck_Bunny_360_10s_1MB.mp4"),
        at_video_dict(video_file_dict, "Big_Buck_Bunny_1080_10s_1MB.mp4"),
        at_video_dict(video_file_dict, "Big_Buck_Bunny_1080_10s_30MB.mp4"),
        at_video_dict(video_file_dict, "Big_Buck_Bunny_360_10s_1MB.webm"),
        at_video_dict(video_file_dict, "Big_Buck_Bunny_360_10s_1MB.mkv"),
    ]

    files = temp_video_files(video_urls, cached_file_manager)

    metadatas = [
        FFprobeData("av1", 10.0),
        FFprobeData("h264", 10.0),
        FFprobeData("h264", 10.0),
        FFprobeData("vp9", None),
        FFprobeData("h264", 10.0),
    ]

    def delete_results(_: list[tuple[Path, FFprobeData]]) -> None:
        files.drop(files.data)

    data: list[tuple[Path, FFprobeData]] = list(zip(files.data, metadatas, strict=True))

    finalizer: Finalizer[list[tuple[Path, FFprobeData]]] = Finalizer[
        list[tuple[Path, FFprobeData]]
    ](data, delete_results)

    return TempFFProbeVideoFiles(finalizer)


@pytest.fixture(scope="package")
def mp4_test_parse_files(
    video_file_dict: dict[str, str],
    cached_file_manager: CachedFileManager,
) -> TempVideoFiles:

    video_urls = [
        at_video_dict(video_file_dict, "file_example_MP4_480_1_5MG.mp4"),
    ]

    return TempVideoFiles(temp_video_files(video_urls, cached_file_manager))


@pytest.fixture(scope="package")
def avi_test_parse_files(
    video_file_dict: dict[str, str],
    cached_file_manager: CachedFileManager,
) -> TempVideoFiles:

    video_urls = [
        at_video_dict(video_file_dict, "file_example_AVI_480_750kB.avi"),
    ]

    return TempVideoFiles(temp_video_files(video_urls, cached_file_manager))


DummyFiles = FinalizerFixture[list[tuple[Path, bool]]]


@pytest.fixture(scope="package")
def ffprobe_dummy_files() -> DummyFiles:
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
