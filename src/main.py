#!/usr/bin/env python3


import json
from pathlib import Path
from typing import Any, Optional, TypedDict, cast
from typing_extensions import override
from enlighten import Justify, Manager, get_manager
from classifier import Classifier
from content import (
    Callback,
    Content,
    ContentCharacteristic,
    ContentType,
    Decoder,
    Encoder,
    ScannedFileType,
    process_folder,
)


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    parse_error_is_exception: bool


class ContentCallback(Callback[Content, ContentCharacteristic, Manager]):
    __options: ContentOptions
    __classifier: Classifier
    __progress_bars: dict[str, Any]
    __manager: Manager
    __status_bar: Any

    def __init__(
        self,
        options: ContentOptions,
        classifier: Classifier,
    ) -> None:
        super().__init__()

        self.__options = options
        self.__classifier = classifier
        self.__progress_bars = dict()
        manager = get_manager()
        if not isinstance(manager, Manager):
            raise RuntimeError("UNREACHABLE (not runnable in notebooks)")

        self.__manager = manager
        self.__status_bar = self.__manager.status_bar(
            status_format="Video Language Detector{fill}Stage: {stage}{fill}{elapsed}",
            color="bold_underline_bright_white_on_blue",
            justify=Justify.CENTER,
            stage="Scanning",
            autorefresh=True,
            min_delta=0.5,
        )

    def get_saved(self) -> Manager:
        return self.__manager

    @override
    def ignore(
        self, file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> bool:
        name: str = file_path.name
        if file_type == ScannedFileType.folder:
            if name.startswith("."):
                return True

            if name in self.__options["ignore_files"]:
                return True
        else:
            extension: str = file_path.suffix[1:]
            if extension not in self.__options["video_formats"]:
                return True

        return False

    @override
    def process(
        self, file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> Optional[Content]:
        content: Optional[Content] = Content.from_scan(
            file_path, file_type, parent_folders
        )
        if content is None:
            if self.__options["parse_error_is_exception"]:
                raise RuntimeError(
                    f"Parse Error: Couldn't parse content from '{file_path}'"
                )

            return None

        content.scan(
            callback=self,
            parent_folders=parent_folders,
            classifier=self.__classifier,
        )

        return content

    @override
    def start(
        self,
        amount: tuple[int, int, int],
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        content_type, file_type = characteristic

        # don't make a bar for episodes!
        # if file_type == ScannedFileType.file:
        # return

        value: tuple[str, str] = ("purple", "folders")

        match content_type:
            case ContentType.collection:
                value = ("blue", "series")
            case ContentType.series:
                value = ("cyan", "seasons")
            case ContentType.season:
                value = ("green", "episodes")
            case ContentType.episode:
                value = ("yellow", "tasks")
            case _:
                pass

        color, unit = value

        total, processing, ignored = amount

        self.__progress_bars[name] = self.__manager.counter(
            total=processing,
            desc=name,
            unit=unit,
            leave=False,
            color=color,
        )
        self.__progress_bars[name].update(0, force=True)

    @override
    def progress(
        self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
        *,
        amount: int = 1,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            raise RuntimeError("No Progressbar, on progress callback")

        content_type, file_type = characteristic

        self.__progress_bars[name].update(amount)

    @override
    def finish(
        self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        if self.__progress_bars.get(name) is None:
            raise RuntimeError("No Progressbar, on progress finish")

        self.__progress_bars[name].close(clear=True)
        del self.__progress_bars[name]

    def __del__(self) -> None:
        self.__status_bar.update(stage="finished")
        self.__manager.stop()


def main() -> None:
    classifier = Classifier()

    ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")
    video_formats: list[str] = ["mp4", "mkv", "avi"]
    ignore_files: list[str] = [
        "metadata",
        "extrafanart",
        "theme-music",
        "Music",
        "Reportagen",
    ]
    parse_error_is_exception: bool = False

    callback = ContentCallback(
        {
            "ignore_files": ignore_files,
            "video_formats": video_formats,
            "parse_error_is_exception": parse_error_is_exception,
        },
        classifier,
    )

    contents: list[Content] = process_folder(
        ROOT_FOLDER,
        callback=callback,
    )

    json_content: str = json.dumps(contents, cls=Encoder)

    # with open("data.json", "w") as file:
    #     file.write(json_content)

    json_loaded: list[Content] = cast(
        list[Content], json.loads(json_content, cls=Decoder)
    )


if __name__ == "__main__":
    main()
