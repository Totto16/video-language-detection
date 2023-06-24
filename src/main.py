#!/usr/bin/env python3


import json
from pathlib import Path
from typing import Optional, TypedDict, cast
from typing_extensions import override

from classifier import Classifier
from tqdm.auto import tqdm
from content import (
    Callback,
    Content,
    ContentCharacteristic,
    Decoder,
    Encoder,
    ScannedFileType,
    process_folder,
)


class ProgressBar:
    __name: str
    __bar: tqdm

    def __init__(self, amount: int, name: str) -> None:
        self.__name = name
        self.__bar = tqdm(desc=name)
        self.reset(amount, name)

    def reset(self, amount: int, name: str) -> None:
        self.__bar.desc = name
        self.__bar.reset(amount)
        self.__name = name

    def advance(self) -> None:
        self.__bar.update()

    def finish(self, name: str) -> None:
        if name != self.__name:
            raise RuntimeError("start and finish names don't match up!")

        self.__bar.refresh()

    def close(self) -> None:
        self.__bar.close()


class ContentOptions(TypedDict):
    ignore_files: list[str]
    video_formats: list[str]
    parse_error_is_exception: bool


class ContentCallback(Callback[Content, ContentCharacteristic]):
    __options: ContentOptions
    __classifier: Classifier
    __progress_bars: list[ProgressBar]
    __bar_index: int
    __with_progress_bar: bool

    def __init__(
        self,
        options: ContentOptions,
        classifier: Classifier,
        *,
        with_progress_bar: bool = True,
    ) -> None:
        super().__init__()

        self.__options = options
        self.__classifier = classifier
        self.__progress_bars = []
        self.__bar_index = 0
        self.__with_progress_bar = with_progress_bar

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
        amount: int,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        if not self.__with_progress_bar:
            print(f"start: {name} - {amount} - {characteristic}")
            return

        # don't make a bar for episodes!

        progress_bar = ProgressBar(amount, name)
        self.__bar_index += 1
        if self.__bar_index >= len(self.__progress_bars):
            self.__progress_bars.append(progress_bar)
        else:
            self.__progress_bars[self.__bar_index - 1].reset(amount, name)

    @override
    def progress(
        self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        if not self.__with_progress_bar:
            print(f"progress: {name} - {characteristic}")
            return

        if self.__bar_index == 0:
            raise RuntimeError("No Progressbar, on progress callback")

        self.__progress_bars[self.__bar_index - 1].advance()

    @override
    def finish(
        self,
        name: str,
        parent_folders: list[str],
        characteristic: ContentCharacteristic,
    ) -> None:
        if not self.__with_progress_bar:
            print(f"finish: {name} - {characteristic}")
            return

        # remove top only if the names match up!!! (not if it's a single episode!)

        if self.__bar_index == 0:
            raise RuntimeError("No Progressbar, on progress finish")

        self.__bar_index -= 1
        self.__progress_bars[self.__bar_index].finish(name)

    def __del__(self) -> None:
        for progress_bar in self.__progress_bars:
            progress_bar.close()


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
        with_progress_bar=False,
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
