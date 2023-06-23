#!/usr/bin/env python3


import json
from pathlib import Path
from typing import Optional, cast
from classifier import Classifier

from content import Content, Decoder, Encoder, ScannedFileType, process_folder


classifier = Classifier()


def main() -> None:
    ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")

    video_formats: list[str] = ["mp4", "mkv", "avi"]

    ignore_files: list[str] = ["metadata"]

    def ignore_fn(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> bool:
        name: str = file_path.name
        if file_type == ScannedFileType.folder:
            if name.startswith("."):
                return True

            if name in ignore_files:
                return True
        else:
            extension: str = file_path.suffix[1:]
            if extension not in video_formats:
                return True

        return False

    def process_file(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> Content:
        content: Optional[Content] = Content.from_scan(
            file_path, file_type, parent_folders
        )
        if content is None:
            raise RuntimeError(f"Parse Error: Couldn't parse content from {file_path}")

        return content

    contents: list[Content] = process_folder(
        ROOT_FOLDER, process_fn=process_file, ignore_fn=ignore_fn
    )

    json_content: str = json.dumps(contents, cls=Encoder)

    with open("test.json", "w") as file:
        file.write(json_content)

    json_loaded: list[Content] = cast(
        list[Content], json.loads(json_content, cls=Decoder)
    )
    print(json_loaded)


if __name__ == "__main__":
    main()