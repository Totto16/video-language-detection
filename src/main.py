#!/usr/bin/env python3

from os import listdir, path
from pathlib import Path
from typing import Callable, Optional
from classifier import Classifier, WAVFile

from content import ScannedFile, ScannedFileType


def process_folder_recursively(
    directory: Path,
    *,
    callback_function: Optional[
        Callable[[Path, ScannedFileType, list[str]], None]
    ] = None,
    ignore_folder_fn: Optional[Callable[[Path, str, list[str]], bool]] = None,
    parent_folders: list[str] = [],
) -> None:
    for file in listdir(directory):
        file_path: Path = Path(path.join(directory, file))
        if file_path.is_dir():
            if ignore_folder_fn is not None:
                should_ignore = ignore_folder_fn(file_path, file, parent_folders)
                if should_ignore:
                    continue

            if callback_function is not None:
                callback_function(file_path, ScannedFileType.folder, parent_folders)

            process_folder_recursively(
                file_path,
                callback_function=callback_function,
                parent_folders=[*parent_folders, file],
                ignore_folder_fn=ignore_folder_fn,
            )
        else:
            if callback_function is not None:
                callback_function(file_path, ScannedFileType.file, parent_folders)


def main() -> None:
    classifier = Classifier()

    ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")

    video_formats: list[str] = ["mp4", "mkv", "avi"]

    def process_file(
        file_path: Path, file_type: ScannedFileType, parent_folders: list[str]
    ) -> None:
        needs_scan: bool = file_type == ScannedFileType.folder
        if file_type == ScannedFileType.file:
            extension: str = file_path.suffix[1:]
            if extension not in video_formats:
                needs_scan = False

        if not needs_scan:
            return

        file: ScannedFile = ScannedFile.from_scan(file_path, file_type, parent_folders)
        print(file)

        if file_type == ScannedFileType.folder:
            return

        return

        wav_file = WAVFile(file_path)

        language, accuracy = classifier.predict(wav_file)
        print(language, accuracy)

    ignore_files: list[str] = ["metadata"]

    def ignore_folders(file_path: Path, file: str, parent_folders: list[str]) -> bool:
        if file.startswith("."):
            return True

        if file in ignore_files:
            return True

        return False

    process_folder_recursively(
        ROOT_FOLDER, callback_function=process_file, ignore_folder_fn=ignore_folders
    )


if __name__ == "__main__":
    main()
