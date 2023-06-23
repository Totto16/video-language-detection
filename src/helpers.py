from typing import Optional, Callable, TypeVar
from os import listdir
from pathlib import Path
from content import ScannedFileType


def parse_int_safely(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None


T = TypeVar("T")


def process_folder(
    directory: Path,
    process_fn: Callable[[Path, ScannedFileType, list[str]], T],
    *,
    ignore_fn: Callable[
        [Path, ScannedFileType, list[str]], bool
    ] = lambda x, y, z: False,
    parent_folders: list[str] = [],
) -> list[T]:
    results: list[T] = []
    for file in listdir(directory):
        file_path: Path = Path(directory) / file

        file_type: ScannedFileType = (
            ScannedFileType.folder if file_path.is_dir() else ScannedFileType.file
        )
        should_ignore: bool = ignore_fn(file_path, file_type, parent_folders)
        if should_ignore:
            continue

        result: T = process_fn(file_path, file_type, parent_folders)
        results.append(result)

    return results
