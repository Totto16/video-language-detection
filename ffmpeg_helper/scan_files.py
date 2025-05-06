import subprocess
from pathlib import Path
from typing import Optional

from scan_helper import get_files_to_scan


def scan_files(args: list[Path]) -> list[Optional[str]]:
    files: list[Path] = get_files_to_scan(args)
    return [scan_file(file) for file in files]


def scan_file(input_file: Path) -> Optional[str]:
    try:
        launch_args: list[str] = [
            "ffmpeg",
            "-v",
            "warning",
            "-i",
            str(input_file),
            "-f",
            "null",
            "-",
        ]

        subprocess.call(launch_args)  # noqa: S603

    except Exception as error:  # noqa: BLE001
        return str(error)
    else:
        return None
