import subprocess
from pathlib import Path
from typing import Optional

from ffmpeg_helper.scan_helper import get_files_to_scan


def scan_files(args: list[Path]) -> list[str]:
    files: list[Path] = get_files_to_scan(args)
    results: list[str] = []
    for file in files:
        res = scan_file(file)
        if res is not None:
            results.append(res)

    return results


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

        ret_code = subprocess.call(launch_args)  # noqa: S603

        if ret_code != 0:
            return f"Process exited with status code: {ret_code}"

    except (RuntimeError, ValueError, TypeError) as error:
        return str(error)
    else:
        return None
