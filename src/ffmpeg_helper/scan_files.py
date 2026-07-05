import subprocess
from pathlib import Path

from ffmpeg_helper.scan_helper import get_files_to_scan
from helper.result import Err, Ok, Result


def scan_files(args: list[Path]) -> list[str]:
    files: list[Path] = get_files_to_scan(args)
    results: list[str] = []
    for file in files:
        res = scan_file(file)
        if res.err():
            results.append(res.as_err())

    return results


def scan_file(input_file: Path) -> Result[None, str]:
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
            return Err(f"Process exited with status code: {ret_code}")

    except (RuntimeError, ValueError, TypeError) as error:
        return Err(str(error))
    else:
        return Ok(None)
