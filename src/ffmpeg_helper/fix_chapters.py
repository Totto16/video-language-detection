import shutil
import subprocess
from pathlib import Path
from typing import Optional


def fix_chapters(files: list[Path]) -> list[str]:
    results: list[str] = []
    for file in files:
        res = fix_chapter(file)
        if res is not None:
            results.append(res)

    return results


def fix_chapter(input_file: Path) -> Optional[str]:
    temp_folder = Path(__file__).parent.parent.parent / "temp"

    if not temp_folder.exists():
        temp_folder.mkdir(parents=True, exist_ok=True)

    try:
        output: Path = input_file.parent / (
            input_file.stem + "_output" + input_file.suffix
        )

        launch_args: list[str] = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-vcodec",
            "copy",
            "-acodec",
            "copy",
            "-map_chapters",
            "-1",
            "-y",
            str(output),
        ]

        ret_code = subprocess.call(launch_args)  # noqa: S603

        if ret_code != 0:
            output.unlink(missing_ok=True)
            return f"Process exited with status code: {ret_code}"

        temp_result = temp_folder / input_file.name

        shutil.move(input_file, temp_result)
        shutil.move(output, input_file)
    except (RuntimeError, ValueError, TypeError) as error:
        return str(error)
    else:
        return None
