import shutil
import subprocess
from pathlib import Path


def fix_chapters(to_process: list[str]) -> None:
    for file in to_process:
        try:
            input_file = Path(file)

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

            subprocess.call(launch_args)  # noqa: S603

            temp_result = (
                Path(__file__).parent
                / (input_file.parent.parent.name)
                / input_file.name
            )

            shutil.move(input_file, temp_result)
            shutil.move(output, input_file)
        except Exception:  # noqa: S110, BLE001
            pass
