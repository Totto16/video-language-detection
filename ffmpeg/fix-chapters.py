from pathlib import Path
import subprocess
import shutil

LIST_TO_PROCESS: list[str] = [
    ## insert here
]


def main() -> None:
    for to_process in LIST_TO_PROCESS:
        try:
            input_file = Path(to_process)

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

            subprocess.call(launch_args)

            temp_result = (
                Path(__file__).parent
                / (input_file.parent.parent.name)
                / input_file.name
            )

            shutil.move(input_file, temp_result)
            shutil.move(output, input_file)
        except Exception:  # noqa: S110, BLE001
            pass


if __name__ == "__main__":
    main()
