from pathlib import Path
import subprocess
import shutil

LIST_TO_PROCESS: list[str] = [
    ## insert here
]


def main() -> None:
    for to_process in LIST_TO_PROCESS:
        input = Path(to_process)

        output: Path = input.parent / (input.stem + "_output" + input.suffix)

        launch_args: list[str] = [
            "ffmpeg",
            "-i",
            str(input),
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

        temp_result = Path(__file__).parent / input.name

        shutil.move(input, temp_result)
        shutil.move(output, input)


if __name__ == "__main__":
    main()
