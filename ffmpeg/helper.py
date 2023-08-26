from os import listdir
from pathlib import Path


def get_files_to_scan(args: list[str]) -> list[Path]:
    result: list[Path] = []

    for arg in args:
        file = Path(arg)
        suffix: str = file.suffix[1:]
        match suffix:
            case "txt":
                with file.open(mode="r") as f:
                    file_content = f.read()
                    lines = file_content.split("\n")
                    result.extend(get_files_to_scan(lines))
            case _:
                if file.is_file():
                    result.append(file)
                elif file.is_dir():
                    files_in_dir = listdir(file)
                    result.extend(get_files_to_scan(files_in_dir))
                else:
                    pass

    return result
