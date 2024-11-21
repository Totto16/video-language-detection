from os import listdir
from pathlib import Path


def get_files_to_scan(args: list[str]) -> list[Path]:
    result: list[Path] = []

    for arg in args:
        local_file = Path(arg)
        suffix: str = local_file.suffix[1:]
        match suffix:
            case "txt":
                with local_file.open(mode="r") as f:
                    file_content: str = f.read()
                    lines: list[str] = file_content.split("\n")
                    result.extend(get_files_to_scan(lines))
            case _:
                if local_file.is_file():
                    result.append(local_file)
                elif local_file.is_dir():
                    files_in_dir: list[str] = listdir(local_file)
                    result.extend(get_files_to_scan(files_in_dir))
                else:
                    pass

    return result
