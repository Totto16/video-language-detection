#!/usr/bin/env python3


from pathlib import Path
from content import Content
from main import NameParser, parse_contents


class CustomNameParser(NameParser):
    pass


def main() -> None:
    ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")
    video_formats: list[str] = ["mp4", "mkv", "avi"]
    ignore_files: list[str] = [
        "metadata",
        "extrafanart",
        "theme-music",
        "Music",
        "Reportagen",
    ]
    parse_error_is_exception: bool = False

    contents: list[Content] = parse_contents(
        ROOT_FOLDER,
        {
            "ignore_files": ignore_files,
            "video_formats": video_formats,
            "parse_error_is_exception": parse_error_is_exception,
        },
        Path("data.json"),
        name_parser=NameParser(),
    )

    print([(content.languages(), content.description) for content in contents])


if __name__ == "__main__":
    main()
