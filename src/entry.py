#!/usr/bin/env python3


import os
import re as regex
import sys
from pathlib import Path
from typing import Optional, Self

from classifier import Language
from content.base_class import Content  # noqa: TCH002
from content.general import NameParser, Summary
from helper.timestamp import parse_int_safely
from main import AllContent, generate_json_schema, parse_contents
from typing_extensions import override


class CustomNameParser(NameParser):
    __season_special_names: list[str]

    def __init__(self: Self, season_special_names: list[str]) -> None:
        super().__init__(Language("de", "German"))
        self.__season_special_names = season_special_names

    @override
    def parse_episode_name(self: Self, name: str) -> Optional[tuple[str, int, int]]:
        match = regex.search(r"Episode (\d{2}) - (.*) \[S(\d{2})E(\d{2})\]\.(.*)", name)
        if match is None:
            return None

        groups = match.groups()
        if len(groups) != 5:
            return None

        _episode_num, name, _season, _episode, _extension = groups
        season = parse_int_safely(_season)
        if season is None:
            return None

        episode = parse_int_safely(_episode)
        if episode is None:
            return None

        return (name, season, episode)

    @override
    def parse_season_name(self: Self, name: str) -> Optional[tuple[int]]:
        match = regex.search(r"Staffel (\d{2})", name)
        if match is None:
            if name in self.__season_special_names:
                return (0,)

            return None

        groups = match.groups()
        if len(groups) != 1:
            return None

        (_season,) = groups
        season = parse_int_safely(_season)
        if season is None:
            return None

        return (season,)

    @override
    def parse_series_name(self: Self, name: str) -> Optional[tuple[str, int]]:
        match = regex.search(r"(.*) \((\d{4})\)", name)
        if match is None:
            return None

        groups = match.groups()
        if len(groups) != 2:
            return None

        name, _year = groups
        year = parse_int_safely(_year)
        if year is None:
            return None

        return (name, year)


SPECIAL_NAMES: list[str] = ["Extras", "Specials", "Special"]
ROOT_FOLDER: Path = Path("/media/totto/Totto_4/Serien")


def main() -> None:
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
        Path("data/data.json"),
        name_parser=CustomNameParser(SPECIAL_NAMES),
        # TODO: make it possible to define to do a full scan or not (full= with language detection, not full = without)
    )

    summaries = [content.summary() for content in contents]
    final = Summary.combine_langauge_dicts([summary.languages for summary in summaries])

    print(final)


def clear_screen() -> None:
    # for windows
    if os.name == "nt":  # noqa: SIM108
        _ = os.system("cls")  # noqa: S605, S607

    # for mac and linux
    else:
        _ = os.system("clear")  # noqa: S605, S607


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            match sys.argv[1]:
                case "schema":
                    generate_json_schema(
                        Path("schema/content_list.json"),
                        list[AllContent],
                    )
                    sys.exit(0)
                case _:
                    # pass to say, i don#t care about other cases
                    pass
        main()
    except KeyboardInterrupt:
        print("Ctrl+C pressed")


# TODO: use logging instead of print() in some scenarios, let the command line arguments decide teh loglevel used
# TODO: use gettext to be able to translate strings (use default locale or fallback (en))
