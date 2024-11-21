#!/usr/bin/env python3


import argparse
import atexit
import re as regex
import sys
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Self, cast, override

from classifier import Classifier, Language
from helper.log import LogLevel, setup_custom_logger

if TYPE_CHECKING:
    from content.base_class import Content

from content.general import NameParser, Summary
from content.scanner import PartialLanguageScanner
from helper.timestamp import parse_int_safely
from helper.translation import get_translator
from main import AllContent, generate_json_schema, parse_contents


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
        scanner=PartialLanguageScanner(Classifier(), config_file=Path("./config.ini")),
    )

    summaries = [content.summary() for content in contents]
    final = Summary.combine_language_dicts([summary.languages for summary in summaries])

    logger.info(final)


SubCommand = Literal["run", "schema"]


class ParsedArgNamespace:
    level: LogLevel
    subcommand: SubCommand


class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]


class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_file: str


AllParsedNameSpaces = RunCommandParsedArgNamespace | SchemaCommandParsedArgNamespace

_ = get_translator()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="video-language-detection",
        description=_("Detect video languages"),
    )

    loglevel_choices: list[LogLevel] = [
        LogLevel.CRITICAL,
        LogLevel.ERROR,
        LogLevel.WARNING,
        LogLevel.INFO,
        LogLevel.DEBUG,
        LogLevel.NOTSET,
    ]
    loglevel_default: LogLevel = LogLevel.INFO
    parser.add_argument(
        "-l",
        "--level",
        choices=loglevel_choices,
        default=loglevel_default,
        dest="level",
        type=lambda s: LogLevel.from_str(s) or cast(LogLevel, s.lower()),
    )

    subparsers = parser.add_subparsers(required=False)
    parser.set_defaults(subcommand="run")

    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(subcommand="run")

    schema_parser = subparsers.add_parser("schema")
    schema_parser.set_defaults(subcommand="schema")
    schema_parser.add_argument(
        "-s",
        "--schema",
        dest="schema_file",
        default="schema/content_list.json",
    )

    args = cast(AllParsedNameSpaces, parser.parse_args())
    logger: Logger = setup_custom_logger(args.level)

    try:
        match args.subcommand:
            case "schema":
                generate_json_schema(
                    Path(args.schema_file),  # type: ignore[union-attr]
                    list[AllContent],
                )
                sys.exit(0)
            case "run":
                main()

    except KeyboardInterrupt:

        def exit_handler() -> None:
            print()  # noqa: T201
            print(_("Ctrl + C pressed"))  # noqa: T201

        atexit.register(exit_handler)
