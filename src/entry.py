#!/usr/bin/env python3


import argparse
import atexit
import re as regex
import sys
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Self, cast, override

from classifier import Classifier, Language
from content.base_class import LanguageScanner, Scanner
from content.metadata.config import get_metadata_scanner_from_config
from content.summary import Summary
from helper.log import LogLevel, setup_custom_logger

if TYPE_CHECKING:
    from content.base_class import Content
    from content.metadata.scanner import MetadataScanner

from config import Config, ParsedConfig
from content.general import NameParser
from content.scanner import (
    get_scanner_from_config,
)
from gui.main import launch
from helper.timestamp import parse_int_safely
from helper.translation import get_translator
from main import generate_schemas, parse_contents


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


def main(config: ParsedConfig) -> None:
    classifier = Classifier(config.classifier)
    language_scanner = LanguageScanner(classifier=classifier)
    metadata_scanner: MetadataScanner = get_metadata_scanner_from_config(
        config.metadata,
    )
    scanner: Scanner = get_scanner_from_config(
        config.scanner,
        language_scanner,
        metadata_scanner,
    )

    contents: list[Content] = parse_contents(
        config.parser.root_folder,
        {
            "ignore_files": config.parser.ignore_files,
            "video_formats": config.parser.video_formats,
            "parse_error_is_exception": config.parser.exception_on_error,
        },
        config.general.target_file,
        name_parser=CustomNameParser(config.parser.special),
        scanner=scanner,
    )

    language_summary, metadata_summary = Summary.combine_summaries(
        content.summary() for content in contents
    )

    logger.info(language_summary)
    logger.info(metadata_summary)


SubCommand = Literal["run", "schema", "gui"]


class ParsedArgNamespace:
    level: LogLevel
    subcommand: SubCommand


class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]
    config: str


class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_folder: str


class GuiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["gui"]
    config: str


AllParsedNameSpaces = (
    RunCommandParsedArgNamespace
    | SchemaCommandParsedArgNamespace
    | GuiCommandParsedArgNamespace
)

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

    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
    )

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument(
        "-s",
        "--schema_folder",
        dest="schema_folder",
        default="schema/",
    )

    gui_parser = subparsers.add_parser("gui")
    gui_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
    )

    args = cast(AllParsedNameSpaces, parser.parse_args())
    logger: Logger = setup_custom_logger(args.level)

    try:
        match args.subcommand:
            case "schema":
                args_schema = cast(SchemaCommandParsedArgNamespace, args)
                generate_schemas(Path(args_schema.schema_folder))
                sys.exit(0)
            case "gui":
                args_gui = cast(GuiCommandParsedArgNamespace, args)
                config = Path(args_gui.config)
                launch(config)
                sys.exit(0)
            case "run":
                args_run = cast(
                    RunCommandParsedArgNamespace,
                    args,
                )
                parsed_config = Config.load(Path(args_run.config))
                if parsed_config is None:
                    sys.exit(1)

                main(parsed_config)

    except KeyboardInterrupt:

        def exit_handler() -> None:
            print()  # noqa: T201
            print(_("Ctrl + C pressed"))  # noqa: T201

        atexit.register(exit_handler)
