#!/usr/bin/env python3


import argparse
import atexit
import json
import re as regex
import sys
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, assert_never, cast, override

from apischema import serialize
from prompt_toolkit.key_binding import KeyBindings

from classifier import Classifier
from content.base_class import LanguageScanner, Scanner
from content.language import Language
from content.language_picker import LanguagePicker, get_picker_from_config
from content.metadata.config import get_metadata_scanner_from_config
from content.summary import Summary
from helper.log import LogLevel, setup_custom_logger

if TYPE_CHECKING:
    from content.base_class import Content
    from content.metadata.scanner import MetadataScanner

from config import AdvancedConfig, FinalConfig, KeyBoardConfig
from content.general import NameParser
from content.scanner import (
    ConfigScanner,
    get_scanner_from_config,
)
from gui.main import launch_gui
from helper.timestamp import parse_int_safely
from helper.translation import get_translator
from main import generate_schemas, parse_contents

PROGRAM_VERSION: str = "2.5.3"


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


def get_keybindings(
    logger: Logger,
    kb_config: KeyBoardConfig,
    scanner: Scanner,
) -> KeyBindings:
    kb = KeyBindings()

    @kb.add(kb_config.abort.value)
    def _abort_callback(_event: Any) -> None:
        if isinstance(scanner, ConfigScanner) and scanner.allow_abort:
            scanner.abort()
            logger.info(_("Aborted scan"))

    return kb


def launch_tui(logger: Logger, config: FinalConfig) -> None:
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

    language_picker: LanguagePicker = get_picker_from_config(config.picker)

    # TODO: this doesn't work atm
    # this is also unnecessary complicated for a tui app, do this in the gui instead
    _kb: KeyBindings = get_keybindings(logger, config.keybindings, scanner)

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
        language_picker=language_picker,
    )

    language_summary, metadata_summary = Summary.combine_summaries(
        content.summary() for content in contents
    )

    scan_summary = language_scanner.summary_manager.get_detailed_summary()

    logger.info(language_summary)
    logger.info(metadata_summary)
    logger.info(scan_summary)


type SubCommand = Literal["run", "schema", "gui", "config_check"]


class ParsedArgNamespace:
    level: LogLevel
    subcommand: SubCommand


class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]
    config: str
    config_to_use: Optional[str]


class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_folder: str


class GuiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["gui"]
    config: str


class ConfigCheckCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["config_check"]
    config: str
    config_to_use: Optional[str]


type AllParsedNameSpaces = (
    RunCommandParsedArgNamespace
    | SchemaCommandParsedArgNamespace
    | GuiCommandParsedArgNamespace
    | ConfigCheckCommandParsedArgNamespace
)

_ = get_translator()


def parse_args() -> AllParsedNameSpaces:
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
        help=_("The loglevel to use"),
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {PROGRAM_VERSION}",
    )

    subparsers = parser.add_subparsers(
        required=True,
        dest="subcommand",
    )

    run_parser = subparsers.add_parser(
        "run",
        description=_("Run the whole program in the terminal"),
    )
    run_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
        help=_("The config to use"),
    )
    run_parser.add_argument(
        "-t",
        "--template",
        dest="config_to_use",
        default=None,
        help=_(
            "The config template to use, if the config specifies, to use the cli one",
        ),
    )

    schema_parser = subparsers.add_parser(
        "schema",
        description=_("Create schemas for config and the resulting data"),
    )
    schema_parser.add_argument(
        "-s",
        "--schema_folder",
        dest="schema_folder",
        default="schema/",
        help=_("The folder where to put the schemas"),
    )

    gui_parser = subparsers.add_parser(
        "gui",
        description=_("Run the whole program as GUI"),
    )
    gui_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
        help=_("The config to use"),
    )

    config_check_parser = subparsers.add_parser(
        "config_check",
        description=_("Check the config for validity"),
    )
    config_check_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
        help=_("The config to check"),
    )
    config_check_parser.add_argument(
        "-t",
        "--template",
        dest="config_to_use",
        default=None,
        help=_(
            "The config template to use, if the config specifies, to use the cli one",
        ),
    )

    return cast(AllParsedNameSpaces, parser.parse_args())


type ExitCode = int


def subcommand_schema(
    logger: Logger,
    args: SchemaCommandParsedArgNamespace,
) -> ExitCode:
    generate_schemas(Path(args.schema_folder))

    logger.info(_("Successfully generated the schemas"))
    return 0


def subcommand_gui(
    _logger: Logger,
    args: GuiCommandParsedArgNamespace,
) -> ExitCode:
    config = Path(args.config)

    launch_gui(config)

    return 0


def subcommand_run(
    logger: Logger,
    args: RunCommandParsedArgNamespace,
) -> ExitCode:
    parsed_config = AdvancedConfig.load_and_resolve(
        Path(args.config),
        args.config_to_use,
    )
    if parsed_config is None:
        return 1

    launch_tui(logger, config=parsed_config)

    return 0


def subcommand_config_check(
    logger: Logger,
    args: ConfigCheckCommandParsedArgNamespace,
) -> ExitCode:
    config = Path(args.config)
    parsed_config = AdvancedConfig.load_and_resolve_with_info(
        config,
        args.config_to_use,
    )
    if parsed_config is None:
        logger.error(
            _("Config '{config}' is not valid!").format(config=config),
        )
        return 1

    final_config, info = parsed_config

    logger.info(_("Config '{config}' is valid!").format(config=config))
    logger.info("Info about config: %s", info)

    seriliazed_config: dict[str, Any] = serialize(
        FinalConfig,
        final_config,
    )
    logger.info("Printing final config as json:")
    logger.info(json.dumps(seriliazed_config, indent=4))
    return 0


def main() -> ExitCode:
    args = parse_args()
    logger: Logger = setup_custom_logger(args.level)

    try:
        match args.subcommand:
            case "schema":
                return subcommand_schema(
                    logger,
                    cast(SchemaCommandParsedArgNamespace, args),
                )
            case "gui":
                return subcommand_gui(logger, cast(GuiCommandParsedArgNamespace, args))
            case "run":
                return subcommand_run(
                    logger,
                    cast(
                        RunCommandParsedArgNamespace,
                        args,
                    ),
                )
            case "config_check":
                return subcommand_config_check(
                    logger,
                    cast(ConfigCheckCommandParsedArgNamespace, args),
                )
            case _:
                assert_never(args.subcommand)

    except KeyboardInterrupt:

        def exit_handler() -> None:
            print()  # noqa: T201
            print(_("Ctrl + C pressed"))  # noqa: T201

        atexit.register(exit_handler)
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
