#!/usr/bin/env python3


import argparse
import atexit
import json
import sys
from logging import Logger
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    assert_never,
    cast,
)

from apischema import serialize

from backend.backend import Address, BackendOptions, launch_api
from gui.gui import launch_gui
from helper.config import (
    AdvancedConfig,
    ConfigFilter,
    FileLockError,
    FinalConfig,
    LockFile,
    filter_configs,
    parse_config_filter_string,
)
from helper.log import LogLevel, setup_custom_logger
from helper.parser import CustomNameParser
from helper.timestamp import parse_int_safely
from helper.translation import get_translator
from helper.tui import launch_tui
from helper.version import PROGRAM_VERSION
from main import AllContent, generate_schemas

type SubCommand = Literal["run", "schema", "gui", "config_check", "api"]


class ParsedArgNamespace:
    level: LogLevel
    subcommand: SubCommand


class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]
    config: str
    template_to_use: Optional[str]
    config_filter: Optional[ConfigFilter]


class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_folder: str


class GuiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["gui"]
    config: str

    backend_host: str
    backend_port: int


class ApiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["api"]
    config: str

    host: str
    port: int


class ConfigCheckCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["config_check"]
    config: str
    template_to_use: Optional[str]
    config_filter: Optional[ConfigFilter]


type AllParsedNameSpaces = (
    RunCommandParsedArgNamespace
    | SchemaCommandParsedArgNamespace
    | GuiCommandParsedArgNamespace
    | ApiCommandParsedArgNamespace
    | ConfigCheckCommandParsedArgNamespace
)

_ = get_translator()


def parse_port(arg: str) -> int:
    value = parse_int_safely(arg)
    if value is None:
        msg = (
            _("expected the argument to be a port but got: {arg}").format(
                arg=arg,
            ),
        )
        raise argparse.ArgumentTypeError(msg)

    return value


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
        dest="template_to_use",
        default=None,
        help=_(
            "The config template to use, if the config specifies, to use the cli one"  # noqa: COM812
        ),
    )
    run_parser.add_argument(
        "-f",
        "--filter",
        dest="config_filter",
        default=None,
        type=parse_config_filter_string,
        action="append",
        help=_(
            "Filter the provided configs, allowed are names or indices"  # noqa: COM812
        ),
    )

    schema_parser = subparsers.add_parser(
        "schema",
        description=_("Create schemas for config and the resulting data"),
    )
    schema_parser.add_argument(
        "-s",
        "--schema-folder",
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
    gui_parser.add_argument(
        "-b",
        "--backend-host",
        dest="backend_host",
        default="127.0.0.1",
        help=_("The host the backend runs on"),
    )
    gui_parser.add_argument(
        "-p",
        "--port",
        dest="backend_port",
        default=4433,
        type=parse_port,
        help=_("The port the backend runs on"),
    )

    api_parser = subparsers.add_parser(
        "api",
        description=_("Run the API"),
    )
    api_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.yaml",
        help=_("The config to use"),
    )
    api_parser.add_argument(
        "-b",
        "--host",
        dest="host",
        default="127.0.0.1",
        help=_("The host the backend runs on"),
    )
    api_parser.add_argument(
        "-p",
        "--port",
        dest="port",
        default=4433,
        type=parse_port,
        help=_("The port the backend runs on"),
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
        dest="template_to_use",
        default=None,
        help=_(
            "The config template to use, if the config specifies, to use the cli one"  # noqa: COM812
        ),
    )
    config_check_parser.add_argument(
        "-f",
        "--filter",
        dest="config_filter",
        default=None,
        type=parse_config_filter_string,
        action="append",
        help=_(
            "Filter the provided configs, allowed are names or indices"  # noqa: COM812
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
    logger: Logger,
    args: GuiCommandParsedArgNamespace,
) -> ExitCode:
    config_file_path = Path(args.config)

    raw_config = AdvancedConfig.load_raw(
        config_file_path,
    )
    if raw_config is None:
        logger.error(_("error while parsing config: can't load config"))
        return 1

    address = Address(host=args.backend_host, port=args.backend_port)
    options = BackendOptions(address=address)

    return launch_gui(options, raw_config, config_file_path)


def subcommand_api(
    logger: Logger,
    args: ApiCommandParsedArgNamespace,
) -> ExitCode:
    config_file_path = Path(args.config)

    raw_config = AdvancedConfig.load_raw(
        config_file_path,
    )
    if raw_config is None:
        logger.error(_("error while parsing config: can't load config"))
        return 1

    address = Address(host=args.host, port=args.port)
    options = BackendOptions(address=address)

    return launch_api(options, raw_config, config_file_path)


def subcommand_run(
    logger: Logger,
    args: RunCommandParsedArgNamespace,
) -> ExitCode:
    parsed_config = AdvancedConfig.load_and_resolve(
        Path(args.config),
        args.template_to_use,
    )
    if parsed_config.is_err():
        logger.error(_("error while parsing config: %s"), parsed_config.get_err())
        return 1

    parsed_configs = parsed_config.get_ok()

    if len(parsed_configs) == 0:
        logger.error(_("parsing returned 0 configs"))
        return 1

    configs = filter_configs(parsed_configs, args.config_filter)

    if len(configs) > len(parsed_configs):
        logger.error(
            _(
                "filtering returned more configs than there are, at least one was used multiple times"  # noqa: COM812
            ),
        )
        return 1

    if len(configs) == 0:
        logger.error(_("filtering returned 0 configs"))
        return 1

    try:
        with LockFile.for_file(Path(args.config)):
            for index, config in enumerate(configs):
                name_parser = CustomNameParser(
                    season_special_names=config.parser.special,
                )

                config_paramaters: Optional[tuple[int, int]] = (
                    None if len(configs) == 1 else (index, len(configs))
                )

                launch_tui(
                    logger=logger,
                    config=config,
                    name_parser=name_parser,
                    all_content_type=AllContent,
                    config_paramaters=config_paramaters,
                )
    except FileLockError as err:
        logger.error(_("File lock error: %s"), str(err))  # noqa: TRY400
        return 1
    return 0


def subcommand_config_check(
    logger: Logger,
    args: ConfigCheckCommandParsedArgNamespace,
) -> ExitCode:
    config = Path(args.config)
    parsed_config = AdvancedConfig.load_and_resolve_with_info(
        config,
        args.template_to_use,
    )
    if parsed_config.is_err():
        logger.error(
            _("Config '{config}' is not valid: {err}").format(
                config=config,
                err=parsed_config.get_err(),
            ),
        )
        return 1

    final_config, info = parsed_config.get_ok()

    logger.info(_("Config '{config}' is valid!").format(config=config))
    logger.info(_("Info about config: %s"), info)

    if len(final_config) == 0:
        logger.error(_("parsing returned 0 configs"))
        return 1

    configs = filter_configs(final_config, args.config_filter)

    if len(configs) > len(final_config):
        logger.error(
            _(
                "filtering returned more configs than there are, at least one was used multiple times"  # noqa: COM812
            ),
        )
        return 1

    if len(configs) == 0:
        logger.error(_("filtering returned 0 configs"))
        return 1

    serialized_config: dict[str, Any] = serialize(
        FinalConfig,
        final_config,
    )
    logger.info(_("Printing final config as json:"))
    logger.info(json.dumps(serialized_config, indent=4))
    return 0


def main() -> ExitCode:
    args = parse_args()
    logger: Logger = setup_custom_logger(args.level)

    try:
        match args.subcommand:
            case "schema":
                return subcommand_schema(
                    logger,
                    args,
                )
            case "gui":
                return subcommand_gui(logger, args)
            case "api":
                return subcommand_api(logger, args)
            case "run":
                return subcommand_run(
                    logger,
                    args,
                )
            case "config_check":
                return subcommand_config_check(
                    logger,
                    args,
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
