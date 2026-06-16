#!/usr/bin/env python3


import argparse
import atexit
import json
import sys
from logging import Logger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    assert_never,
    cast,
)

from content.tagger.mp4_tagger import merge_dicts
from content.tagger.video_tagger import SerializableDict, SerializableDictValue
from helper.filter import Filter, FilterManager
from helper.log import LogLevel, setup_custom_logger
from helper.translation import get_translator
from helper.utils import parse_int_safely
from helper.validator import all_available_validators
from helper.version import PROGRAM_VERSION

_ = get_translator()


type SubCommand = Literal["run", "schema", "gui", "config_check", "api", "tagger"]


class ParsedArgNamespace:
    level: LogLevel
    subcommand: SubCommand


class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]
    config: Path
    template_to_use: Optional[str]
    filter: list[Filter]


class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_folder: Path


class GuiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["gui"]
    config: Path

    backend_host: str
    backend_port: int


TagCommand = Literal["read", "write"]


class TaggerCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["tagger"]

    tag_action: TagCommand


class TaggerReadCommandParsedArgNamespace(TaggerCommandParsedArgNamespace):
    tag_action: Literal["read"]
    file: Path


class TaggerWriteCommandParsedArgNamespace(TaggerCommandParsedArgNamespace):
    tag_action: Literal["write"]
    file: Path

    comment: str
    metadata: list[str]


AllTaggerCommandParsedArgNamespace = (
    TaggerReadCommandParsedArgNamespace | TaggerWriteCommandParsedArgNamespace
)


class ApiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["api"]
    config: Path

    host: str
    port: int


class ConfigCheckCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["config_check"]
    config: Path
    template_to_use: Optional[str]
    filter: list[Filter]


type AllParsedNameSpaces = (
    RunCommandParsedArgNamespace
    | SchemaCommandParsedArgNamespace
    | GuiCommandParsedArgNamespace
    | ApiCommandParsedArgNamespace
    | ConfigCheckCommandParsedArgNamespace
    | AllTaggerCommandParsedArgNamespace
)


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

    filter_manager = FilterManager(all_available_validators)

    def parse_filter_string(arg: str) -> Filter:
        if arg in ["help", "?", "h"]:
            filter_manager.print_help()
            raise SystemExit(0)

        filter_parsed = filter_manager.parse_filter(arg)

        if filter_parsed.err():
            raise argparse.ArgumentTypeError(filter_parsed.as_err())

        return filter_parsed.as_ok()

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
        default=Path("config.yaml"),
        type=Path,
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
        dest="filter",
        default=[],
        type=parse_filter_string,
        action="append",
        help=_(
            "Generic filter, see '--filter help' for all available options"  # noqa: COM812
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
        default=Path("schema/"),
        type=Path,
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
        default=Path("config.yaml"),
        type=Path,
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
        default=Path("config.yaml"),
        type=Path,
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
        default=Path("config.yaml"),
        type=Path,
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
        dest="filter",
        default=[],
        type=parse_filter_string,
        action="append",
        help=_(
            "Generic filter, see '--filter help' for all available options"  # noqa: COM812
        ),
    )

    tagger_parser = subparsers.add_parser(
        "tagger",
        description=_("Invoke the tagger on a file"),
    )

    tagger_subparsers = tagger_parser.add_subparsers(
        required=True,
        dest="tag_action",
    )

    tagger_read_parser = tagger_subparsers.add_parser(
        "read",
        description=_("read tags from a file"),
    )

    tagger_read_parser.add_argument(
        "-f",
        "--file",
        dest="file",
        required=True,
        type=Path,
        help=_("The file to read from"),
    )

    tagger_write_parser = tagger_subparsers.add_parser(
        "write",
        description=_("write tags to a file"),
    )

    tagger_write_parser.add_argument(
        "-f",
        "--file",
        dest="file",
        required=True,
        type=Path,
        help=_("The file to write to"),
    )

    tagger_write_parser.add_argument(
        "-c",
        "--comment",
        dest="comment",
        default="<No comment>",
        help=_("The comment to write"),
    )

    tagger_write_parser.add_argument(
        "-m",
        "--metadata",
        dest="metadata",
        default=[],
        action="append",
        help=_(
            "Add custom metadata to write, format: <key>:<value>, where key is a string, value can be a literal or a json encoded string",
        ),
    )

    return cast(AllParsedNameSpaces, parser.parse_args())


type ExitCode = int

# NOTE: no need to import these "heavy modules at top, so they are only imported when needed.
# some imports take literally seconds (my large iso language list, torch, speechbrain)

# ruff: disable[PLC0415]


def subcommand_schema(
    logger: Logger,
    args: SchemaCommandParsedArgNamespace,
) -> ExitCode:
    from main import generate_schemas

    generate_schemas(args.schema_folder)

    logger.info(_("Successfully generated the schemas"))
    return 0


def subcommand_gui(
    logger: Logger,
    args: GuiCommandParsedArgNamespace,
) -> ExitCode:
    from backend.backend import Address, BackendOptions
    from gui.gui import launch_gui
    from helper.config import AdvancedConfig

    raw_config = AdvancedConfig.load_raw(
        args.config,
    )
    if raw_config is None:
        logger.error(_("error while parsing config: can't load config"))
        return 1

    address = Address(host=args.backend_host, port=args.backend_port)
    options = BackendOptions(address=address)

    return launch_gui(options, raw_config, args.config)


def subcommand_api(
    logger: Logger,
    args: ApiCommandParsedArgNamespace,
) -> ExitCode:
    from backend.backend import Address, BackendOptions, launch_api
    from helper.config import AdvancedConfig

    raw_config = AdvancedConfig.load_raw(
        args.config,
    )
    if raw_config is None:
        logger.error(_("error while parsing config: can't load config"))
        return 1

    address = Address(host=args.host, port=args.port)
    options = BackendOptions(address=address)

    return launch_api(options, raw_config, args.config)


def subcommand_run(
    logger: Logger,
    args: RunCommandParsedArgNamespace,
) -> ExitCode:
    from helper.config import (
        AdvancedConfig,
        FileLockError,
        LockFile,
        filter_configs,
    )
    from helper.parser import CustomNameParser
    from helper.tui import launch_tui
    from main import AllContent

    if TYPE_CHECKING:
        from helper.manager import ConfigParameters

    parsed_config = AdvancedConfig.load_and_resolve(
        args.config,
        args.template_to_use,
    )
    if parsed_config.err():
        logger.error(
            _("error while parsing config: {err}").format(err=parsed_config.as_err()),
        )
        return 1

    parsed_configs = parsed_config.as_ok()

    if len(parsed_configs) == 0:
        logger.error(_("parsing returned 0 configs"))
        return 1

    configs = filter_configs(parsed_configs, args.filter)

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
        with LockFile.for_file(args.config):
            for index, config in enumerate(configs):
                name_parser = CustomNameParser(
                    season_special_names=config.parser.special,
                )

                config_paramaters: Optional[ConfigParameters] = (
                    None if len(configs) == 1 else (index, len(configs))
                )

                launch_tui(
                    logger=logger,
                    config=config,
                    name_parser=name_parser,
                    all_content_type=AllContent,
                    config_paramaters=config_paramaters,
                    filters=args.filter,
                )
    except FileLockError as err:
        logger.error(_("File lock error: {err}").format(err=str(err)))  # noqa: TRY400
        return 1
    return 0


def subcommand_config_check(
    logger: Logger,
    args: ConfigCheckCommandParsedArgNamespace,
) -> ExitCode:
    from apischema import serialize

    from helper.config import (
        AdvancedConfig,
        FinalConfig,
        filter_configs,
    )

    parsed_config = AdvancedConfig.load_and_resolve_with_info(
        args.config,
        args.template_to_use,
    )
    if parsed_config.err():
        logger.error(
            _("Config '{config}' is not valid: {err}").format(
                config=args.config,
                err=parsed_config.as_err(),
            ),
        )
        return 1

    final_config, info = parsed_config.as_ok()

    logger.info(_("Config '{config}' is valid!").format(config=args.config))
    logger.info(_("Info about config: {info}").format(info=info))

    if len(final_config) == 0:
        logger.error(_("parsing returned 0 configs"))
        return 1

    configs = filter_configs(final_config, args.filter)

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


def subcommand_tagger_read(
    logger: Logger,
    file: Path,
) -> ExitCode:
    from content.tagger.tagger import get_tagger_for_file
    from helper.manager import NoopManager

    handle_result = get_tagger_for_file(file)
    if handle_result.err():
        logger.error(
            _("Can't read tags from file '{file}': {reason}").format(
                file=file,
                reason=handle_result.as_err(),
            ),
        )
        return 1

    handle = handle_result.as_ok()

    manager = NoopManager()

    with handle.context(manager=manager) as ctx:
        tags = ctx.get_tags()

        logger.info(_("Read tags:"))

        logger.info(_("Comment: {comment}").format(comment=tags.comment))

        logger.info(_("UUID: {uuid}").format(uuid=tags.uuid))

        logger.info(_("Generic Metadata:"))
        for key, value in tags.metadata.items():
            msg = f"{key}: {value}"
            logger.info(msg)

        logger.info(_("Unrecognized tags:"))
        for key, value in tags.unrecognized:
            msg = f"{key}: {value}"
            logger.info(msg)

    return 0


def subcommand_tagger_write(
    logger: Logger,
    file: Path,
    args: TaggerWriteCommandParsedArgNamespace,
) -> ExitCode:
    from uuid import uuid4

    from content.tagger.tagger import get_tagger_for_file
    from content.tagger.video_tagger import MetadataTags
    from helper.manager import NoopManager, TuiManager

    handle_result = get_tagger_for_file(file)
    if handle_result.err():
        logger.error(
            _("Can't write tags from file '{file}': {reason}").format(
                file=file,
                reason=handle_result.as_err(),
            ),
        )
        return 1

    handle = handle_result.as_ok()

    tui_manager = TuiManager()

    def get_key_value(value: str) -> tuple[str, SerializableDictValue | None]:
        temp = value.split(":", 1)
        if len(temp) == 1:
            return (temp[0], None)

        if len(temp) != 2:
            msg = f"Implementation error, only two values expected, but got {len(temp)}"
            raise RuntimeError(msg)

        key, val = temp

        try:
            json_val = json.loads(val)
            return (key, json_val)  # noqa: TRY300
        except ValueError:
            return (key, val)

    metadata: SerializableDict = {}

    for value in args.metadata:
        key, val = get_key_value(value)
        metadata = merge_dicts(metadata, {key: val}, "error")

    write_tags: MetadataTags = MetadataTags(
        comment=args.comment,
        uuid=uuid4(),
        metadata=metadata,
    )

    with handle.context(manager=tui_manager) as ctx:
        ctx.write_tags(write_tags)

        logger.info(_("Wrote tags:"))

        logger.info(_("Comment: {comment}").format(comment=write_tags.comment))

        logger.info(_("UUID: {uuid}").format(uuid=write_tags.uuid))

        logger.info(_("Generic Metadata:"))
        for key, value in write_tags.metadata.items():
            msg = f"{key}: {value}"
            logger.info(msg)

    noop_manager = NoopManager()

    with handle.context(manager=noop_manager) as ctx:
        tags = ctx.get_tags()

        print()  # noqa: T201
        logger.info(_("Which resulted in these tags tags:"))

        logger.info(_("Comment: {comment}").format(comment=tags.comment))

        logger.info(_("UUID: {uuid}").format(uuid=tags.uuid))

        logger.info(_("Generic Metadata:"))
        for key, value in tags.metadata.items():
            msg = f"{key}: {value}"
            logger.info(msg)

        logger.info(_("Unrecognized tags:"))
        for key, value in tags.unrecognized:
            msg = f"{key}: {value}"
            logger.info(msg)

    return 0


def subcommand_tagger(
    logger: Logger,
    args: AllTaggerCommandParsedArgNamespace,
) -> ExitCode:
    if not args.file.exists():
        logger.error(_("File '{file}' doesn't exist").format(file=args.file))
        return 1

    if args.tag_action == "read":
        return subcommand_tagger_read(logger, args.file)

    if args.tag_action == "write":
        return subcommand_tagger_write(logger, args.file, args)

    assert_never(args.tag_action)


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
            case "tagger":
                return subcommand_tagger(
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


# ruff: enable[PLC0415]

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
