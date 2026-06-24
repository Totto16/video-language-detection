#!/usr/bin/env python3


import argparse
import atexit
import json
import sys
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import (
    Any,
    Literal,
    Never,
    Optional,
    Self,
    assert_never,
    assert_type,
    cast,
    override,
)

from content.tagger.utils import merge_dicts
from content.tagger.video_tagger import (
    InspectElement,
    InspectNotImplemented,
    InspectPrinter,
    InspectPriority,
    SerializableDict,
    SerializableDictValue,
)
from helper.base import app_status_bar_manager
from helper.decorator import decorate_class
from helper.filter import Filter, FilterHelpOptions, FilterManager
from helper.log import LogLevel, setup_custom_logger
from helper.manager import TuiManager
from helper.translation import get_translator
from helper.utils import parse_int_safely
from helper.validator import validator_checks
from helper.version import PROGRAM_VERSION

_ = get_translator()


type SubCommand = Literal[
    "run",
    "schema",
    "gui",
    "config_check",
    "api",
    "tagger",
    "ffmpeg",
]


@decorate_class(slots=False, allow_defaults=False)
class ParsedArgNamespace(argparse.Namespace):
    level: LogLevel
    subcommand: SubCommand


@decorate_class(slots=False, allow_defaults=False)
class RunCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["run"]
    config: Path
    template_to_use: Optional[str]
    filter: list[Filter]


@decorate_class(slots=False, allow_defaults=False)
class SchemaCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["schema"]
    schema_folder: Path


@decorate_class(slots=False, allow_defaults=False)
class GuiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["gui"]
    config: Path

    backend_host: str
    backend_port: int


TagCommand = Literal["read", "write", "inspect"]


@decorate_class(slots=False, allow_defaults=False)
class TaggerCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["tagger"]

    tag_action: TagCommand


@decorate_class(slots=False, allow_defaults=False)
class TaggerReadCommandParsedArgNamespace(TaggerCommandParsedArgNamespace):
    tag_action: Literal["read"]
    file: Path


type ShortLanguageStrWrapper = Any


@decorate_class(slots=False, allow_defaults=False)
class TaggerWriteCommandParsedArgNamespace(TaggerCommandParsedArgNamespace):
    tag_action: Literal["write"]
    file: Path

    comment: str
    metadata: list[str]
    language: Optional[ShortLanguageStrWrapper]


class InspectOutputFormat(Enum):
    Json = "json"
    Normal = "normal"

    @staticmethod
    def from_str(inp: str) -> Optional["InspectOutputFormat"]:
        for level in InspectOutputFormat:
            if str(level).lower() == inp.lower():
                return level

        return None

    def __str__(self: Self) -> str:
        return str(self.name).lower()

    def __repr__(self: Self) -> str:
        return self.__str__()


@decorate_class(slots=False, allow_defaults=False)
class TaggerInspectCommandParsedArgNamespace(TaggerCommandParsedArgNamespace):
    tag_action: Literal["inspect"]
    file: Path

    output_format: InspectOutputFormat
    priority: InspectPriority


AllTaggerCommandParsedArgNamespace = (
    TaggerReadCommandParsedArgNamespace
    | TaggerWriteCommandParsedArgNamespace
    | TaggerInspectCommandParsedArgNamespace
)


FfmpegCommand = Literal["scan", "fix-chapter"]


@decorate_class(slots=False, allow_defaults=False)
class FfmpegCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["ffmpeg"]

    ffmpeg_command: FfmpegCommand


@decorate_class(slots=False, allow_defaults=False)
class FfmpegScanCommandParsedArgNamespace(FfmpegCommandParsedArgNamespace):
    ffmpeg_command: Literal["scan"]

    files: list[str]


@decorate_class(slots=False, allow_defaults=False)
class FfmpegFixChapterCommandParsedArgNamespace(FfmpegCommandParsedArgNamespace):
    ffmpeg_command: Literal["fix-chapter"]

    files: list[str]


AllFfmpegCommandParsedArgNamespace = (
    FfmpegScanCommandParsedArgNamespace | FfmpegFixChapterCommandParsedArgNamespace
)


@decorate_class(slots=False, allow_defaults=False)
class ApiCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["api"]
    config: Path

    host: str
    port: int


@decorate_class(slots=False, allow_defaults=False)
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
    | AllFfmpegCommandParsedArgNamespace
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


def parse_short_language(arg: str) -> ShortLanguageStrWrapper:
    from content.language import ShortLanguageStr  # noqa: PLC0415

    return ShortLanguageStr.from_str_unsafe(arg)


def parse_args() -> AllParsedNameSpaces:  # noqa: PLR0915

    def help_cb() -> Never:
        raise SystemExit(0)

    filter_manager = FilterManager(
        validator_checks=validator_checks,
        help_options=FilterHelpOptions(cb=help_cb),
    )

    def parse_filter_string(arg: str) -> Filter:
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

    tagger_write_parser.add_argument(
        "-l",
        "--language",
        dest="language",
        default=None,
        type=parse_short_language,
        help=_(
            "Add a language to all streams",
        ),
    )

    tagger_inspect_parser = tagger_subparsers.add_parser(
        "inspect",
        description=_("read tags from a file"),
    )

    tagger_inspect_parser.add_argument(
        "-f",
        "--file",
        dest="file",
        required=True,
        type=Path,
        help=_("The file to inspect"),
    )

    output_format_choices: list[InspectOutputFormat] = [
        InspectOutputFormat.Json,
        InspectOutputFormat.Normal,
    ]
    output_format_default: InspectOutputFormat = InspectOutputFormat.Normal
    tagger_inspect_parser.add_argument(
        "-o",
        "--output",
        choices=output_format_choices,
        default=output_format_default,
        dest="output_format",
        type=lambda s: InspectOutputFormat.from_str(s)
        or cast(InspectOutputFormat, s.lower()),
        help=_("The output format to use"),
    )

    priority_choices: list[InspectPriority] = [
        InspectPriority.Ignore,
        InspectPriority.Normal,
        InspectPriority.Important,
    ]
    priority_default: InspectPriority = InspectPriority.Normal
    tagger_inspect_parser.add_argument(
        "-p",
        "--priority",
        choices=priority_choices,
        default=priority_default,
        dest="priority",
        type=lambda s: InspectPriority.from_str(s) or cast(InspectPriority, s.lower()),
        help=_("Which boxes to inspect"),
    )

    ffmpeg_parser = subparsers.add_parser(
        "ffmpeg",
        description=_("FFmpeg helper actions"),
    )

    ffmpeg_subparsers = ffmpeg_parser.add_subparsers(
        required=True,
        dest="ffmpeg_command",
    )

    ffmpeg_scan_parser = ffmpeg_subparsers.add_parser(
        "scan",
        description=_("scan files"),
    )

    ffmpeg_scan_parser.add_argument(
        nargs="*",
        dest="files",
        help=_("The files to process"),
    )

    ffmpeg_fix_chapter_parser = ffmpeg_subparsers.add_parser(
        "fix-chapter",
        description=_("fix chapters"),
    )

    ffmpeg_fix_chapter_parser.add_argument(
        nargs="*",
        dest="files",
        help=_("The files to process"),
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
    from helper.custom_parser import CustomNameParser
    from helper.tui import launch_tui
    from main import AllContent

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

    manager = TuiManager()

    try:
        with (
            LockFile.for_file(args.config),
            app_status_bar_manager(configs, manager) as status_bar,
        ):
            for index, config in enumerate(configs):
                with status_bar.config(index, config):
                    name_parser = CustomNameParser(
                        season_special_names=config.parser.special,
                    )

                    launch_tui(
                        logger=logger,
                        config=config,
                        name_parser=name_parser,
                        all_content_type=AllContent,
                        filters=args.filter,
                        manager=manager,
                        status_bar=status_bar,
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
    from content.language import Language
    from content.tagger.tagger import get_tagger_for_file
    from helper.manager import NoopManager
    from helper.result import Result

    def read_language_str(lang_res: Result[Optional[Language], str]) -> str:
        if lang_res.err():
            return _("<Err: {err}>").format(err=lang_res.as_err())

        lang = lang_res.as_ok()

        if lang is None:
            return _("<Nothing>")

        return f"{lang}"

    handle_result = get_tagger_for_file(file)
    if handle_result.err():
        logger.error(
            _(
                "Can't read tags from file '{file}': Opening a handle failed: {reason}"  # noqa: COM812
            ).format(
                file=file,
                reason=handle_result.as_err(),
            ),
        )
        return 1

    handle = handle_result.as_ok()

    manager = NoopManager()

    with handle.r_ctx(manager=manager) as ctx:
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

        language = read_language_str(ctx.read_language())

        logger.info(_("Read Language: {language}").format(language=language))
    return 0


def subcommand_tagger_write(  # noqa: PLR0915
    logger: Logger,
    file: Path,
    args: TaggerWriteCommandParsedArgNamespace,
) -> ExitCode:
    from uuid import uuid4

    from content.language import Language, ShortLanguageStr
    from content.tagger.tagger import get_tagger_for_file
    from content.tagger.video_tagger import MetadataTags
    from helper.manager import NoopManager, TuiManager
    from helper.result import Result

    def read_language_str(lang_res: Result[Optional[Language], str]) -> str:
        if lang_res.err():
            return _("<Err: {err}>").format(err=lang_res.as_err())

        lang = lang_res.as_ok()

        if lang is None:
            return _("<Nothing>")

        return f"{lang}"

    handle_result = get_tagger_for_file(file)
    if handle_result.err():
        logger.error(
            _(
                "Can't write tags to file '{file}': Opening a handle failed: {reason}"  # noqa: COM812
            ).format(
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
            msg = _(
                "Implementation error, only two values expected, but got {length}"  # noqa: COM812
            ).format(length=len(temp))
            raise RuntimeError(msg)

        key, val = temp

        try:
            json_val = json.loads(val)
            return (key, json_val)  # noqa: TRY300
        except ValueError:
            return (key, val)

    metadata: SerializableDict = {}

    for value_m in args.metadata:
        key, val = get_key_value(value_m)

        if val is None:
            metadata = merge_dicts(metadata, {key: ""}, "error")
            continue

        metadata = merge_dicts(metadata, {key: val}, "error")

    write_tags: MetadataTags = MetadataTags(
        comment=args.comment,
        uuid=uuid4(),
        metadata=metadata,
    )

    with handle.w_ctx(manager=tui_manager) as w_ctx:
        w_ctx.write_tags(write_tags)

        logger.info(_("Wrote tags:"))

        logger.info(_("Comment: {comment}").format(comment=write_tags.comment))

        logger.info(_("UUID: {uuid}").format(uuid=write_tags.uuid))

        logger.info(_("Generic Metadata:"))
        for key_m, value in write_tags.metadata.items():
            msg = f"{key_m}: {value}"
            logger.info(msg)

        if args.language is not None:
            if not isinstance(args.language, ShortLanguageStr):
                msg = _("Implementation error: language is wrong type: {typ}").format(
                    typ=type(args.language),
                )
                raise RuntimeError(msg)

            language = Language.from_values_unsafe(str(args.language), "Not applicable")
            w_ctx.write_language(language)

            logger.info(_("Wrote language: {lang}").format(lang=str(language.short)))

    noop_manager = NoopManager()

    with handle.r_ctx(manager=noop_manager) as r_ctx:
        tags = r_ctx.get_tags()

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

        read_language = read_language_str(r_ctx.read_language())

        logger.info(_("Read Language: {language}").format(language=read_language))

    return 0


def subcommand_tagger_inspect(
    logger: Logger,
    args: TaggerInspectCommandParsedArgNamespace,
) -> ExitCode:
    from content.tagger.tagger import get_tagger_for_file

    handle_result = get_tagger_for_file(args.file)
    if handle_result.err():
        logger.error(
            _(
                "Can't inspect file '{file}': Opening a handle failed: {reason}"  # noqa: COM812
            ).format(
                file=args.file,
                reason=handle_result.as_err(),
            ),
        )
        return 1

    handle = handle_result.as_ok()

    if args.output_format == InspectOutputFormat.Normal:
        logger.info(_("Inspect file: {file}").format(file=args.file.absolute()))

    @decorate_class(slots=True)
    class NormalPrinter(InspectPrinter):

        def __init__(self: Self) -> None:
            super().__init__()

        @override
        def element(self: Self, element: InspectElement, depth: int) -> None:

            print(f"{" " * depth}{element.name}")  # noqa: T201

        @override
        def start(
            self: Self,
        ) -> None:
            pass

        @override
        def end(
            self: Self,
        ) -> None:
            pass

    @decorate_class(slots=True)
    class JsonPrinter(InspectPrinter):
        __pos: int

        def __init__(self: Self) -> None:
            super().__init__()

            self.__pos = 0

        @override
        def element(self: Self, element: InspectElement, depth: int) -> None:

            entry: dict[str, str | int] = {
                "depth": depth,
                "name": element.name,
                "size": element.size,
            }

            if self.__pos != 0:
                print(",")  # noqa: T201

            print(json.dumps(entry), end="")  # noqa: T201

            self.__pos = self.__pos + 1

        @override
        def start(
            self: Self,
        ) -> None:
            print("[")  # noqa: T201

        @override
        def end(
            self: Self,
        ) -> None:
            print("\n]")  # noqa: T201

    printer: InspectPrinter
    match args.output_format:
        case InspectOutputFormat.Normal:
            printer = NormalPrinter()
        case InspectOutputFormat.Json:
            printer = JsonPrinter()
        case _:
            assert_never(args.output_format)

    inspect_res = handle.inspect(printer, args.priority)

    if isinstance(inspect_res, InspectNotImplemented):
        logger.error(
            _(
                "Can't inspect file '{file}': Inspection not supported"  # noqa: COM812
            ).format(
                file=args.file,
            ),
        )
        return 1

    assert_type(inspect_res, None)

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

    if args.tag_action == "inspect":
        return subcommand_tagger_inspect(logger, args)

    assert_never(args.tag_action)


def subcommand_ffmpeg(
    logger: Logger,
    args: AllFfmpegCommandParsedArgNamespace,
) -> ExitCode:
    files: list[Path] = [Path(file) for file in args.files]
    if len(files) == 0:
        logger.error(_("No path given, using CWD"))
        files = [Path.cwd().absolute()]

    from ffmpeg_helper.fix_chapters import fix_chapters
    from ffmpeg_helper.scan_files import scan_files

    match args.ffmpeg_command:
        case "scan":
            errors = scan_files(files)
            if len(errors) == 0:
                return 0
            for err in errors:
                logger.error(err)
            return 1
        case "fix-chapter":
            errors = fix_chapters(files)
            if len(errors) == 0:
                return 0
            for err in errors:
                logger.error(err)
            return 1
        case _:
            assert_never(args.ffmpeg_command)


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
            case "ffmpeg":
                return subcommand_ffmpeg(
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
