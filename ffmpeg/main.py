#!/usr/bin/env python3


import argparse
import atexit
from pathlib import Path
from typing import Literal, cast

from fix_chapters import fix_chapters
from scan_files import scan_files

SubCommand = Literal["scan", "fix-chapter"]


class ParsedArgNamespace:
    files: list[str]
    subcommand: SubCommand


class ScanCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["scan"]


class FixChapterCommandParsedArgNamespace(ParsedArgNamespace):
    subcommand: Literal["fix-chapter"]


AllParsedNameSpaces = (
    ScanCommandParsedArgNamespace | FixChapterCommandParsedArgNamespace
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ffmpeg-helper",
        description="Scan Video Files",
    )

    parser.add_argument(
        nargs="*",
        dest="files",
    )

    subparsers: argparse._SubParsersAction[  # noqa: SLF001
        argparse.ArgumentParser
    ] = parser.add_subparsers(required=False)
    parser.set_defaults(subcommand="run")

    scan_parser = subparsers.add_parser("scan")
    scan_parser.set_defaults(subcommand="scan")

    fix_chapter_parser = subparsers.add_parser("fix-chapter")
    fix_chapter_parser.set_defaults(subcommand="fix-chapter")

    args = cast(AllParsedNameSpaces, parser.parse_args())
    try:
        files = args.files
        if len(files) == 0:
            files = [str(Path.cwd().absolute())]

        match args.subcommand:
            case "scan":
                scan_files(files)
            case "fix-chapter":
                fix_chapters(files)

    except KeyboardInterrupt:

        def exit_handler() -> None:
            print()  # noqa: T201
            print("Ctrl + C pressed")  # noqa: T201

        atexit.register(exit_handler)
