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

    subparsers = parser.add_subparsers(required=True)
    parser.set_defaults(subcommand="run")

    scan_parser = subparsers.add_parser("scan")
    scan_parser.set_defaults(subcommand="scan")

    fix_chapter_parser = subparsers.add_parser("fix-chapter")
    fix_chapter_parser.set_defaults(subcommand="fix-chapter")

    for sub_parser in [scan_parser, fix_chapter_parser]:
        sub_parser.add_argument(
            nargs="*",
            dest="files",
        )

    args = cast(AllParsedNameSpaces, parser.parse_args())
    try:
        files: list[Path] = [Path(file) for file in args.files]
        if len(files) == 0:
            print("No path given, using CWD")  # noqa: T201
            files = [Path.cwd().absolute()]

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
