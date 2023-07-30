#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import (
    Optional,
)

from content.base_class import Content
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import (
    NameParser,
    ScannedFile,
    ScannedFileType,
)
from content.season_content import SeasonContent
from content.series_content import SeriesContent


def content_from_scan(
    file_path: Path,
    file_type: ScannedFileType,
    parent_folders: list[str],
    name_parser: NameParser,
) -> Optional[Content]:
    scanned_file: ScannedFile = ScannedFile.from_scan(
        file_path,
        file_type,
        parent_folders,
    )

    name = file_path.name

    # [collection] -> series -> season -> file_name
    parents: list[str] = [*parent_folders, name]

    top_is_collection: bool = (
        (len(parents) == 1 and file_path.is_dir()) or len(parents) != 1
    ) and not SeriesContent.is_valid_name(parents[0], name_parser)

    try:
        if len(parents) == 4:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting 4 - {file_path}!",
                )

            return EpisodeContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 3 and not top_is_collection:
            if file_type == ScannedFileType.folder:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting 3 - {file_path}!",
                )

            return EpisodeContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 3 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting 3 - {file_path}!",
                )

            return SeasonContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 2 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!",
                )

            return SeasonContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 2 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting: 2 - {file_path}!",
                )

            return SeriesContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 1 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!",
                )

            return SeriesContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 1 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise RuntimeError(
                    f"Not expected file type {file_type} with the received nesting: 1 - {file_path}!",
                )
            return CollectionContent.from_path(file_path, scanned_file)

        raise RuntimeError("UNREACHABLE")

    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        return None