from logging import Logger
from pathlib import Path
from typing import (
    Never,
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
from helper.log import get_logger

logger: Logger = get_logger()


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

    def raise_inner(msg: str) -> Never:
        raise RuntimeError(msg)

    try:
        if len(parents) == 4:
            if file_type == ScannedFileType.folder:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting 4: '{file_path}",
                )

            return EpisodeContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 3 and not top_is_collection:
            if file_type == ScannedFileType.folder:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting 3: '{file_path}",
                )

            return EpisodeContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 3 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting 3: '{file_path}",
                )

            return SeasonContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 2 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting: 2: '{file_path}",
                )

            return SeasonContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 2 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting: 2: '{file_path}",
                )

            return SeriesContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 1 and not top_is_collection:
            if file_type == ScannedFileType.file:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting: 1: '{file_path}",
                )

            return SeriesContent.from_path(file_path, scanned_file, name_parser)
        if len(parents) == 1 and top_is_collection:
            if file_type == ScannedFileType.file:
                raise_inner(
                    f"Not expected file type {file_type} with the received nesting: 1: '{file_path}",
                )

            return CollectionContent.from_path(file_path, scanned_file)

        raise_inner("UNREACHABLE")
    except NameError as err:
        logger.error(str(err))  # noqa: TRY400
        return None
    except Exception:
        logger.exception("Content scan")
        return None
