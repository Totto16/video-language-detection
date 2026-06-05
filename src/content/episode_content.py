import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Self,
    override,
)

from apischema import alias, schema
from apischema.metadata import none_as_undefined

from content.base_class import (
    CallbackData,
    Content,
    ContentCharacteristic,
)
from content.general import (
    Callback,
    CallbackWorkload,
    ContentType,
    EpisodeDescription,
    NameParser,
    ScannedFile,
)
from content.language import Language
from content.metadata.metadata import HandlesType, MetadataHandle, SkipHandle
from content.shared import ScanType
from content.summary import Summary
from content.tagger.tagger import get_tagger_for_file
from content.video_metadata import VideoMetadata
from helper.apischema import narrow_type
from helper.error import ErrorMode
from helper.log import get_logger
from helper.manager import CounterInterface, ManagerInterface
from helper.translation import get_translator
from helper.version import PROGRAM_VERSION

logger: Logger = get_logger()
_ = get_translator()


# see below on why this hacks is needed
needs_migration_for_video_metadata: bool = os.getenv(
    "VIDEO_LANG_DETECT_MIGRATION_FOR_VIDEO_METADATA",
) in ["1", "true", "TRUE"]


global_counter_wip = 1


@schema(extra=narrow_type(("type", Literal[ContentType.episode])))
@dataclass(slots=True, repr=True)
class EpisodeContent(Content):
    __description: EpisodeDescription = field(metadata=alias("description"))
    __language: Language = field(metadata=alias("language"))
    __video_metadata: Optional[VideoMetadata] = field(
        default=None,
        metadata=alias("video_metadata") | none_as_undefined,
    )

    @staticmethod
    def from_path(
        path: Path,
        scanned_file: ScannedFile,
        name_parser: NameParser,
    ) -> "EpisodeContent":
        description: Optional[EpisodeDescription] = EpisodeContent.parse_description(
            path.name,
            name_parser,
        )
        if description is None:
            msg = _("Couldn't get EpisodeDescription from '{path}'").format(path=path)
            raise NameError(msg, name="EpisodeDescription")

        if description.episode < 1:
            msg = _(
                "EpisodeDescription is invalid, episode number is < 1: {description} -> '{path}'"  # noqa: COM812
            ).format(description=description, path=path)
            raise NameError(msg, name="EpisodeDescription")

        return EpisodeContent(
            ContentType.episode,
            scanned_file,
            None,
            description,
            Language.get_default(),
            None,
        )

    @property
    def description(self: Self) -> EpisodeDescription:
        return self.__description

    @property
    def language(self: Self) -> Language:
        return self.__language

    @staticmethod
    def is_valid_name(
        name: str,
        name_parser: NameParser,
    ) -> bool:
        return EpisodeContent.parse_description(name, name_parser) is not None

    @staticmethod
    def parse_description(
        name: str,
        name_parser: NameParser,
    ) -> Optional[EpisodeDescription]:
        result = name_parser.parse_episode_name(name)
        if result is None:
            return None

        return EpisodeDescription(
            name=result.name,
            season=result.season,
            episode=result.episode,
        )

    @override
    def summary(self: Self, *, detailed: bool = False) -> Summary:
        return Summary.construct_for_episode(
            language=self.__language,
            metadata=self.metadata,
            description=self.__description,
            video_metadata=self.__video_metadata,
            detailed=detailed,
        )

    def __get_handles(
        self: Self,
        handles: HandlesType,
    ) -> Optional[tuple[MetadataHandle, MetadataHandle] | SkipHandle]:
        if handles is None:
            return None

        if isinstance(handles, SkipHandle):
            return SkipHandle()

        if len(handles) != 2:
            msg = _("Length of handles is invalid, expected 2 but got {length}").format(
                length=len(handles),
            )
            logger.warning(msg)
            return None

        return (handles[0], handles[1])

    def __reset_metadata_of_file(self: Self) -> None:
        print("RESET OF THE METADATA")
        self.__language = Language.get_default()
        self.scanned_file.reset_file_data()
        self.__video_metadata = None
        # note, reset other metadata here, once new one is added

    def __metadata_for_file(self: Self) -> str:
        metadata: dict[str, str | int | dict[str, Any] | dict[str, str | int]] = {
            "original_path": str(self.scanned_file.path.absolute()),
            "description": {
                "name": self.__description.name,
                "season": self.__description.season,
                "episode": self.__description.episode,
            },
            "language": {
                "short": str(self.language.short),
                "long": str(self.language.long),
            },
        }

        if self.scanned_file.stats.checksum is not None:
            metadata["checksum"] = self.scanned_file.stats.checksum

        serialized_metadata = (
            None if self.metadata is None else self.metadata.serialize()
        )
        if serialized_metadata is not None:
            metadata["metadata"] = serialized_metadata

        return json.dumps(metadata)

    def update_video_metadata(
        self: Self,
        manager: ManagerInterface,
        callback: Callback[Content, ContentCharacteristic, CallbackData],
        error_mode: ErrorMode,
        *,
        only_update_file: bool = False,
    ) -> None:

        changed_file: bool = False

        def write_file_metadata() -> None:
            nonlocal changed_file

            handle_result = get_tagger_for_file(self.scanned_file.path)
            if handle_result.is_err():
                logger.error(
                    _("Can't tag the file '{file}': {reason}").format(
                        file=self.scanned_file.path,
                        reason=handle_result.get_err(),
                    ),
                )
                return

            handle = handle_result.get_ok()

            try:

                def metadata_prefix(name: str) -> str:
                    return f"video_language_detect_{name}"

                now = datetime.now()  # noqa: DTZ005

                metadata: dict[str, str] = {
                    metadata_prefix("metadata"): self.__metadata_for_file(),
                    metadata_prefix("version"): PROGRAM_VERSION,
                    metadata_prefix("iso_time"): now.isoformat(),
                }

                # TODO: remove
                global global_counter_wip

                if global_counter_wip > 0:
                    global_counter_wip -= 1

                    with handle.writer(manager=manager) as writer:
                        writer.write_metadata(
                            comment=[
                                "see other metadata for more info by video_language_detect",
                            ],
                            language=self.language,
                            metadata=metadata,
                        )
                        print(metadata)
                        self.scanned_file.reset_file_data()
                        changed_file = True
            except RuntimeError as err:
                logger.error(  # noqa: TRY400
                    _("Write Video Metadata: {err}").format(err=err),
                )

        def update_checksum() -> None:
            if changed_file:
                self.generate_checksum(manager)

        def analyze_vide_metadata() -> None:
            if only_update_file:
                return

            try:
                if self.__video_metadata is None:
                    bar: CounterInterface = manager.counter(
                        total=1,
                        desc="get video metadata",
                        leave=False,
                        color="red",
                    )
                    bar.update(0, force=True)

                    result = VideoMetadata.from_file(
                        file=self.scanned_file.path,
                        error_mode=error_mode,
                    )
                    if result.is_ok():
                        self.__video_metadata = result.get_ok()
                    else:
                        logger.error(
                            _(
                                "Error in getting video metadata for file '{file}': {reason}"  # noqa: COM812
                            ).format(
                                file=self.scanned_file.path,
                                reason=result.get_err(),
                            ),
                        )

                    bar.close(clear=True)
                    print("video_metadata", self.__video_metadata)
            except RuntimeError as err:
                logger.error(  # noqa: TRY400
                    _("Analyze Video Metadata: {err}").format(err=err),
                )

        callback_workload: list[CallbackWorkload] = [
            write_file_metadata,
            update_checksum,
            analyze_vide_metadata,
        ]

        characteristic: ContentCharacteristic = (self.type, self.scanned_file.type)

        callback.process_workload(
            callback_workload,
            "update video metadata",
            self.scanned_file.parents,
            characteristic,
        )

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, CallbackData],
        *,
        handles: HandlesType,
        parent_folders: list[str],
        trailer_names: list[str],
        rescan: bool = False,
    ) -> None:
        manager, scanner, language_picker, error_mode = callback.get_saved().as_tuple()

        current_handles = self.__get_handles(handles)

        characteristic: ContentCharacteristic = (self.type, self.scanned_file.type)

        if rescan:
            is_outdated: bool = self.scanned_file.is_outdated(manager)
            if not is_outdated:

                # note: this is needed, as video and track metadata is only written on new files, to migrate older files, it is ugly, but it is like this unfortunately
                if needs_migration_for_video_metadata:

                    def update_video_metadata_migration() -> None:
                        self.update_video_metadata(
                            manager,
                            callback,
                            error_mode,
                            only_update_file=False,
                        )

                    def generate_checksum_migration() -> None:
                        self.generate_checksum_if_needed(manager)

                    callback_workload_migration: list[CallbackWorkload] = [
                        update_video_metadata_migration,
                        generate_checksum_migration,
                    ]

                    callback.process_workload(
                        callback_workload_migration,
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                if Language.is_default_value(self.__language) or self._metadata is None:

                    def scan_language_outdated() -> None:
                        if Language.is_default_value(
                            self.__language,
                        ) and scanner.should_scan_language(ScanType.rescan):
                            language = scanner.language_scanner.get_language(
                                self.scanned_file,
                                language_picker,
                                error_mode=error_mode,
                                manager=manager,
                            )

                            if language is None:
                                self.__language = Language.get_default()
                            else:
                                self.__language = language
                                self.update_video_metadata(
                                    manager,
                                    callback,
                                    error_mode,
                                    only_update_file=True,
                                )

                    def scan_metadata_outdated() -> None:
                        if (
                            current_handles is not None
                            and not isinstance(current_handles, SkipHandle)
                            and scanner.should_scan_metadata(
                                ScanType.rescan,
                                self.metadata,
                            )
                        ):
                            series_handle, season_handle = current_handles

                            self._metadata = (
                                scanner.metadata_scanner.get_episode_metadata(
                                    series_handle,
                                    season_handle,
                                    self.description.episode,
                                )
                            )
                            self.update_video_metadata(
                                manager,
                                callback,
                                error_mode,
                                only_update_file=True,
                            )
                        else:
                            # don't need new metadata for changed files
                            pass

                    callback_workload_outdated: list[CallbackWorkload] = [
                        scan_language_outdated,
                        scan_metadata_outdated,
                    ]

                    callback.process_workload(
                        callback_workload_outdated,
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                return

        # we fall through, if we are outdated, this means some things are different to handle here in this block
        # i hate python for its indentation, so it's a bit confusing
        is_outdated = rescan

        if is_outdated:
            self.__reset_metadata_of_file()

        def update_video_metadata() -> None:
            self.update_video_metadata(
                manager,
                callback,
                error_mode,
                only_update_file=False,
            )

        def generate_checksum() -> None:
            self.generate_checksum_if_needed(manager)

        def scan_language() -> None:
            scan_type = ScanType.rescan if rescan else ScanType.first_scan
            if scanner.should_scan_language(scan_type=scan_type):
                language = scanner.language_scanner.get_language(
                    self.scanned_file,
                    language_picker,
                    error_mode=error_mode,
                    manager=manager,
                )

                if language is None:
                    self.__language = Language.get_default()
                else:
                    self.__language = language
                    self.update_video_metadata(
                        manager,
                        callback,
                        error_mode,
                        only_update_file=True,
                    )

        def scan_metadata() -> None:
            if (
                current_handles is not None
                and self.metadata is None
                and not isinstance(current_handles, SkipHandle)
                and scanner.should_scan_metadata(ScanType.first_scan, self.metadata)
            ):
                series_handle, season_handle = current_handles

                self._metadata = scanner.metadata_scanner.get_episode_metadata(
                    series_handle,
                    season_handle,
                    self.description.episode,
                )
                self.update_video_metadata(
                    manager,
                    callback,
                    error_mode,
                    only_update_file=True,
                )
            else:
                # don't need new metadata for changed files
                pass

        callback_workload: list[CallbackWorkload] = [
            update_video_metadata,
            generate_checksum,
            scan_language,
            scan_metadata,
        ]

        callback.process_workload(
            callback_workload,
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )
