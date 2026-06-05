from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
    override,
)

from apischema import alias, schema

from content.base_class import (
    CallbackData,
    Content,
    ContentCharacteristic,
)
from content.general import (
    Callback,
    CallbackWorkload,
    ContentType,
    NameParser,
    NumeratedDescription,
    ScannedFile,
)
from content.language import Language
from content.metadata.metadata import (
    HandlesType,
    MetadataHandle,
    SkipHandle,
    should_skip_metadata,
)
from content.shared import ScanType
from content.summary import Summary
from helper.apischema import narrow_type
from helper.log import get_logger

logger: Logger = get_logger()


@schema(extra=narrow_type(("type", Literal[ContentType.numerated])))
@dataclass(slots=True, repr=True)
class NumeratedContent(Content):
    __description: NumeratedDescription = field(metadata=alias("description"))
    __language: Language = field(metadata=alias("language"))

    @staticmethod
    def from_path(
        path: Path,
        scanned_file: ScannedFile,
        name_parser: NameParser,
    ) -> "NumeratedContent":
        description: Optional[NumeratedDescription] = (
            NumeratedContent.parse_description(
                path.name,
                name_parser,
            )
        )
        if description is None:
            msg = f"Couldn't get NumeratedDescription from '{path}'"
            raise NameError(msg, name="NumeratedDescription")

        return NumeratedContent(
            ContentType.numerated,
            scanned_file,
            None,
            description,
            Language.get_default(),
        )

    @property
    def description(self: Self) -> NumeratedDescription:
        return self.__description

    @property
    def language(self: Self) -> Language:
        return self.__language

    @staticmethod
    def is_valid_name(
        name: str,
        name_parser: NameParser,
    ) -> bool:
        return NumeratedContent.parse_description(name, name_parser) is not None

    @staticmethod
    def parse_description(
        name: str,
        name_parser: NameParser,
    ) -> Optional[NumeratedDescription]:
        # TODO
        # result = name_parser.parse_numerated_name(name)
        # if result is None:
        #    return None
        return None

        # name, season, episode, number = result

        # return NumeratedDescription(name, season, episode, number)

    @override
    def summary(self: Self, *, detailed: bool = False) -> Summary:
        return Summary(languages=[], metadatas=[], video_metadata=[], descriptions=[])
        # TODO
        # return Summary.construct_for_episode(
        #    self.__language,
        #    self.metadata,
        #    self.__description,
        #    detailed=detailed,
        # )

    def __get_handles(
        self: Self,
        handles: HandlesType,
    ) -> Optional[tuple[MetadataHandle, MetadataHandle] | SkipHandle]:
        if handles is None:
            return None

        if should_skip_metadata(handles):
            return SkipHandle()

        if len(handles) != 2:
            msg = f"Length of handles is invalid, expected 2 but got {len(handles)}"
            logger.warning(msg)
            return None

        return (handles[0], handles[1])

    def __reset_metadata_of_file(self: Self) -> None:
        self.__language = Language.get_default()
        # note, reset other metadata here, once new one is added

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
                if Language.is_default_value(self.__language) or self._metadata is None:

                    def scan_language_outdated() -> None:
                        if Language.is_default_value(
                            self.__language,
                        ) and scanner.should_scan_language(ScanType.rescan):
                            self.__language = (
                                scanner.language_scanner.get_language_or_default(
                                    self.scanned_file,
                                    language_picker,
                                    error_mode=error_mode,
                                    manager=manager,
                                )
                            )

                    def scan_metadata_outdated() -> None:
                        if (
                            current_handles is not None
                            and self.metadata is None
                            and not should_skip_metadata(current_handles)
                            and scanner.should_scan_metadata(
                                ScanType.rescan,
                                self.metadata,
                            )
                        ):
                            series_handle, season_handle = current_handles
                            self._metadata = SkipHandle()
                            """self._metadata = (
                                scanner.metadata_scanner.get_numerated_metadata(
                                    series_handle,
                                    season_handle,
                                    self.description.episode,
                                )
                            ) """

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

            self.__reset_metadata_of_file()

        def generate_checksum() -> None:
            self.generate_checksum(manager)

        def scan_language() -> None:
            if scanner.should_scan_language(ScanType.first_scan):
                self.__language = scanner.language_scanner.get_language_or_default(
                    self.scanned_file,
                    language_picker,
                    error_mode=error_mode,
                    manager=manager,
                )
            else:
                self.__reset_metadata_of_file()

        def scan_metadata() -> None:
            if (
                current_handles is not None
                and self.metadata is None
                and not should_skip_metadata(current_handles)
                and scanner.should_scan_metadata(ScanType.first_scan, self.metadata)
            ):
                series_handle, season_handle = current_handles
                self._metadata = SkipHandle()
                """ self._metadata = scanner.metadata_scanner.get_numerated_metadata(
                    series_handle,
                    season_handle,
                    self.description.episode,
                ) """
            else:
                # don't need new metadata for changed files
                pass

        callback_workload: list[CallbackWorkload] = [
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
