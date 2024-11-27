from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
    override,
)

from apischema import alias, schema

from classifier import Language
from content.base_class import (
    CallbackTuple,
    Content,
    ContentCharacteristic,
    ContentDict,
)
from content.general import (
    Callback,
    ContentType,
    EpisodeDescription,
    NameParser,
    ScannedFile,
    Summary,
    narrow_type,
)
from content.metadata.metadata import MetadataHandle
from content.shared import ScanKind, ScanType


class EpisodeContentDict(ContentDict):
    description: EpisodeDescription
    language: Language
    metadata: Optional[MetadataHandle]


# TODO: Remove this temporary helper
DUMMY_HANDLE_TODO: MetadataHandle = MetadataHandle("", {})


@schema(extra=narrow_type(("type", Literal[ContentType.episode])))
@dataclass(slots=True, repr=True)
class EpisodeContent(Content):
    __description: EpisodeDescription = field(metadata=alias("description"))
    __language: Language = field(metadata=alias("language"))
    __metadata: Optional[MetadataHandle] = field(
        metadata=alias("metadata"),
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
            msg = f"Couldn't get EpisodeDescription from '{path}'"
            raise NameError(msg, name="EpisodeDescription")

        return EpisodeContent(
            ContentType.episode,
            scanned_file,
            description,
            Language.unknown(),
            None,
        )

    @property
    def description(self: Self) -> EpisodeDescription:
        return self.__description

    @property
    def language(self: Self) -> Language:
        return self.__language

    @property
    def metadata(self: Self) -> Optional[MetadataHandle]:
        return self.__metadata

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

        name, season, episode = result

        return EpisodeDescription(name, season, episode)

    @override
    def summary(self: Self, *, detailed: bool = False) -> Summary:
        return Summary.from_single(
            self.__language,
            self.__description,
            detailed=detailed,
        )

    @override
    def scan(
        self: Self,
        callback: Callback[Content, ContentCharacteristic, CallbackTuple],
        *,
        parent_folders: list[str],
        rescan: bool = False,
    ) -> None:
        manager, scanner = callback.get_saved()

        # TODO: get this as input param or similar
        episode: int = 0

        characteristic: ContentCharacteristic = (self.type, self.scanned_file.type)

        if rescan:
            is_outdated: bool = self.scanned_file.is_outdated(manager)

            if not is_outdated:
                if self.__language == Language.unknown() or self.__metadata is None:
                    callback.start(
                        (2, 2, 0),
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                    if self.__language == Language.unknown() and scanner.should_scan(
                        self.__description,
                        ScanType.rescan,
                        ScanKind.language,
                    ):
                        self.__language = scanner.language_scanner.get_language(
                            self.scanned_file,
                            manager=manager,
                        )

                    callback.progress(
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                    if self.__metadata is None and scanner.should_scan(
                        self.__description,
                        ScanType.rescan,
                        ScanKind.metadata,
                    ):
                        self.__metadata = scanner.metadata_scanner.get_episode_metadata(
                            DUMMY_HANDLE_TODO,
                            DUMMY_HANDLE_TODO,
                            episode,
                        )

                    callback.progress(
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                    callback.finish(
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        0,
                        characteristic,
                    )

                return

        callback.start(
            (3, 3, 0),
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        self.generate_checksum(manager)
        callback.progress(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        if scanner.should_scan(
            self.__description,
            ScanType.first_scan,
            ScanKind.language,
        ):
            self.__language = scanner.language_scanner.get_language(
                self.scanned_file,
                manager=manager,
            )

        callback.progress(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        if scanner.should_scan(
            self.__description,
            ScanType.first_scan,
            ScanKind.metadata,
        ):
            self.__metadata = scanner.metadata_scanner.get_episode_metadata(
                DUMMY_HANDLE_TODO,
                DUMMY_HANDLE_TODO,
                episode,
            )

        callback.progress(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            characteristic,
        )

        callback.finish(
            self.scanned_file.path.name,
            self.scanned_file.parents,
            0,
            characteristic,
        )
