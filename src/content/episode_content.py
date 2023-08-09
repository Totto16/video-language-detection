from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Self,
)

from apischema import alias, schema
from classifier import Classifier, FileMetadataError, Language, WAVFile
from enlighten import Manager
from typing_extensions import override

from content.base_class import Content, ContentCharacteristic, ContentDict
from content.general import (
    Callback,
    ContentType,
    EpisodeDescription,
    NameParser,
    ScannedFile,
    Summary,
    narrow_type,
)


class EpisodeContentDict(ContentDict):
    description: EpisodeDescription
    language: Language


# TODO: remove
GLOBAL_ITER_MAX: int = 200
SKIP_ITR: int = 530
itr: int = 0


# TODO: remove
def itr_print_percent() -> None:
    global itr  # noqa: PLW0602
    if itr < SKIP_ITR:
        return

    if itr >= GLOBAL_ITER_MAX + SKIP_ITR:
        return

    percent: float = (itr - SKIP_ITR) / GLOBAL_ITER_MAX * 100.0

    print(f"{percent:.02f} %")


@schema(extra=narrow_type(("type", Literal[ContentType.episode])))
@dataclass(slots=True, repr=True)
class EpisodeContent(Content):
    __description: EpisodeDescription = field(metadata=alias("description"))
    __language: Language = field(metadata=alias("language"))

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
            raise NameError(msg)

        return EpisodeContent(
            ContentType.episode,
            scanned_file,
            description,
            Language.unknown(),
        )

    @property
    def description(self: Self) -> EpisodeDescription:
        return self.__description

    @property
    def language(self: Self) -> Language:
        return self.__language

    def __get_language(
        self: Self,
        classifier: Classifier,
        manager: Optional[Manager] = None,
    ) -> Language:
        try:
            wav_file = WAVFile(self.scanned_file.path)

            best, _ = classifier.predict(
                wav_file,
                self.scanned_file.path,
                manager,
            )
        except FileMetadataError as err:
            print(err)
            return Language.unknown()
        else:
            # python is funky xD, leaking variables a s desired pattern xD
            return best.language

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
        callback: Callback[Content, ContentCharacteristic, Manager],
        name_parser: NameParser,
        *,
        parent_folders: list[str],
        classifier: Classifier,
        rescan: bool = False,
    ) -> None:
        manager: Manager = callback.get_saved()

        characteristic: ContentCharacteristic = (self.type, self.scanned_file.type)

        if rescan:
            is_outdated: bool = self.scanned_file.is_outdated(manager)

            if not is_outdated:
                if self.__language == Language.unknown():
                    callback.start(
                        (1, 1, 0),
                        self.scanned_file.path.name,
                        self.scanned_file.parents,
                        characteristic,
                    )

                    # TODO: remove
                    global itr  # noqa: PLW0603
                    if itr < GLOBAL_ITER_MAX + SKIP_ITR:
                        itr_print_percent()
                        itr = itr + 1
                        if itr >= SKIP_ITR:
                            self.__language = self.__get_language(classifier, manager)

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
            (2, 2, 0),
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

        # TODO: re-enable
        if itr < GLOBAL_ITER_MAX + SKIP_ITR:
            itr_print_percent()
            itr = itr + 1
            if itr >= SKIP_ITR:
                self.__language = self.__get_language(classifier, manager)

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
