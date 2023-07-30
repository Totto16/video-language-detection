#!/usr/bin/env python3


from json import JSONEncoder
from pathlib import Path
from typing import (
    Any,
    Optional,
    Self,
)

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


class EpisodeContent(Content):
    __description: EpisodeDescription
    __language: Language

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
            raise NameError(f"Couldn't get EpisodeDescription from '{path}'")

        return EpisodeContent(scanned_file, description)

    def __init__(
        self: Self,
        scanned_file: ScannedFile,
        description: EpisodeDescription,
        language: Language = Language.unknown(),  # noqa: B008
    ) -> None:
        super().__init__(ContentType.episode, scanned_file)

        self.__description = description
        self.__language = language

    @property
    def description(self: Self) -> EpisodeDescription:
        return self.__description

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
            return best.language
        except FileMetadataError:
            return Language.unknown()

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
    def summary(self: Self, detailed: bool = False) -> Summary:
        return Summary.from_single(self.__language, self.__description, detailed)

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
            characteristic,
        )

    @override
    def as_dict(
        self: Self,
        json_encoder: Optional[JSONEncoder] = None,
    ) -> dict[str, Any]:
        def encode(x: Any) -> Any:
            return x if json_encoder is None else json_encoder.default(x)

        as_dict: dict[str, Any] = super().as_dict(json_encoder)
        as_dict["description"] = encode(self.__description)
        as_dict["language"] = encode(self.__language)
        return as_dict

    @staticmethod
    def from_dict(dct: EpisodeContentDict) -> "EpisodeContent":
        return EpisodeContent(dct["scanned_file"], dct["description"], dct["language"])

    def __str__(self: Self) -> str:
        return str(self.as_dict())

    def __repr__(self: Self) -> str:
        return self.__str__()
