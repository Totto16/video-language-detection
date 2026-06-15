from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Self, override
from uuid import UUID

from content.language import Language
from helper.log import get_logger
from helper.manager import ManagerInterface
from helper.translation import get_translator

logger: Logger = get_logger()
_ = get_translator()


VIDEO_FILE_TAG_UPDATE_BAR_FORMAT: str = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


SerializableDictValue = str | int | dict[str, str | int] | dict[str, Any]
SerializableDict = dict[str, SerializableDictValue]


@dataclass
class MetadataTags:
    comment: str
    uuid: UUID
    metadata: SerializableDict


@dataclass
class MetadataTagsRead:
    comment: Optional[str]
    uuid: Optional[UUID]
    metadata: SerializableDict
    unrecognized: list[tuple[str, str]]


class VideoTaggerContext(ABC):
    __manager: ManagerInterface

    def __init__(
        self: Self,
        manager: ManagerInterface,
    ) -> None:
        super().__init__()
        self.__manager = manager

    @abstractmethod
    def write_tags(
        self: Self,
        tags: MetadataTags,
    ) -> None: ...

    @abstractmethod
    def write_language(
        self: Self,
        language: Language,
    ) -> bool: ...

    @abstractmethod
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead: ...

    @property
    def manager(self: Self) -> ManagerInterface:
        return self.__manager


class VideoTagger(ABC):
    __file: Path

    def __init__(self: Self, file: Path) -> None:
        super().__init__()
        self.__file = file

    @abstractmethod
    def context(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContext]: ...

    @property
    def file(self: Self) -> Path:
        return self.__file


class VideoTaggerContextMultiple(VideoTaggerContext):
    __contexts: list[AbstractContextManager[VideoTaggerContext]]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        contexts: list[AbstractContextManager[VideoTaggerContext]],
    ) -> None:
        super().__init__(manager)
        self.__contexts = contexts

    @override
    def write_tags(self: Self, tags: MetadataTags) -> None:
        for context in self.__contexts:
            with context as ctx:
                ctx.write_tags(tags)

    @override
    def write_language(
        self: Self,
        language: Language,
    ) -> bool:
        for context in self.__contexts:
            with context as ctx:
                result = ctx.write_language(language)
                if result:
                    return result

        return False

    @override
    def get_tags(self: Self) -> MetadataTagsRead:
        msg = "Merging the tags is not implemented yet!"
        raise NotImplementedError(msg)


class VideoTaggerMultiple(VideoTagger):
    __tagger: list[VideoTagger]

    def __init__(self: Self, file: Path, tagger: list[VideoTagger]) -> None:
        super().__init__(file)
        self.__tagger = tagger

    @override
    def context(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContext]:
        contexts = [tagger.context(manager) for tagger in self.__tagger]

        class VideoTaggerContextCtx(AbstractContextManager[VideoTaggerContext]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> VideoTaggerContext:
                return VideoTaggerContextMultiple(manager, contexts)

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                return False

        return VideoTaggerContextCtx()


TAGGER_DOMAIN = "lt.totto.vld"


@dataclass
class AppleItunesFreeformKey:
    mean: str
    name: str


class TaggerDomain:
    @staticmethod
    def get(key: str) -> str:
        return f"----:{TAGGER_DOMAIN}:video_language_detect:{key}"

    @staticmethod
    def get_freeform(key: str) -> AppleItunesFreeformKey:
        return AppleItunesFreeformKey(
            mean=TAGGER_DOMAIN,
            name=f"video_language_detect:{key}",
        )

    @staticmethod
    def key_start_with(key: str) -> bool:
        return key.startswith(f"----:{TAGGER_DOMAIN}:video_language_detect:")

    @staticmethod
    def get_raw(key: str) -> str:
        return key.replace(f"----:{TAGGER_DOMAIN}:video_language_detect:", "")

    @staticmethod
    def get_raw_name(key: str) -> str:
        return key.replace("video_language_detect:", "")

    UUID_RAW_KEY: str = f"----:{TAGGER_DOMAIN}:video_language_detect_uuid:raw"

    UUID_HEX_KEY: str = f"----:{TAGGER_DOMAIN}:video_language_detect_uuid:hex"

    UUID_RAW_KEY_FREEFORM: AppleItunesFreeformKey = AppleItunesFreeformKey(
        mean=TAGGER_DOMAIN,
        name="video_language_detect_uuid:raw",
    )

    UUID_HEX_KEY_FREEFORM: AppleItunesFreeformKey = AppleItunesFreeformKey(
        mean=TAGGER_DOMAIN,
        name="video_language_detect_uuid:hex",
    )


def uuid_from_str(value: str) -> UUID:
    return UUID(hex=value)


def uuid_to_str(value: UUID) -> str:
    return value.hex
