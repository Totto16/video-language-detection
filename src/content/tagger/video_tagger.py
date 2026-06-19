from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Self, assert_never, cast, override
from uuid import UUID

from content.language import Language
from helper.decorator import decorate_class
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

@decorate_class(slots=True)
class VideoTaggerContextInterface(ABC):
    __manager: ManagerInterface

    def __init__(
        self: Self,
        manager: ManagerInterface,
    ) -> None:
        super().__init__()
        self.__manager = manager

    @property
    def manager(self: Self) -> ManagerInterface:
        return self.__manager

@decorate_class(slots=True)
class VideoTaggerContextReadable(VideoTaggerContextInterface):
    @abstractmethod
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead: ...

@decorate_class(slots=True)
class VideoTaggerContextWriteable(VideoTaggerContextInterface):
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


class VideoTaggerContextRW(VideoTaggerContextReadable, VideoTaggerContextWriteable):
    pass

@decorate_class(slots=True)
class VideoTagger(ABC):
    __file: Path

    def __init__(self: Self, file: Path) -> None:
        super().__init__()
        self.__file = file

    @abstractmethod
    def r_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextReadable]: ...

    @abstractmethod
    def w_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextWriteable]: ...

    @abstractmethod
    def rw_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextRW]: ...

    @property
    def file(self: Self) -> Path:
        return self.__file


ContextType = Literal["r", "w", "rw"]

@decorate_class(slots=True)
class VideoTaggerContextWrapperGeneric(VideoTaggerContextRW):
    __impl: VideoTaggerContextRW
    __ctx: ContextType

    def __init__(
        self: Self,
        manager: ManagerInterface,
        impl: VideoTaggerContextRW,
        ctx: ContextType,
    ) -> None:
        super().__init__(manager)
        self.__impl = impl
        self.__ctx = ctx

    @override
    def write_tags(self: Self, tags: MetadataTags) -> None:
        if self.__ctx not in ["w", "rw"]:
            msg = f"Invalid context: can't write with the type '{self.__ctx}'"
            raise RuntimeError(msg)

        return self.__impl.write_tags(tags)

    @override
    def write_language(
        self: Self,
        language: Language,
    ) -> bool:
        if self.__ctx not in ["w", "rw"]:
            msg = f"Invalid context: can't write with the type '{self.__ctx}'"
            raise RuntimeError(msg)

        return self.__impl.write_language(language)

    @override
    def get_tags(self: Self) -> MetadataTagsRead:
        if self.__ctx not in ["r", "rw"]:
            msg = f"Invalid context: can't read with the type '{self.__ctx}'"
            raise RuntimeError(msg)

        return self.__impl.get_tags()

@decorate_class(slots=True)
class VideoTaggerContextMultipleRW(VideoTaggerContextRW):
    __contexts: list[AbstractContextManager[VideoTaggerContextRW]]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        contexts: list[AbstractContextManager[VideoTaggerContextRW]],
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

@decorate_class(slots=True)
class VideoTaggerMultiple(VideoTagger):
    __tagger: list[VideoTagger]

    def __init__(self: Self, file: Path, tagger: list[VideoTagger]) -> None:
        super().__init__(file)
        self.__tagger = tagger

    def __context_impl(
        self: Self,
        manager: ManagerInterface,
        ctx: ContextType,
    ) -> AbstractContextManager[VideoTaggerContextRW]:

        def get_context(
            tgr: VideoTagger,
        ) -> AbstractContextManager[VideoTaggerContextInterface]:
            match ctx:
                case "r":
                    return tgr.r_ctx(manager)
                case "w":
                    return tgr.w_ctx(manager)
                case "rw":
                    return tgr.rw_ctx(manager)
                case _:
                    assert_never(ctx)

        contexts = [get_context(tagger) for tagger in self.__tagger]

        @decorate_class(slots=True)
        class VideoTaggerContextCtx(AbstractContextManager[VideoTaggerContextRW]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> VideoTaggerContextRW:
                return VideoTaggerContextWrapperGeneric(
                    manager,
                    VideoTaggerContextMultipleRW(
                        manager,
                        cast(
                            list[AbstractContextManager[VideoTaggerContextRW]], contexts,
                        ),
                    ),
                    ctx,
                )

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                return False

        return VideoTaggerContextCtx()

    @override
    def r_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextReadable]:
        return self.__context_impl(manager, "r")

    @override
    def w_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextWriteable]:
        return self.__context_impl(manager, "w")

    @override
    def rw_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextRW]:
        return self.__context_impl(manager, "rw")


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
