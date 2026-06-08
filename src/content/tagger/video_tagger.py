from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
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


SerializableDict = dict[str, str | int | dict[str, str | int] | dict[str, Any]]


class VideoTaggerWriter(ABC):
    __manager: ManagerInterface

    def __init__(
        self: Self,
        manager: ManagerInterface,
    ) -> None:
        super().__init__()
        self.__manager = manager

    @abstractmethod
    def write_metadata(
        self: Self,
        comment: str,
        uuid: UUID,
        language: Language,
        metadata: SerializableDict,
    ) -> None: ...

    @property
    def manager(self: Self) -> ManagerInterface:
        return self.__manager


class VideoTagger(ABC):
    __file: Path

    def __init__(self: Self, file: Path) -> None:
        super().__init__()
        self.__file = file

    @abstractmethod
    def writer(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerWriter]: ...

    @property
    def file(self: Self) -> Path:
        return self.__file


class VideoTaggerWriterMultiple(VideoTaggerWriter):
    __writer: list[AbstractContextManager[VideoTaggerWriter]]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        writer: list[AbstractContextManager[VideoTaggerWriter]],
    ) -> None:
        super().__init__(manager)
        self.__writer = writer

    @override
    def write_metadata(
        self: Self,
        comment: str,
        uuid: UUID,
        language: Language,
        metadata: SerializableDict,
    ) -> None:
        for writer in self.__writer:
            with writer as w:
                w.write_metadata(comment, uuid, language, metadata)


class VideoTaggerMultiple(VideoTagger):
    __tagger: list[VideoTagger]

    def __init__(self: Self, file: Path, tagger: list[VideoTagger]) -> None:
        super().__init__(file)
        self.__tagger = tagger

    @override
    def writer(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerWriter]:
        writer = [tagger.writer(manager) for tagger in self.__tagger]

        class VideoTaggerWriterCtx(AbstractContextManager[VideoTaggerWriter]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> VideoTaggerWriter:
                return VideoTaggerWriterMultiple(manager, writer)

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                return False

        return VideoTaggerWriterCtx()
