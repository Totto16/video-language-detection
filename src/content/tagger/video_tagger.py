from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    BinaryIO,
    Literal,
    Optional,
    Self,
    assert_never,
    cast,
    final,
    override,
)
from uuid import UUID

from content.language import Language
from helper.decorator import decorate_class
from helper.ffprobe import FFProbeResult, ffprobe
from helper.log import get_logger
from helper.manager import ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

logger: Logger = get_logger()
_ = get_translator()


VIDEO_FILE_TAG_UPDATE_BAR_FORMAT: str = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


SerializableDictValue = str | int | dict[str, str | int] | dict[str, Any]
SerializableDict = dict[str, SerializableDictValue]


@dataclass(slots=True, repr=True)
class MetadataTags:
    comment: str
    uuid: UUID
    metadata: SerializableDict


@dataclass(slots=True, repr=True)
class MetadataTagsRead:
    comment: Optional[str]
    uuid: Optional[UUID]
    metadata: SerializableDict
    unrecognized: list[tuple[str, str]]


@decorate_class(slots=True)
class VideoTaggerContextInterface(ABC):
    __manager: ManagerInterface
    __file: Path

    def __init__(self: Self, manager: ManagerInterface, file: Path) -> None:
        super().__init__()
        self.__manager = manager
        self.__file = file

    @property
    def manager(self: Self) -> ManagerInterface:
        return self.__manager

    @property
    def file(self: Self) -> Path:
        return self.__file


@decorate_class(slots=True)
class VideoTaggerContextReadable(VideoTaggerContextInterface):
    @abstractmethod
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead: ...

    @abstractmethod
    def read_language(
        self: Self,
    ) -> Result[Optional[Language], str]: ...


@decorate_class(slots=True)
class RestoreFileNotSupported:
    pass


def is_the_same_file(
    pre_res: FFProbeResult,
    after_res: FFProbeResult,
) -> Result[None, str]:
    try:
        if len(pre_res.streams) != len(after_res.streams):
            return Err(
                f"Number of streams differs: {len(pre_res.streams)} != {len(after_res.streams)}",
            )

        if pre_res.file_info.duration() != after_res.file_info.duration():
            return Err(
                f"Duration differs: {pre_res.file_info.duration()} != {after_res.file_info.duration()}",
            )

        if (
            pre_res.file_info.raw["format_name"]
            != after_res.file_info.raw["format_name"]
        ):
            return Err(
                f"Format name differs: {pre_res.file_info.raw["format_name"] } != {after_res.file_info.raw["format_name"] }",
            )

        return Ok(None)
    except (KeyError, RuntimeError, ValueError, TypeError) as err:
        return Err(f"Excpetion occurred: {err!s}")


@decorate_class(slots=True)
class VideoTaggerContextWriteable(VideoTaggerContextInterface):
    @abstractmethod
    def write_tags(
        self: Self,
        tags: MetadataTags,
    ) -> None: ...

    @final
    def write_tags_safe(self: Self, tags: MetadataTags) -> Result[None, str]:
        try:
            pre_write_res = ffprobe(self.file)
            if pre_write_res.err():
                return Err(f"FFprobe err: {pre_write_res.as_err()}")

            pre_write = pre_write_res.as_ok()

            self.write_tags(tags)

            after_write_res = ffprobe(self.file)
            if after_write_res.err():
                msg = f"FFprobe err: {after_write_res.as_err()}"
                raise RuntimeError(msg)  # noqa: TRY301

            after_write = after_write_res.as_ok()

            same_res = is_the_same_file(pre_write, after_write)

            if same_res.err():
                msg = f"FFProbe detected differences: {same_res.as_err()}"
                raise RuntimeError(msg)  # noqa: TRY301

            return Ok(None)
        except Exception as err:  # noqa: BLE001
            restore_result = self.restore_file()
            if isinstance(restore_result, RestoreFileNotSupported):
                return Err(f"Can't restore file, original error: {err!s}")

            if restore_result.err():
                return Err(
                    f"Restore file error: {restore_result.as_err()}, original error: {err!s}",
                )

            return Err(str(err))

    @abstractmethod
    def restore_file(
        self: Self,
    ) -> RestoreFileNotSupported | Result[None, str]: ...

    @abstractmethod
    def write_language(
        self: Self,
        language: Language,
    ) -> bool: ...


class VideoTaggerContextRW(VideoTaggerContextReadable, VideoTaggerContextWriteable):
    pass


@decorate_class(slots=True)
class InspectNotImplemented:
    pass


class InspectPriority(Enum):
    Important = "important"
    Normal = "normal"
    Ignore = "ignore"

    def as_int(self: Self) -> int:
        match self:
            case InspectPriority.Important:
                return 0
            case InspectPriority.Normal:
                return 1
            case InspectPriority.Ignore:
                return 2
            case _:
                assert_never(self)

    @staticmethod
    def from_str(inp: str) -> Optional["InspectPriority"]:
        for level in InspectPriority:
            if str(level).lower() == inp.lower():
                return level

        return None


@dataclass(slots=True, repr=True)
class InspectElement:
    name: str
    size: int


@decorate_class(slots=True)
class InspectPrinter(ABC):
    def __init__(self: Self) -> None:
        super().__init__()

    @abstractmethod
    def element(
        self: Self,
        element: InspectElement,
        depth: int,
    ) -> None: ...

    @abstractmethod
    def start(
        self: Self,
    ) -> None: ...

    @abstractmethod
    def end(
        self: Self,
    ) -> None: ...

    @abstractmethod
    def skip(
        self: Self,
        parent: str,
        amount: int,
        depth: int,
    ) -> None: ...


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

    @abstractmethod
    def inspect(
        self: Self,
        printer: InspectPrinter,
        priority: InspectPriority,
    ) -> Optional[InspectNotImplemented]: ...


ContextType = Literal["r", "w", "rw"]


@decorate_class(slots=True)
class VideoTaggerContextWrapperGeneric(VideoTaggerContextRW):
    __impl: VideoTaggerContextRW
    __ctx: ContextType
    __restore_backup_fn: Callable[[], RestoreFileNotSupported | Result[None, str]]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        impl: VideoTaggerContextRW,
        ctx: ContextType,
        file: Path,
        restore_backup_fn: Callable[[], RestoreFileNotSupported | Result[None, str]],
    ) -> None:
        super().__init__(manager=manager, file=file)
        self.__impl = impl
        self.__ctx = ctx
        self.__restore_backup_fn = restore_backup_fn

    @override
    def write_tags(self: Self, tags: MetadataTags) -> None:
        if self.__ctx not in ["w", "rw"]:
            msg = f"Invalid context: can't write with the type '{self.__ctx}'"
            raise RuntimeError(msg)

        return self.__impl.write_tags(tags)

    @override
    def restore_file(
        self: Self,
    ) -> RestoreFileNotSupported | Result[None, str]:
        res = self.__impl.restore_file()

        if isinstance(res, RestoreFileNotSupported):
            return self.__restore_backup_fn()

        return res

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

    @override
    def read_language(
        self: Self,
    ) -> Result[Optional[Language], str]:
        if self.__ctx not in ["r", "rw"]:
            msg = f"Invalid context: can't read with the type '{self.__ctx}'"
            raise RuntimeError(msg)

        return self.__impl.read_language()


@decorate_class(slots=True)
class VideoTaggerContextCtxGeneric(AbstractContextManager[VideoTaggerContextRW]):
    __writer: Optional[BinaryIO]
    __backup: Optional[bytes]
    __file: Path
    __ctx: ContextType
    __manager: ManagerInterface

    def __init__(
        self: Self,
        file: Path,
        ctx: ContextType,
        manager: ManagerInterface,
    ) -> None:
        super().__init__()
        self.__writer = None
        self.__file = file
        self.__backup = None
        self.__ctx = ctx
        self.__manager = manager

    @abstractmethod
    def get_context(
        self: Self,
        manager: ManagerInterface,
        writer: BinaryIO,
    ) -> VideoTaggerContextRW: ...

    def __restore_backup_impl(self: Self) -> Result[None, str]:
        if self.__backup is None:
            return Err("Backup for file not present")

        # restore file backup
        if self.__ctx != "r":
            restore_writer = self.__file.open("rb+")
            restore_writer.truncate()
            restore_writer.write(self.__backup)
            restore_writer.close()
            print(f"RESTORED BACKUP FOR FILE: '{self.__file}'")  # noqa: T201

        self.__backup = None
        return Ok(None)

    @final
    def restore_backup(self: Self) -> None:
        res = self.__restore_backup_impl()

        if res.err():
            raise RuntimeError(res.as_err())

    @final
    @override
    def __enter__(self: Self) -> VideoTaggerContextRW:
        writer = self.__file.open(mode="rb" if self.__ctx == "r" else "rb+")

        writer.seek(0, 2)
        filesize = writer.tell()
        writer.seek(0)

        backup = writer.read(-1)

        writer.seek(0)

        if len(backup) != filesize:
            writer.close()
            msg = f"Error: reading file bytes for backup failed. didn't get enough bytes: {len(backup)} != {filesize}"
            raise RuntimeError(msg)

        self.__writer = writer
        self.__backup = backup

        return VideoTaggerContextWrapperGeneric(
            self.__manager,
            self.get_context(
                self.__manager,
                self.__writer,
            ),
            self.__ctx,
            self.__file,
            self.__restore_backup_impl,
        )

    @final
    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        if self.__writer is not None:
            self.__writer.close()
            self.__writer = None

        if exc_val is not None:
            self.restore_backup()

        if self.__backup is not None:
            self.__backup = None

        return False


@decorate_class(slots=True)
class VideoTaggerContextMultipleRW(VideoTaggerContextRW):
    __contexts: list[AbstractContextManager[VideoTaggerContextRW]]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        file: Path,
        contexts: list[AbstractContextManager[VideoTaggerContextRW]],
    ) -> None:
        super().__init__(manager, file=file)
        self.__contexts = contexts

    @override
    def write_tags(self: Self, tags: MetadataTags) -> None:
        for context in self.__contexts:
            with context as ctx:
                ctx.write_tags(tags)

    @override
    def restore_file(
        self: Self,
    ) -> RestoreFileNotSupported | Result[None, str]:
        for context in self.__contexts:
            with context as ctx:
                res = ctx.restore_file()
                if isinstance(res, RestoreFileNotSupported):
                    continue

                if res.ok():
                    return res

        return RestoreFileNotSupported()

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

    @override
    def read_language(
        self: Self,
    ) -> Result[Optional[Language], str]:
        return Err("Merging the languages is not implemented yet!")


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

        file = self.file

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
                        file,
                        cast(
                            list[AbstractContextManager[VideoTaggerContextRW]],
                            contexts,
                        ),
                    ),
                    ctx,
                    file,
                    RestoreFileNotSupported,
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

    @override
    def inspect(
        self: Self,
        printer: InspectPrinter,
        priority: InspectPriority,
    ) -> Optional[InspectNotImplemented]:
        return InspectNotImplemented()


TAGGER_DOMAIN = "lt.totto.vld"


@dataclass(slots=True, repr=True)
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
