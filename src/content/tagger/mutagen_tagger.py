import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Optional, Self, assert_never, override
from uuid import UUID

import mutagen._file as mutagen
from mutagen import mp4
from mutagen._util import MutagenError

from content.language import Language
from content.tagger.parser import ISOM_BYTE_ORDER, uuid_from_bytes, uuid_to_bytes
from content.tagger.video_tagger import (
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    ContextType,
    MetadataTags,
    MetadataTagsRead,
    SerializableDict,
    TaggerDomain,
    VideoTagger,
    VideoTaggerContextReadable,
    VideoTaggerContextRW,
    VideoTaggerContextWrapperGeneric,
    VideoTaggerContextWriteable,
    uuid_from_str,
    uuid_to_str,
)
from helper.decorator import decorate_class
from helper.manager import PROGRESS_CHUNK_SIZE, CounterInterface, ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

_ = get_translator()


## see https://mutagen.readthedocs.io/en/latest/user/filelike.html#
class IOInterface(ABC):
    def __init__(
        self: Self,
    ) -> None:
        super().__init__()

    """This is the interface mutagen expects from custom file-like
    objects.

    For loading read(), tell() and seek() have to be implemented. "name"
    is optional.

    For saving/deleting write(), flush() and truncate() have to be
    implemented in addition. fileno() is optional.
    """

    # For loading
    @abstractmethod
    def tell(self: Self) -> int:
        """Returns he current offset as int. Always >= 0.

        Raises IOError in case fetching the position is for some reason
        not possible.
        """

        raise NotImplementedError

    @abstractmethod
    def read(self: Self, size: int = -1) -> bytes:
        """Returns 'size' amount of bytes or less if there is no more data.
        If no size is given all data is returned. size can be >= 0.

        Raises IOError in case reading failed while data was available.
        """

        raise NotImplementedError

    @abstractmethod
    def seek(self: Self, offset: int, whence: int = 0) -> int:
        """Move to a new offset either relative or absolute. whence=0 is
        absolute, whence=1 is relative, whence=2 is relative to the end.

        Any relative or absolute seek operation which would result in a
        negative position is undefined and that case can be ignored
        in the implementation.

        Any seek operation which moves the position after the stream
        should succeed. tell() should report that position and read()
        should return an empty bytes object.

        Returns Nothing.
        Raise IOError in case the seek operation asn't possible.
        """

        raise NotImplementedError

    # For loading, but optional

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Should return text. For example the file name.

        If not available the attribute can be missing or can return
        an empty string.

        Will be used for error messages and type detection.
        """

        raise NotImplementedError

    # For writing

    @abstractmethod
    def write(self: Self, data: bytes) -> int:
        """Write data to the file.

        Returns Nothing.
        Raises IOError
        """

        raise NotImplementedError

    @abstractmethod
    def truncate(self: Self, size: Optional[int] = None) -> int:
        """Truncate to the current position or size if size is given.

        The current position or given size will never be larger than the
        file size.

        This has to flush write buffers in case writing is buffered.

        Returns Nothing.
        Raises IOError.
        """

        raise NotImplementedError

    @abstractmethod
    def flush(self: Self) -> None:
        """Flush the write buffer.

        Returns Nothing.
        Raises IOError.
        """

        raise NotImplementedError

    # For writing, but optional

    @abstractmethod
    def fileno(self: Self) -> int:
        """Returns the file descriptor (int) or raises IOError
        if there is none.

        Will be used for low level operations if available.
        """

        raise NotImplementedError


@dataclass(slots=True, repr=True)
class IOOpProgress:
    type: Literal["progress"]
    which: Literal["read", "write"]
    amount: int


@dataclass(slots=True, repr=True)
class IOOpSeek:
    type: Literal["seek"]
    amount: int


@dataclass(slots=True, repr=True)
class IOOpTruncate:
    type: Literal["truncate"]
    amount: int


IOOp = IOOpProgress | IOOpSeek | IOOpTruncate

OpCallback = Callable[[IOOp], None]

@decorate_class(slots=True)
class MutagenFileWrapper(IOInterface):
    __file: Path
    __impl: BinaryIO
    __callbacks: list[OpCallback]
    __chunk_size: int

    def __init__(
        self: Self,
        file: Path,
        chunk_size: int,
        *,
        ctx: ContextType,
    ) -> None:
        super().__init__()

        self.__file = file
        self.__impl = self.__file.open(mode="rb" if ctx == "r" else "rb+")
        self.__callbacks = []
        self.__chunk_size = chunk_size

    def __emit(self: Self, op: IOOp) -> None:
        for callback in self.__callbacks:
            callback(op)

    @override
    def tell(self: Self) -> int:
        return self.__impl.tell()

    @override
    def read(self: Self, size: int = -1) -> bytes:

        chunk_size = self.__chunk_size

        if chunk_size < 0:
            data = self.__impl.read(size)
            self.__emit(IOOpProgress("progress", "read", amount=len(data)))
            return data

        if size < 0:
            chunks_infinite: list[bytes] = []
            total_infinite: int = 0

            while True:
                chunk = self.__impl.read(chunk_size)
                if not chunk:
                    break

                chunks_infinite.append(chunk)

                chunk_len = len(chunk)
                total_infinite += chunk_len
                self.__emit(IOOpProgress("progress", "read", amount=chunk_len))

            return b"".join(chunks_infinite)

        chunks_limited: list[bytes] = []
        total_limited: int = 0
        remaining: int = size

        while remaining > 0:
            chunk = self.__impl.read(min(chunk_size, remaining))
            if not chunk:
                break

            chunks_limited.append(chunk)

            chunk_len = len(chunk)
            total_limited += chunk_len
            remaining -= chunk_len
            self.__emit(IOOpProgress("progress", "read", amount=chunk_len))

        return b"".join(chunks_limited)

    @override
    def seek(self: Self, offset: int, whence: int = 0) -> int:
        result = self.__impl.seek(offset, whence)
        self.__emit(IOOpSeek("seek", result))
        return result

    @override
    @property
    def name(self: Self) -> str:
        return self.__file.name

    @override
    def write(self: Self, data: bytes) -> int:

        chunk_size = self.__chunk_size

        if chunk_size < 0:
            result = self.__impl.write(data)
            self.__emit(IOOpProgress("progress", "write", amount=result))
            return result

        total: int = 0

        for offset in range(0, len(data), chunk_size):
            chunk = data[offset : offset + chunk_size]

            written = self.__impl.write(chunk)

            total += written
            self.__emit(IOOpProgress("progress", "write", amount=written))

        return total

    @override
    def truncate(self: Self, size: Optional[int] = None) -> int:
        size = self.__impl.truncate(size)
        self.__emit(IOOpTruncate("truncate", amount=size))
        return size

    @override
    def flush(self: Self) -> None:
        return self.__impl.flush()

    def fileno(self: Self) -> int:
        return self.__impl.fileno()

    def size(self: Self) -> int:
        return os.fstat(self.fileno()).st_size

    def close(self: Self) -> None:
        if not self.__impl.closed:
            self.__impl.close()

    def __del__(self: Self) -> None:
        self.close()

    def callback_ctx(self: Self, callback: OpCallback) -> AbstractContextManager[None]:

        def add_cb() -> None:
            self.__callbacks.append(callback)

        def remove_cb() -> None:
            self.__callbacks.remove(callback)

        @decorate_class(slots=True)
        class CallbackCtx(AbstractContextManager[None]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> None:
                add_cb()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                remove_cb()
                return False

        return CallbackCtx()

@decorate_class(slots=True)
class VideoTaggerContextMutagen(VideoTaggerContextRW):
    __filething: MutagenFileWrapper
    __instance: mutagen.FileType

    def __init__(
        self: Self,
        filething: MutagenFileWrapper,
        instance: mutagen.FileType,
        manager: ManagerInterface,
    ) -> None:
        super().__init__(manager)
        self.__instance = instance
        self.__filething = filething

    def __save_impl(self: Self) -> None:
        total: int = self.__filething.size()

        bar: CounterInterface = self.manager.counter(
            total=float(total),
            desc="update video tags",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        position: int = self.__filething.tell()

        bar.update(float(position))
        bar_position: int = position

        try:

            def process_op(op: IOOp) -> None:
                # NOTE. the progress bar can only go forwards, so the position tracker keeps track of the position and we update the peogress bar only, when we move forwards

                nonlocal position
                nonlocal bar_position

                match op.type:
                    case "progress":
                        position += op.amount
                    case "seek":
                        position = op.amount
                    case "truncate":
                        # only the total changed, which we can't adjust
                        pass
                    case _:
                        assert_never(op.type)

                if position > bar_position:
                    if position <= total:
                        bar.update(incr=float(position - bar_position))
                        bar_position = position
                    else:
                        bar.update(0, force=True)
                else:
                    bar.update(0, force=True)

            with self.__filething.callback_ctx(process_op):
                self.__instance.save(self.__filething)
                self.__filething.flush()

        except MutagenError as err:
            msg = "tag error"
            raise RuntimeError(msg) from err
        finally:
            bar.close(clear=True)

    @override
    def write_tags(self: Self, tags: MetadataTags) -> None:
        self.__instance["\xa9cmt"] = [tags.comment]
        if isinstance(self.__instance, mp4.MP4):

            for key, value in tags.metadata.items():
                value_enc = json.dumps(value).encode()
                self.__instance[TaggerDomain.get(key)] = [
                    mp4.MP4FreeForm(value_enc, mp4.AtomDataType.UTF8),
                ]

            previous_uuid_raw = self.__instance.get(TaggerDomain.UUID_RAW_KEY)

            if not previous_uuid_raw:
                self.__instance[TaggerDomain.UUID_RAW_KEY] = [
                    mp4.MP4FreeForm(
                        uuid_to_bytes(ISOM_BYTE_ORDER, tags.uuid),
                        mp4.AtomDataType.UUID,
                    ),
                ]

            previous_uuid_hex = self.__instance.get(TaggerDomain.UUID_HEX_KEY)

            if not previous_uuid_hex:
                self.__instance[TaggerDomain.UUID_HEX_KEY] = [
                    mp4.MP4FreeForm(
                        uuid_to_str(tags.uuid).encode(),
                        mp4.AtomDataType.UTF8,
                    ),
                ]

        else:
            msg = _(
                "Unrecognized mutagen instance, this is an implementation error: {clazz}"  # noqa: COM812
            ).format(clazz=type(self.__instance))
            raise TypeError(msg)

        self.__save_impl()

    @override
    def write_language(
        self: Self,
        language: Language,
    ) -> bool:
        return False

    @override
    def get_tags(  # noqa: PLR0915
        self: Self,
    ) -> MetadataTagsRead:

        def decode_mutagen_tag_value[A](
            value: list[str] | list[mp4.MP4FreeForm] | str | mp4.MP4FreeForm | Any,
            cb: Callable[[str | mp4.MP4FreeForm], A],
        ) -> A:
            if isinstance(value, list):
                if len(value) != 1:
                    msg = f"Invalid amount of tags: {len(value)}"
                    raise RuntimeError(msg)
                return cb(value[0])

            if isinstance(value, (str, mp4.MP4FreeForm)):
                return cb(value)

            msg = f"Invalid type in decode_mutagen_tag_value: {type(value)}"
            raise TypeError(msg)

        def mutagen_tag_as_str(value: str | mp4.MP4FreeForm) -> str:
            if isinstance(value, str):
                return value

            if isinstance(value, mp4.MP4FreeForm):
                if value.dataformat != mp4.AtomDataType.UTF8:
                    msg = f"Invalid AtomDataType for str tag: {value.dataformat}"
                    raise RuntimeError(msg)
                return bytes(value).decode()

            assert_never(value)

        def mutagen_tag_as_json(
            value: str | mp4.MP4FreeForm,
        ) -> SerializableDict | str | int:
            str_value = mutagen_tag_as_str(value)

            return json.loads(str_value)

        def mutagen_tag_as_uuid(
            *,
            is_hex: bool,
        ) -> Callable[[str | mp4.MP4FreeForm], UUID]:
            if not is_hex:

                def impl_raw(value: str | mp4.MP4FreeForm) -> UUID:
                    if isinstance(value, str):
                        msg = "Invalid type for raw UUID: str"
                        raise TypeError(msg)

                    if isinstance(value, mp4.MP4FreeForm):
                        if value.dataformat != mp4.AtomDataType.UUID:
                            msg = f"Invalid AtomDataType for raw uuid tag: {value.dataformat}"
                            raise RuntimeError(msg)

                        return uuid_from_bytes(ISOM_BYTE_ORDER, bytes(value))

                    assert_never(value)

                return impl_raw

            def impl_hex(value: str | mp4.MP4FreeForm) -> UUID:
                if isinstance(value, str):
                    return uuid_from_str(value)

                if isinstance(value, mp4.MP4FreeForm):
                    if value.dataformat != mp4.AtomDataType.UTF8:
                        msg = (
                            f"Invalid AtomDataType for hex uuid tag: {value.dataformat}"
                        )
                        raise RuntimeError(msg)

                    return uuid_from_str(bytes(value).decode())

                assert_never(value)

            return impl_hex

        result: MetadataTagsRead = MetadataTagsRead(None, None, {}, [])

        if not isinstance(self.__instance, mp4.MP4):
            msg = _(
                "Unrecognized mutagen instance, this is an implementation error: {clazz}"  # noqa: COM812
            ).format(clazz=type(self.__instance))
            raise TypeError(msg)

        for key, value in self.__instance.items():
            if key == "\xa9cmt":
                if result.comment is not None:
                    msg = f"Duplicate comment tag read: {value}"
                    raise RuntimeError(msg)

                result.comment = decode_mutagen_tag_value(value, mutagen_tag_as_str)

            elif TaggerDomain.key_start_with(key):
                actual_key = TaggerDomain.get_raw(key)

                if result.metadata.get(actual_key, None) is not None:
                    msg = f"Duplicate metadata key tag read: {actual_key} -> {value}"
                    raise RuntimeError(msg)

                result.metadata[actual_key] = decode_mutagen_tag_value(
                    value,
                    mutagen_tag_as_json,
                )

            elif key in [TaggerDomain.UUID_HEX_KEY, TaggerDomain.UUID_RAW_KEY]:
                uuid = decode_mutagen_tag_value(
                    value,
                    mutagen_tag_as_uuid(is_hex=key == TaggerDomain.UUID_HEX_KEY),
                )

                if result.uuid is not None:
                    if result.uuid != uuid:
                        msg = f"Duplicate uuid tag read, that are not the same: {uuid}"
                        raise RuntimeError(msg)
                else:
                    result.uuid = uuid

            else:
                result.unrecognized.append(
                    (key, decode_mutagen_tag_value(value, mutagen_tag_as_str)),
                )

        return result

@decorate_class(slots=True)
class VideoTaggerMutagen(VideoTagger):
    __file: Path

    def __init__(
        self: Self,
        file: Path,
    ) -> None:
        self.__file = file

    @staticmethod
    def __get_handle_impl(
        file: Path,
        *,
        ctx: ContextType,
    ) -> Result[
        tuple[MutagenFileWrapper, mutagen.FileType],
        str,
    ]:
        filething = MutagenFileWrapper(
            file=file,
            chunk_size=PROGRESS_CHUNK_SIZE,
            ctx=ctx,
        )

        try:

            instance = mutagen.File(filething, easy=False)

            if instance is None:
                return Err(
                    _("Not supported file type"),
                )

            return Ok((filething, instance))
        except RuntimeError as err:
            return Err(
                _("get tag handle {err}").format(err=err),
            )
        except MutagenError as err:
            return Err(
                _("get tag handle (mutagen impl error): {err}").format(err=err),
            )

    @staticmethod
    def get_handle(file: Path) -> Result["VideoTaggerMutagen", str]:
        result = VideoTaggerMutagen.__get_handle_impl(file, ctx="r")

        if result.err():
            return Err(result.as_err())

        filething, _instance = result.as_ok()

        filething.close()

        return Ok(VideoTaggerMutagen(file))

    def __context_impl(
        self: Self,
        manager: ManagerInterface,
        ctx: ContextType,
    ) -> AbstractContextManager[VideoTaggerContextRW]:

        def get_things() -> tuple[MutagenFileWrapper, mutagen.FileType]:
            result = VideoTaggerMutagen.__get_handle_impl(self.__file, ctx=ctx)

            if result.err():
                msg = _(
                    "Mutagen failed, after we checked, that it would work: {err}"  # noqa: COM812
                ).format(err=result.as_err())
                raise RuntimeError(msg)

            return result.as_ok()

        @decorate_class(slots=True)
        class VideoTaggerContextCtx(AbstractContextManager[VideoTaggerContextRW]):
            __filething: Optional[MutagenFileWrapper]

            def __init__(self: Self) -> None:
                self.__filething = None

            @override
            def __enter__(self: Self) -> VideoTaggerContextRW:
                filething, instance = get_things()
                self.__filething = filething

                return VideoTaggerContextWrapperGeneric(
                    manager,
                    VideoTaggerContextMutagen(
                        filething=filething,
                        instance=instance,
                        manager=manager,
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
                if self.__filething is not None:
                    self.__filething.close()
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
