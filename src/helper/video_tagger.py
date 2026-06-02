from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, Optional, Self, assert_never, override

import mutagen._file as mutagen
from mutagen._util import MutagenError

from helper.manager import CounterInterface, ManagerInterface
import mutagen.mp4 as mp4


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


@dataclass
class IOOpProgress:
    type: Literal["progress"]
    which: Literal["read", "write"]
    amount: int


@dataclass
class IOOpSeek:
    type: Literal["seek"]
    amount: int


@dataclass
class IOOpTruncate:
    type: Literal["truncate"]
    amount: int


IOOp = IOOpProgress | IOOpSeek | IOOpTruncate

OpCallback = Callable[[IOOp], None]


class MutagenFileWrapper(IOInterface):
    __file: Path
    __impl: BinaryIO
    __callbacks: list[OpCallback]

    def __init__(
        self: Self,
        file: Path,
    ) -> None:
        super().__init__()

        self.__file = file
        self.__impl = self.__file.open(mode="rb+")
        self.__callbacks = []

    def __emit(self: Self, op: IOOp) -> None:
        for callback in self.__callbacks:
            callback(op)

    @override
    def tell(self: Self) -> int:
        return self.__impl.tell()

    @override
    def read(self: Self, size: int = -1) -> bytes:
        data = self.__impl.read(size)
        # TODO: read in blocks
        self.__emit(IOOpProgress("progress", "read", amount=len(data)))
        return data

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
        result = self.__impl.write(data)
        # TODO: write in blocks
        self.__emit(IOOpProgress("progress", "write", amount=result))
        return result

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


VIDEO_FILE_TAG_UPDATE_BAR_FORMAT: str = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


class VideoTaggerWriter:
    __filething: MutagenFileWrapper
    __instance: mutagen.FileType
    __manager: ManagerInterface

    def __init__(
        self: Self,
        filething: MutagenFileWrapper,
        instance: mutagen.FileType,
        manager: ManagerInterface,
    ) -> None:
        self.__instance = instance
        self.__filething = filething
        self.__manager = manager

    def __save_impl(self: Self) -> None:
        size: float = float(self.__filething.size())

        bar: CounterInterface = self.__manager.counter(
            total=size,
            desc="update video tags",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        bar.update(self.__filething.tell())

        try:

            def process_op(op: IOOp) -> None:
                match op.type:
                    case "progress":
                        bar.update(float(op.amount))
                    case "seek":
                        bar.update()
                    case "truncate":
                        bar.update()
                    case _:
                        assert_never(op.type)

            with self.__filething.callback_ctx(process_op):
                self.__instance.save(self.__filething)
        except MutagenError as err:
            msg = "tag error"
            raise RuntimeError(msg) from err
        finally:
            bar.close(clear=True)

    def write_metadata(
        self: Self,
        comment: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.__instance["\xa9cmt"] = comment
        if isinstance(self.__instance, mp4.MP4):
            for key, value in metadata.items():
                self.__instance[f"----:lt.totto:video_language_detect:{key}"] = [
                    mp4.MP4FreeForm(value, mp4.AtomDataType.IMPLICIT),
                ]

                self.__instance[key] = value

        for key, value in metadata.items():
            self.__instance[key] = value

        self.__save_impl()


class VideoTagger:
    __filething: MutagenFileWrapper
    __instance: mutagen.FileType

    def __init__(
        self: Self,
        filething: MutagenFileWrapper,
        instance: mutagen.FileType,
    ) -> None:
        self.__instance = instance
        self.__filething = filething

    @staticmethod
    def get_handle(file: Path) -> Optional["VideoTagger"]:

        filething = MutagenFileWrapper(file=file)

        instance = mutagen.File(filething=filething, easy=False)

        if instance is None:
            return None

        return VideoTagger(filething, instance)

    def writer(
        self: Self, manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerWriter]:

        filething = self.__filething
        instance = self.__instance

        class VideoTaggerWriterCtx(AbstractContextManager[VideoTaggerWriter]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> VideoTaggerWriter:
                return VideoTaggerWriter(
                    filething=filething,
                    instance=instance,
                    manager=manager,
                )

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                filething.close()
                return False

        return VideoTaggerWriterCtx()
