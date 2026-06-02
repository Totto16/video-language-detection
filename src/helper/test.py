from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, Self, override

import mutagen
import mutagen.mp4 as mp4

file_path = "/media/totto/Totto_4/Serien/Bodyguard (2018)/Staffel 01/Episode 01 - Folge 1 [S01E01].mp4"


## see https://mutagen.readthedocs.io/en/latest/user/filelike.html#
class IOInterface(object):
    """This is the interface mutagen expects from custom file-like
    objects.

    For loading read(), tell() and seek() have to be implemented. "name"
    is optional.

    For saving/deleting write(), flush() and truncate() have to be
    implemented in addition. fileno() is optional.
    """

    # For loading

    def tell(self):
        """Returns he current offset as int. Always >= 0.

        Raises IOError in case fetching the position is for some reason
        not possible.
        """

        raise NotImplementedError

    def read(self, size=-1):
        """Returns 'size' amount of bytes or less if there is no more data.
        If no size is given all data is returned. size can be >= 0.

        Raises IOError in case reading failed while data was available.
        """

        raise NotImplementedError

    def seek(self, offset, whence=0):
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
    def name(self):
        """Should return text. For example the file name.

        If not available the attribute can be missing or can return
        an empty string.

        Will be used for error messages and type detection.
        """

        raise NotImplementedError

    # For writing

    def write(self, data):
        """Write data to the file.

        Returns Nothing.
        Raises IOError
        """

        raise NotImplementedError

    def truncate(self, size=None):
        """Truncate to the current position or size if size is given.

        The current position or given size will never be larger than the
        file size.

        This has to flush write buffers in case writing is buffered.

        Returns Nothing.
        Raises IOError.
        """

        raise NotImplementedError

    def flush(self):
        """Flush the write buffer.

        Returns Nothing.
        Raises IOError.
        """

        raise NotImplementedError

    # For writing, but optional

    def fileno(self):
        """Returns the file descriptor (int) or raises IOError
        if there is none.

        Will be used for low level operations if available.
        """

        raise NotImplementedError


class MutagenFileWrapper(IOInterface):
    __file: Path
    __impl: BinaryIO

    @override
    def __init__(
        self: Self,
        file: Path,
    ) -> None:
        """write=False"""
        self.__file = file
        self.__impl = self.__file.open(mode="rb+")
        # self.__impl = self.__file.open(mode="ab+" if write else "rb")

    @override
    def tell(self: Self) -> int:
        print("tell")
        return self.__impl.tell()

    @override
    def read(self: Self, size: int = -1) -> bytes:
        print("read", size)
        return self.__impl.read(size)

    @override
    def seek(self: Self, offset: int, whence: int = 0) -> int:
        print("seek", offset, whence)
        return self.__impl.seek(offset, whence)

    # For loading, but optional

    @override
    @property
    def name(self: Self) -> str:
        print("name")
        return self.__file.name

    # For writing

    @override
    def write(self: Self, data: bytes) -> int:
        print("write", type(data), len(data))
        return self.__impl.write(data)

    @override
    def truncate(self: Self, size: Optional[int] = None) -> int:
        print("truncate", size)
        return self.__impl.truncate(size)

    @override
    def flush(self: Self) -> None:
        print("flush")
        return self.__impl.flush()

    # For writing, but optional

    def fileno(self: Self) -> int:
        print("fileno", fileno)
        return self.__impl.fileno()


file = MutagenFileWrapper(Path(file_path))

f = mutagen.File(file)

f["\xa9cmt"] = ["Some comment test"]  # comment

f["test"] = datetime.now().isoformat()

f["----:lt.totto:video_language_detect"] = [mp4.MP4FreeForm(b"Acme Corp test")]


print("SAVING NOW")
print(f)

file2 = MutagenFileWrapper(Path(file_path))

f.save(file)

print(f)
