from collections.abc import Generator
from contextlib import AbstractContextManager
from io import BufferedIOBase, BytesIO
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Self, override

from content.language import Alpha3LanguageStr, Language
from content.tagger.parser import (
    ByteOrder,
    Packable,
    Packer,
    Unpacker,
    UnsignedInt,
    UnsignedLongLong,
    UnsignedShort,
    read_checked,
)
from content.tagger.video_tagger import (
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    VideoTagger,
    VideoTagger__HandleResult,
    VideoTaggerWriter,
)
from helper.manager import CounterInterface, ManagerInterface
from helper.translation import get_translator

_ = get_translator()


class FOURCC:
    __value: bytes

    def __init__(self: Self, value: bytes) -> None:
        self.__value = value

        if len(value) != 4:
            msg = f"Invalid FOURCC name length {len(value)}"
            raise ValueError(msg)

        def is_valid_byte(byte: int) -> bool:
            val = bytes([byte])

            if val.lower():
                return True

            if val.upper():
                return True

            return val == b" "

        if not all(is_valid_byte(val) for val in value):
            msg = f"FOURCC not valid {value!s}"
            raise ValueError(msg)

    @property
    def value(self: Self) -> bytes:
        return self.__value

    def __str__(self: Self) -> str:
        return str(self.__value)

    def __repr__(self: Self) -> str:
        return repr(self.__value)

    def __hash__(self: Self) -> int:
        return hash(self.__value)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, FOURCC):
            return self.__value == other.__value

        if isinstance(other, str):
            return self.__value == other.encode()

        if isinstance(other, bytes):
            return self.__value == other

        return False


class PackableFOURCC(Packable[bytes]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "4s"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4


RIFF_FOURCC: FOURCC = FOURCC(b"RIFF")
AVI__FOURCC: FOURCC = FOURCC(b"AVI ")
AVIX_FOURCC: FOURCC = FOURCC(b"AVIX")
LIST_FOURCC: FOURCC = FOURCC(b"LIST")


class AVIBoxSpan:
    start: int
    size: int
    header_size: int

    def __init__(
        self: Self,
        start: int,
        size: int,
        header_size: int,
    ) -> None:
        self.start = start
        self.size = size
        self.header_size = header_size

        if self.size < 8:
            msg = f"Invalid box: sitze too small: {self.size}"
            raise RuntimeError(msg)

        if self.size < self.header_size:
            msg = f"Invalid box size {self.size} at {self.start}"
            raise RuntimeError(msg)

    @staticmethod
    def from_avi_specified_size(
        start: int, size: int, header_size: int
    ) -> "AVIBoxSpan":
        return AVIBoxSpan(start, size + 8, header_size)

    @property
    def end(self: Self) -> int:
        return self.start + self.size

    @property
    def payload_start(self: Self) -> int:
        return self.start + self.header_size

    @property
    def payload_size(self: Self) -> int:
        return self.size - self.header_size

    def add_header_size(self: Self, header_size: int) -> None:
        self.header_size = self.header_size + header_size
        if self.size < self.header_size:
            msg = f"Invalid box size {self.size} at {self.start}"
            raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<AVIBoxSpan start: {self.start} size: {self.size} header: [0, {self.header_size}] payload: [{self.payload_start}, {self.payload_size}]>"

    def __repr__(self: Self) -> str:
        return str(self)


AVI_BYTE_ORDER = ByteOrder.Little


class AVIChunk:
    fourcc: FOURCC
    span: AVIBoxSpan
    is_list: bool

    def __init__(
        self: Self, fourcc: FOURCC, span: AVIBoxSpan, *, is_list: bool
    ) -> None:
        self.fourcc = fourcc
        self.span = span
        self.is_list = is_list

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AVIChunk":
        # AVI Chunk structure:
        # fourcc | 4 bytes | char[4]
        # size   | 4 bytes | unsigned int
        # ... data

        # Note: size is the size after it, so 8 bytes less then the whole size

        # typedef struct {
        #     DWORD dwFourCC
        #     DWORD dwSize
        #     BYTE data[dwSize]
        # } CHUNK;

        f.seek(offset)

        hdr = read_checked(f, 8)

        fourcc_raw, size = Unpacker.unpack_two(
            AVI_BYTE_ORDER,
            (PackableFOURCC(), UnsignedInt()),
            hdr,
        )

        fourcc = FOURCC(fourcc_raw)
        span = AVIBoxSpan.from_avi_specified_size(offset, size, header_size=8)
        return AVIChunk(fourcc, span, is_list=False)

    def __str__(self: Self) -> str:
        return f"<AVIChunk fourcc: {self.fourcc} span: {self.span} is_list: {self.is_list}>"

    def __repr__(self: Self) -> str:
        return str(self)


class AVIList(AVIChunk):
    type: FOURCC

    def __init__(self: Self, parent: AVIChunk, typ: FOURCC) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=True)

        self.type = typ

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: AVIChunk) -> "AVIList":
        # AVI List structure:
        # box    | <chunk size> bytes | parent chunk
        # type   | 4 bytes | char[4]

        # typedef struct {
        #     DWORD dwList
        #     DWORD dwSize
        #     DWORD dwFourCC
        #     BYTE data[dwSize-4]
        # } LIST;

        f.seek(parent.span.payload_start)

        typ_raw = read_checked(f, 4)

        typ = FOURCC(typ_raw)

        parent.span.add_header_size(4)

        return AVIList(parent, typ)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AVIList":
        chunk = AVIChunk.read_from_stream(f, offset)
        return AVIList.__read_from_stream_impl(f, chunk)

    def __str__(self: Self) -> str:
        return f"<AVIList parent: {AVIChunk.__str__(self)} type {self.type}>"

    def __repr__(self: Self) -> str:
        return str(self)


def read_chunk_from_stream(f: BufferedIOBase, pos: int) -> AVIChunk:
    chunk = AVIChunk.read_from_stream(f, pos)

    match chunk.fourcc.value:
        case b"RIFF":
            return AVIList.read_from_stream(f, pos)
        case b"LIST":
            return AVIList.read_from_stream(f, pos)
        case _:
            return chunk


def avi_iter_chunks(f: BufferedIOBase, start: int, end: int) -> Generator[AVIChunk]:
    pos = start

    while pos < end:
        chunk = read_chunk_from_stream(f, pos)

        if pos + chunk.span.size > end:
            msg = f"Box {chunk.fourcc!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield chunk
        pos += chunk.span.size


def find_mdhd_boxes_with_type(
    f: BufferedIOBase,
    types: list[ISOAtomName],
) -> Generator["MediaHeaderBox"]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[int, int, list[ISOAtomName]]] = [(0, filesize, [])]

    while stack:
        start, end, path = stack.pop()

        for box in mp4_iter_boxes(f, start, end):

            if box.type == TRAK_ATOM_NAME:
                if not isinstance(box, TrackBox):
                    msg = "Invalid TrackBox: type not dispatched to correct class"
                    raise ValueError(msg)

                hdlr = box.hdlr.handler_type

                if hdlr not in types:
                    continue

            if box.type == MDHD_ATOM_NAME:
                if not isinstance(box, MediaHeaderBox):
                    msg = "Invalid MediaHeaderBox: type not dispatched to correct class"
                    raise ValueError(msg)

                current = path
                if current != [
                    MOOV_ATOM_NAME,
                    TRAK_ATOM_NAME,
                    MDIA_ATOM_NAME,
                ]:
                    msg = f"invalid mdhd box hierarchy: {current}"
                    raise RuntimeError(msg)

                yield box

            if box.container:
                stack.append((box.span.payload_start, box.span.end, [*path, box.type]))


def is_avi_file(f: BufferedIOBase) -> Optional[str]:
    f.seek(0)

    first_chunk = read_chunk_from_stream(f, 0)

    if not isinstance(first_chunk, AVIList):
        return _("Not a valid RIFF / AVI file")

    if first_chunk.fourcc != RIFF_FOURCC:
        return _(
            "RIFF/AVI file has valid chunk, but it is not the correct starting chunk: {first_chunk!r}"
        ).format(first_chunk=first_chunk.fourcc)

    if first_chunk.type not in [AVI__FOURCC, AVIX_FOURCC]:
        return _(
            "RIFF/AVI file has valid chunk, but it sis not the correct starting chunk, list type invalid: {list_type!r}"
        ).format(list_type=first_chunk.type)

    f.seek(0)
    return None
