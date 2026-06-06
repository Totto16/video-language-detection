from collections.abc import Generator
from io import BufferedIOBase
from typing import Optional, Self, override

from content.language import ShortLanguageStr
from content.tagger.lcid_languages import LCID
from content.tagger.parser import (
    ByteOrder,
    Packable,
    Packer,
    Unpacker,
    UnsignedInt,
    UnsignedShort,
    read_checked,
)
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
STRH_FOURCC: FOURCC = FOURCC(b"strh")
STRL_FOURCC: FOURCC = FOURCC(b"strl")
HDRL_FOURCC: FOURCC = FOURCC(b"hdrl")

AUDS_FOURCC = FOURCC(b"auds")
MIDS_FOURCC = FOURCC(b"mids")
TXTS_FOURCC = FOURCC(b"txts")
VIDS_FOURCC = FOURCC(b"vids")


class AVIChunkSpan:
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
            msg = f"Invalid chunk: sitze too small: {self.size}"
            raise RuntimeError(msg)

        if self.size < self.header_size:
            msg = f"Invalid chunk size {self.size} at {self.start}"
            raise RuntimeError(msg)

    @staticmethod
    def from_avi_specified_size(
        start: int, size: int, header_size: int
    ) -> "AVIChunkSpan":
        return AVIChunkSpan(start, size + 8, header_size)

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
            msg = f"Invalid chunk size {self.size} at {self.start}"
            raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<AVIChunkSpan start: {self.start} size: {self.size} header: [0, {self.header_size}] payload: [{self.payload_start}, {self.payload_size}]>"

    def __repr__(self: Self) -> str:
        return str(self)


AVI_BYTE_ORDER = ByteOrder.Little


class AVIChunk:
    fourcc: FOURCC
    span: AVIChunkSpan
    is_list: bool

    def __init__(
        self: Self, fourcc: FOURCC, span: AVIChunkSpan, *, is_list: bool
    ) -> None:
        self.fourcc = fourcc
        self.span = span
        self.is_list = is_list

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AVIChunk":
        # spec https://learn.microsoft.com/en-us/previous-versions/ms779636(v=vs.85)
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
        span = AVIChunkSpan.from_avi_specified_size(offset, size, header_size=8)
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
        # spec https://learn.microsoft.com/en-us/previous-versions/ms779636(v=vs.85)
        # AVI List structure:
        # chunk    | <chunk size> bytes | parent chunk
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


class AVIStreamHeader(AVIChunk):
    type: FOURCC

    def __init__(self: Self, parent: AVIChunk, typ: FOURCC) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.type = typ

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: AVIChunk,
    ) -> "AVIStreamHeader":
        # spec https://learn.microsoft.com/en-us/previous-versions/ms779638(v=vs.85)
        # AVI Stream Header structure:
        # chunk    | <chunk size> bytes | parent chunk
        # ... data, see below

        # typedef struct _avistreamheader {
        #     FOURCC fcc; < -|
        #     DWORD  cb; # <- both in  the parent
        #     FOURCC fccType;
        #     FOURCC fccHandler;
        #     DWORD  dwFlags;
        #     WORD   wPriority;
        #     WORD   wLanguage;
        #     DWORD  dwInitialFrames;
        #     DWORD  dwScale;
        #     DWORD  dwRate;
        #     DWORD  dwStart;
        #     DWORD  dwLength;
        #     DWORD  dwSuggestedBufferSize;
        #     DWORD  dwQuality;
        #     DWORD  dwSampleSize;
        #     struct {
        #         short int left;
        #         short int top;
        #         short int right;
        #         short int bottom;
        #     }  rcFrame;
        # } AVISTREAMHEADER;

        f.seek(parent.span.payload_start)

        typ_raw = read_checked(f, 4)

        typ = FOURCC(typ_raw)

        additional_header_size = (
            4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + (2 + 2 + 2 + 2)
        )

        parent.span.add_header_size(additional_header_size)

        if parent.span.payload_size != 0:
            msg = f"Implementation error"
            raise ValueError(msg)

        return AVIStreamHeader(parent, typ)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AVIStreamHeader":
        chunk = AVIChunk.read_from_stream(f, offset)
        return AVIStreamHeader.__read_from_stream_impl(f, chunk)

    @property
    def __language_offset(self: Self) -> int:
        return 4 + 4 + 4 + 4 + 4 + 2

    def read_language(self: Self, f: BufferedIOBase) -> ShortLanguageStr | str:
        f.seek(self.span.start + self.__language_offset)

        lang_bytes = read_checked(f, 2)
        packed = Unpacker.unpack_one(AVI_BYTE_ORDER, UnsignedShort(), lang_bytes)

        return LCID.decode_language(packed)

    def patch_language(
        self: Self, f: BufferedIOBase, new_language: ShortLanguageStr,
    ) -> None:
        packed = LCID.encode_language(new_language)

        f.seek(self.span.start + self.__language_offset)

        packed_bytes = Packer.pack_one(AVI_BYTE_ORDER, UnsignedShort(), packed, 2)

        f.write(packed_bytes)
        f.flush()

        f.seek(self.span.start + self.__language_offset)
        verify_bytes = read_checked(f, 2)
        verify = Unpacker.unpack_one(AVI_BYTE_ORDER, UnsignedShort(), verify_bytes)

        if verify != packed:
            msg = "Invalid overwrite"
            raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<AVIStreamHeader parent: {AVIChunk.__str__(self)} type {self.type}>"

    def __repr__(self: Self) -> str:
        return str(self)


def read_chunk_from_stream(f: BufferedIOBase, pos: int) -> AVIChunk:
    chunk = AVIChunk.read_from_stream(f, pos)

    match chunk.fourcc.value:
        case b"RIFF":
            return AVIList.read_from_stream(f, pos)
        case b"LIST":
            return AVIList.read_from_stream(f, pos)
        case b"strh":
            return AVIStreamHeader.read_from_stream(f, pos)
        case _:
            return chunk


def avi_iter_chunks(f: BufferedIOBase, start: int, end: int) -> Generator[AVIChunk]:
    pos = start

    while pos < end:
        chunk = read_chunk_from_stream(f, pos)

        if pos + chunk.span.size > end:
            msg = f"chunk {chunk.fourcc!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield chunk
        pos += chunk.span.size

        # align by WORD (2 bytes)
        if (pos % 2) != 0:
            f.seek(pos)
            val = read_checked(f, 1)
            if val != b"\x00":
                msg = f"Invalid padding byte: {val!r}, it has to be 0x00"
                raise RuntimeError(msg)

            pos += 1


def find_strh_chunks_with_type(
    f: BufferedIOBase,
    types: list[FOURCC],
) -> Generator[AVIStreamHeader]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[int, int, list[FOURCC]]] = [(0, filesize, [])]

    while stack:
        start, end, path = stack.pop()

        for chunk in avi_iter_chunks(f, start, end):

            if chunk.fourcc == STRH_FOURCC:
                if not isinstance(chunk, AVIStreamHeader):
                    msg = (
                        "Invalid AVIStreamHeader: type not dispatched to correct class"
                    )
                    raise ValueError(msg)

                if chunk.type not in types:
                    continue

                current = path[-2:]
                if current != [
                    HDRL_FOURCC,
                    STRL_FOURCC,
                ]:
                    msg = f"invalid strh chunk hierarchy: {current}"
                    raise RuntimeError(msg)

                yield chunk

            if chunk.is_list:
                typ = chunk.fourcc
                if isinstance(chunk, AVIList):
                    typ = chunk.type
                stack.append((chunk.span.payload_start, chunk.span.end, [*path, typ]))


def is_avi_file(f: BufferedIOBase) -> Optional[str]:
    f.seek(0)

    try:

        first_chunk = read_chunk_from_stream(f, 0)

        if not isinstance(first_chunk, AVIList):
            return _("Not a valid RIFF / AVI file")

        if first_chunk.fourcc != RIFF_FOURCC:
            return _(
                "RIFF/AVI file has valid chunk, but it is not the correct starting chunk: {first_chunk!r}"
            ).format(first_chunk=first_chunk.fourcc)

        if first_chunk.type not in [AVI__FOURCC, AVIX_FOURCC]:
            return _(
                "RIFF/AVI file has valid chunk, but it is not the correct starting chunk, list type invalid: {list_type!r}"
            ).format(list_type=first_chunk.type)

        f.seek(0)
    except RuntimeError as err:
        return str(err)
    except ValueError as err:
        return str(err)
    return None
