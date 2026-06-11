from collections.abc import Generator
from io import BufferedIOBase
from typing import Any, Optional, Self, final, override

from content.language import ShortLanguageStr
from content.tagger.lcid_languages import LCID
from content.tagger.parser import (
    BoundedIO,
    ByteOrder,
    Packable,
    Packer,
    SimpleSpan,
    Unpacker,
    UnsignedInt,
    UnsignedShort,
)
from helper.translation import get_translator

_ = get_translator()


@final
class FOURCC:
    __value: bytes

    def __init__(self: Self, value: bytes) -> None:
        self.__value = value

        if len(value) != 4:
            msg = f"Invalid FOURCC name length {len(value)}"
            raise ValueError(msg)

        def is_valid_byte(byte: int) -> bool:
            val = bytes([byte])

            if val.islower():
                return True

            if val.isupper():
                return True

            if val.isdigit():
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


@final
class PackableFOURCC(Packable[FOURCC, bytes]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "4s"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4

    @override
    def to_underlying(self: Self, value: FOURCC) -> bytes:
        return value.value

    @override
    def from_underlying(self: Self, value: bytes) -> FOURCC:
        return FOURCC(value)


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


@final
class AVIChunkSpan:
    __total: SimpleSpan

    __intervals: list[int]

    def __init__(
        self: Self,
        span: SimpleSpan,
        header_size: int,
    ) -> None:
        self.__total = span
        self.__intervals = [header_size]

        if self.__total.size < 8:
            msg = f"Invalid chunk: sitze too small: {self.__total.size}"
            raise RuntimeError(msg)

        if self.__total.size < header_size:
            msg = f"Invalid chunk size {self.__total.size} at {self.__total.start}"
            raise RuntimeError(msg)

    @staticmethod
    def from_avi_specified_size(
        span: SimpleSpan,
        header_size: int,
    ) -> "AVIChunkSpan":
        return AVIChunkSpan(SimpleSpan(span.start, span.size + 8), header_size)

    def __interval_span_impl(self: Self, depth: int = 0) -> SimpleSpan:
        if len(self.__intervals) == 0:
            msg = "Implementation error: intervals list is empty"
            raise RuntimeError(msg)

        if depth < 0:
            msg = f"Invalid depth, it is negative: {depth}"
            raise RuntimeError(msg)

        if depth > len(self.__intervals):
            msg = f"Invalid depth of interval size: {depth}, max is {len(self.__intervals)}"
            raise RuntimeError(msg)

        interval_start = self.__total.start
        interval_end = self.__total.end

        if depth != 0:
            interval_start = self.__total.start + sum(self.__intervals[0:depth])

        if depth != len(self.__intervals):
            interval_end = self.__total.start + sum(self.__intervals[0 : depth + 1])

        interval_size = interval_end - interval_start

        if interval_size > self.__total.size or interval_size < 0:
            msg = f"Implementation error, interval_size out of bounds [0, {self.__total.size}]: {interval_size}"

        return SimpleSpan(interval_start, interval_size)

    def header_span(self: Self, depth: int = 0) -> SimpleSpan:
        if depth == -1:
            return self.header_span(len(self.__intervals) - 1)

        if depth >= len(self.__intervals):
            msg = f"Invalid depth of header size: {depth}, max is {len(self.__intervals) - 1}"
            raise RuntimeError(msg)

        return self.__interval_span_impl(depth)

    @property
    def payload_span(self: Self) -> SimpleSpan:
        return self.__interval_span_impl(len(self.__intervals))

    def add_header(self: Self, header_size: int) -> None:
        self.__intervals.append(header_size)

        if self.__total.size < sum(self.__intervals):
            msg = f"Invalid total header size {self.__total.size} < {sum(self.__intervals)} at {self.__total.start}"
            raise RuntimeError(msg)

    @property
    def total(self: Self) -> SimpleSpan:
        return self.__total

    def __str__(self: Self) -> str:
        header_string = ", ".join(
            str(self.header_span(i)) for i in range(0, len(self.__intervals))
        )
        return f"<AVIChunkSpan total: {self.__total} header: [ {header_string} ] payload: {self.payload_span}>"

    def __repr__(self: Self) -> str:
        return str(self)


AVI_BYTE_ORDER = ByteOrder.Little


class FinalAVIChunk:
    __final__avi_chunk__ = True


class NonFinalAVIChunk:
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        is_final = getattr(cls, "__final__avi_chunk__", False)

        if not is_final:
            for fn_name in [
                "read",
                "read_from_parent",
            ]:
                if fn_name in cls.__dict__:
                    msg = f"{cls.__name__} defines {fn_name}(), but only final classes may do so"
                    raise TypeError(msg)


class AVIChunk(NonFinalAVIChunk):
    fourcc: FOURCC
    span: AVIChunkSpan
    is_list: bool

    def __init__(
        self: Self,
        fourcc: FOURCC,
        span: AVIChunkSpan,
        *,
        is_list: bool,
    ) -> None:
        self.fourcc = fourcc
        self.span = span
        self.is_list = is_list

    @staticmethod
    def read_avi_chunk(io: BoundedIO) -> "AVIChunk":
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

        with io.r_ctx(force_entire_read=False) as f:
            hdr = f.read(8)

            fourcc, size = Unpacker.unpack_two(
                AVI_BYTE_ORDER,
                (PackableFOURCC(), UnsignedInt()),
                hdr,
            )

            span = AVIChunkSpan.from_avi_specified_size(
                SimpleSpan(io.span.start, size),
                header_size=8,
            )
            return AVIChunk(fourcc, span, is_list=False)

    @final
    def payload_io(self: Self, io: BoundedIO) -> BoundedIO:
        return io.new_span_io(
            self.span.payload_span,
        )

    @final
    def header_io(self: Self, io: BoundedIO, depth: int = 0) -> BoundedIO:
        return io.new_span_io(
            self.span.header_span(depth),
        )

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
    def __read_impl(io: BoundedIO, parent: AVIChunk) -> "AVIList":
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

        with io.r_ctx(force_entire_read=False) as f:
            typ_raw = f.read(4)

            typ = FOURCC(typ_raw)

            parent.span.add_header(4)

            return AVIList(parent, typ)

    @staticmethod
    def read_avi_list(io: BoundedIO) -> "AVIList":
        chunk = AVIChunk.read_avi_chunk(io)
        return AVIList.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_avi_list_from_parent(io: BoundedIO, parent: AVIChunk) -> "AVIList":
        return AVIList.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<AVIList parent: {AVIChunk.__str__(self)} type {self.type}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class AVIStreamHeader(AVIChunk, FinalAVIChunk):
    type: FOURCC

    def __init__(self: Self, parent: AVIChunk, typ: FOURCC) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.type = typ

    @staticmethod
    def __read_impl(
        io: BoundedIO,
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

        with io.r_ctx(force_entire_read=True) as f:
            typ_raw = f.read(4)

            typ = FOURCC(typ_raw)

            additional_header_size = (
                4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + (2 + 2 + 2 + 2)
            )

            f.skip(additional_header_size - 4)
            parent.span.add_header(additional_header_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return AVIStreamHeader(parent, typ)

    @staticmethod
    def read(io: BoundedIO) -> "AVIStreamHeader":
        chunk = AVIChunk.read_avi_chunk(io)
        return AVIStreamHeader.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: AVIChunk) -> "AVIStreamHeader":
        return AVIStreamHeader.__read_impl(io, parent)

    @property
    def __language_offset(self: Self) -> int:
        # offset from the own header start, not the start of the whole chunk!
        return 4 + 4 + 4 + 2

    def read_language(self: Self, io_base: BufferedIOBase) -> ShortLanguageStr | str:
        io = self.header_io(BoundedIO.get_new(io_base, self.span.total), -1)

        with io.r_ctx(force_entire_read=False) as f:
            f.skip(self.__language_offset)

            lang_bytes = f.read(2)
            packed = Unpacker.unpack_one(AVI_BYTE_ORDER, UnsignedShort(), lang_bytes)

        return LCID.decode_language(packed)

    def patch_language(
        self: Self,
        io_base: BufferedIOBase,
        new_language: ShortLanguageStr,
    ) -> None:
        packed = LCID.encode_language(new_language)

        io = self.header_io(BoundedIO.get_new(io_base, self.span.total), -1)

        with io.rw_ctx(force_entire_read=False) as f:
            f.skip(self.__language_offset)

            packed_bytes = Packer.pack_one(AVI_BYTE_ORDER, UnsignedShort(), packed, 2)

            f.write(packed_bytes)
            f.flush()

        with io.r_ctx(force_entire_read=False) as f:
            f.skip(self.__language_offset)
            verify_bytes = f.read(2)
            verify = Unpacker.unpack_one(AVI_BYTE_ORDER, UnsignedShort(), verify_bytes)

            if verify != packed:
                msg = "Invalid overwrite"
                raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<AVIStreamHeader parent: {AVIChunk.__str__(self)} type {self.type}>"

    def __repr__(self: Self) -> str:
        return str(self)


def read_chunk(io: BoundedIO) -> AVIChunk:
    chunk = AVIChunk.read_avi_chunk(io)

    match chunk.fourcc.value:
        case b"RIFF":
            return AVIList.read_avi_list_from_parent(chunk.payload_io(io), chunk)
        case b"LIST":
            return AVIList.read_avi_list_from_parent(chunk.payload_io(io), chunk)
        case b"strh":
            return AVIStreamHeader.read_from_parent(chunk.payload_io(io), chunk)
        case _:
            return chunk


def avi_iter_chunks(
    io_base: BufferedIOBase,
    span: SimpleSpan,
) -> Generator[AVIChunk]:
    pos = span.start

    while pos < span.end:
        io = BoundedIO.get_new(io_base, SimpleSpan(pos, span.end - pos))
        chunk = read_chunk(io)

        if pos + chunk.span.total.size > span.end:
            msg = f"chunk {chunk.fourcc!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield chunk
        pos += chunk.span.total.size

        # align by WORD (2 bytes)
        if (pos % 2) != 0:
            with BoundedIO.get_new(io_base, SimpleSpan(pos, 1)).r_ctx(
                force_entire_read=True,
            ) as f:
                val = f.read(1)
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

    stack: list[tuple[SimpleSpan, list[FOURCC]]] = [(SimpleSpan(0, filesize), [])]

    while stack:
        span, path = stack.pop()

        for chunk in avi_iter_chunks(f, span):

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
                stack.append((chunk.span.payload_span, [*path, typ]))


def is_avi_file(
    f: BufferedIOBase,
) -> Optional[str]:
    f.seek(0)

    try:
        f.seek(0, 2)
        filesize = f.tell()
        first_chunk = read_chunk(BoundedIO.get_new(f, SimpleSpan(0, filesize)))

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
    except (RuntimeError, ValueError) as err:
        return str(err)
    return None
