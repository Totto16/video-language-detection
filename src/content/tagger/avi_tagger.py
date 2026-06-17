from collections.abc import Generator
from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Optional, Self, final, override

from content.language import Language, ShortLanguageStr
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
from content.tagger.video_tagger import (
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    ContextType,
    MetadataTags,
    MetadataTagsRead,
    VideoTagger,
    VideoTaggerContextReadable,
    VideoTaggerContextRW,
    VideoTaggerContextWrapperGeneric,
    VideoTaggerContextWriteable,
)
from helper.manager import CounterInterface, ManagerInterface
from helper.result import Err, Ok, Result
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
            msg = f"Invalid chunk: size too small: {self.__total.size}"
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
            str(self.header_span(i)) for i in range(len(self.__intervals))
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

    def add_content_afterwards_avi_chunk(
        self: Self,
        f: BinaryIO,
        data: bytes,
    ) -> Optional[str]:
        # TODO: check, that we are the top level chunk, otherwise we might need some data changes, alias relocation

        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)

        if self.span.total.start != 0 and self.span.total.end != filesize:
            return f"Can't add content to a non toplevel AVI chunk: {self.span}"

        old_size = self.span.total.size
        new_size = old_size + len(data)

        # align by WORD (2 bytes)
        if (old_size % 2) != 0:
            new_size = new_size + 1
            data = b"\x00" + data

        # Note: size is the size after it, so 8 bytes less then the whole size
        if (new_size - 8) >= 0xFFFFFFFF:
            return f"Size is too big, can't fit in the AVI chunk size: {new_size - 8}"

        io = BoundedIO.get_new(io_base=f, span=self.span.total)

        # write the new size
        with self.header_io(io).rw_ctx(force_entire_read=True) as ctx:
            ctx.skip(4)

            size_bytes = Packer.pack_one(
                AVI_BYTE_ORDER,
                UnsignedInt(),
                new_size,
                4,
            )

            ctx.write(size_bytes)
            ctx.flush()

        f.seek(old_size)

        if filesize != f.tell():
            return "can't write over other chunks atm"

        f.write(data)
        f.flush()

        return None

    @staticmethod
    def write_to_buffer_avi_chunk(
        fourcc: FOURCC,
        data: bytes,
    ) -> bytes:
        buf = BytesIO()

        # Note: size is the size after it, so 8 bytes less then the whole size (but the exact size as the data)
        final_size: int = len(data)

        hdr = Packer.pack_two(
            AVI_BYTE_ORDER,
            (PackableFOURCC(), UnsignedInt()),
            (
                fourcc,
                final_size,
            ),
            8,
        )

        buf.write(hdr)

        buf.write(data)

        # align by WORD (2 bytes)
        if buf.tell() % 2 != 0:
            buf.write(b"\x00")

        return buf.getvalue()

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

    def add_content_afterwards_avi_list(
        self: Self,
        f: BinaryIO,
        data: bytes,
    ) -> Optional[str]:
        # the exact same as the avi chunk method, but it might be different, this is an implementation detail

        return self.add_content_afterwards_avi_chunk(f, data)

    @staticmethod
    def write_to_buffer_avi_list(
        fourcc: FOURCC,
        typ: FOURCC,
        data: list[bytes],
    ) -> bytes:

        if fourcc not in [LIST_FOURCC, RIFF_FOURCC]:
            msg = f"Only {LIST_FOURCC} and {RIFF_FOURCC} as list fourcc supported atm, but got: {fourcc}"
            raise RuntimeError(msg)

        buf = BytesIO()

        buf.write(typ.value)

        for single_data in data:
            buf.write(single_data)

            # align by WORD (2 bytes)
            if buf.tell() % 2 != 0:
                buf.write(b"\x00")

        final_data = buf.getvalue()

        return AVIChunk.write_to_buffer_avi_chunk(fourcc, final_data)


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

    def read_language(self: Self, io_base: BinaryIO) -> ShortLanguageStr | str:
        io = self.header_io(BoundedIO.get_new(io_base, self.span.total), -1)

        with io.r_ctx(force_entire_read=False) as f:
            f.skip(self.__language_offset)

            lang_bytes = f.read(2)
            packed = Unpacker.unpack_one(AVI_BYTE_ORDER, UnsignedShort(), lang_bytes)

        return LCID.decode_language(packed)

    def patch_language(
        self: Self,
        io_base: BinaryIO,
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
    io_base: BinaryIO,
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
    f: BinaryIO,
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
    f: BinaryIO,
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
                "RIFF/AVI file has valid chunk, but it is not the correct starting chunk: {first_chunk!r}",
            ).format(first_chunk=first_chunk.fourcc)

        if first_chunk.type not in [AVI__FOURCC, AVIX_FOURCC]:
            return _(
                "RIFF/AVI file has valid chunk, but it is not the correct starting chunk, list type invalid: {list_type!r}",
            ).format(list_type=first_chunk.type)

        f.seek(0)
    except (RuntimeError, ValueError) as err:
        return str(err)
    return None


class VideoTaggerContextAVI(VideoTaggerContextRW):
    __writer: BinaryIO
    __streams: int
    __types: list[FOURCC]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        writer: BinaryIO,
        streams: int,
        types: list[FOURCC],
    ) -> None:
        super().__init__(manager)
        self.__writer = writer
        self.__streams = streams
        self.__types = types

    @override
    def write_tags(
        self: Self,
        tags: MetadataTags,
    ) -> None:

        f = self.__writer

        f.seek(0, 2)
        filesize = f.tell()

        span = SimpleSpan(0, filesize)

        top_level_chunks = list(avi_iter_chunks(f, span))

        if len(top_level_chunks) != 1:
            msg = f"Expected only one RIFF top level chunk, but got {len(top_level_chunks)}"
            raise RuntimeError(msg)

        top_level_chunk = top_level_chunks[0]

        if not isinstance(top_level_chunk, AVIList):
            msg = f"Expected only one RIFF top level chunk, but got {top_level_chunk}"
            raise TypeError(msg)

        if top_level_chunk.fourcc != RIFF_FOURCC:
            msg = f"Expected only one RIFF top level chunk, but got {top_level_chunk.fourcc}"
            raise TypeError(msg)

        icmt_data = AVIChunk.write_to_buffer_avi_chunk(
            FOURCC(b"ICMT"),
            b"Test Comment\x00",
        )

        # vldc stands for video language detector chunk
        CUSTOM_FOURCC = FOURCC(b"vldc")

        custom_data = AVIChunk.write_to_buffer_avi_chunk(
            CUSTOM_FOURCC,
            b"Test Custom data\x00",
        )

        info_list_data = AVIList.write_to_buffer_avi_list(
            LIST_FOURCC,
            FOURCC(b"INFO"),
            [icmt_data, custom_data],
        )

        top_level_chunk.add_content_afterwards_avi_list(f, info_list_data)

    @override
    def write_language(
        self: Self,
        language: Language,
    ) -> bool:
        new_language = language.short

        # TODO. replace VIDEO_FILE_TAG_UPDATE_BAR_FORMAT everywhere, as we don't use bytes here!
        bar: CounterInterface = self.manager.counter(
            total=float(self.__streams + 1),
            desc="update avi language",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        try:
            self.__writer.seek(0)
            for strh in find_strh_chunks_with_type(self.__writer, self.__types):

                should_write_language = True

                lang = strh.read_language(self.__writer)
                if isinstance(lang, ShortLanguageStr) and new_language == lang:
                    should_write_language = False

                if should_write_language:
                    strh.patch_language(self.__writer, new_language)

                bar.update(1, force=True)

            self.__writer.flush()
        finally:
            bar.close(clear=True)

        return True

    def __impl(
        self: Self,
        f: BinaryIO,
        span: SimpleSpan,
        depth: int,
    ) -> None:
        for chunk in avi_iter_chunks(f, span):
            if chunk.is_list:
                if not isinstance(chunk, AVIList):
                    msg = "Invalid AVIList: type not dispatched to correct class"
                    raise ValueError(msg)

                msg = ("  " * depth) + f"{chunk.fourcc} - {chunk.type}"
                print(msg)
                self.__impl(f, chunk.span.payload_span, depth + 1)

                # if chunk.type == b"INFO":
                #     print("             INFO chunk: ")
                #     f.seek(chunk.span.total.start)
                #     c = f.read(chunk.span.total.size)
                #     print(c)

                continue

            msg = ("  " * depth) + f"{chunk.fourcc}"
            print(msg)

    @override
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead:
        f = self.__writer

        f.seek(0, 2)
        filesize = f.tell()

        span = SimpleSpan(0, filesize)

        self.__impl(f, span, 0)

        return MetadataTagsRead(None, None, {}, [])
        # raise NotImplementedError("TODO")


class VideoTaggerAVI(VideoTagger):
    __streams: int
    __types: list[FOURCC]

    def __init__(
        self: Self,
        file: Path,
        streams: int,
        types: list[FOURCC],
    ) -> None:
        super().__init__(file)
        self.__streams = streams
        self.__types = types

        streams = 0

    @staticmethod
    def get_handle(file: Path) -> Result["VideoTagger", str]:

        try:

            with file.open("rb") as f:
                avi_res = is_avi_file(f)
                if avi_res is not None:
                    return Err(avi_res)

                f.seek(0)

                streams = 0
                types: list[FOURCC] = [AUDS_FOURCC, VIDS_FOURCC]

                # read the file, so that we check if we can parse it correctly and that it is an avi file
                for strh in find_strh_chunks_with_type(f, types):
                    streams = streams + 1
                    lang = strh.read_language(f)
                    # check if this lang is valid

                    if isinstance(lang, str):
                        msg = _("Invalid language in avi detected: {lang}").format(
                            lang=lang,
                        )
                        return Err(msg)

                return Ok(
                    VideoTaggerAVI(file, streams, types),
                )
        except (RuntimeError, ValueError, TypeError) as err:
            return Err(str(err))

    def __context_impl(
        self: Self,
        manager: ManagerInterface,
        ctx: ContextType,
    ) -> AbstractContextManager[VideoTaggerContextRW]:

        file = self.file
        streams = self.__streams
        types = self.__types

        class VideoTaggerContextCtx(AbstractContextManager[VideoTaggerContextRW]):
            __writer: Optional[BinaryIO]
            __backup: Optional[bytes]

            def __init__(self: Self) -> None:
                super().__init__()
                self.__writer = None

            @override
            def __enter__(self: Self) -> VideoTaggerContextRW:
                writer = file.open(mode="rb" if ctx == "r" else "rb+")

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
                    manager,
                    VideoTaggerContextAVI(manager, writer, streams, types),
                    ctx,
                )

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
                    if self.__backup is None:
                        msg = "Backup for file not present"
                        raise RuntimeError(msg) from exc_val

                    # restore file backup
                    if ctx != "r":
                        restore_writer = file.open("rb+")
                        restore_writer.truncate()
                        restore_writer.write(self.__backup)
                        restore_writer.close()
                        print(f"RESTORED BACKUP FOR FILE: '{file}'")  # noqa: T201

                    self.__backup = None

                if self.__backup is not None:
                    self.__backup = None

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
