import json
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from io import BytesIO
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
    uuid_from_bytes,
    uuid_to_bytes,
)
from content.tagger.utils import merge_dicts
from content.tagger.video_tagger import (
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    ContextType,
    MetadataTags,
    MetadataTagsRead,
    SerializableDict,
    SerializableDictValue,
    TaggerDomain,
    VideoTagger,
    VideoTaggerContextReadable,
    VideoTaggerContextRW,
    VideoTaggerContextWrapperGeneric,
    VideoTaggerContextWriteable,
    uuid_from_str,
    uuid_to_str,
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
INFO_FOURCC: FOURCC = FOURCC(b"INFO")

AUDS_FOURCC: FOURCC = FOURCC(b"auds")
MIDS_FOURCC: FOURCC = FOURCC(b"mids")
TXTS_FOURCC: FOURCC = FOURCC(b"txts")
VIDS_FOURCC: FOURCC = FOURCC(b"vids")

ICMT_FOURCC: FOURCC = FOURCC(b"ICMT")

KNOWN_INFO_SUBCHUNK_FOURCCS: dict[FOURCC, str] = {ICMT_FOURCC: "comment"}

# custom fourcc's

# vld<t> stands for video language detector <type>
VLD_STR_CHUNK_FOURCC = FOURCC(b"vlds")
VLD_JSON_CHUNK_FOURCC = FOURCC(b"vldj")
VLD_LIST_FOURCC = FOURCC(b"vldl")
VLD_UUID_FOURCC = FOURCC(b"vldu")
VLD_KEY_VALUE_FOURCC = FOURCC(b"vldk")


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
                new_size - 8,
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
        return f"<AVIList parent: {AVIChunk.__str__(self)} type: {self.type}>"

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
        return f"<AVIStreamHeader parent: {AVIChunk.__str__(self)} type: {self.type}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class VLDStrChunk(AVIChunk, FinalAVIChunk):
    value: str

    def __init__(self: Self, parent: AVIChunk, value: str) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.value = value

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: AVIChunk,
    ) -> "VLDStrChunk":
        # custom chunk
        # just contains some string data

        with io.r_ctx(force_entire_read=True) as f:
            value_raw = f.read(parent.span.payload_span.size)

            value = value_raw.decode()

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return VLDStrChunk(parent, value)

    @staticmethod
    def read(io: BoundedIO) -> "VLDStrChunk":
        chunk = AVIChunk.read_avi_chunk(io)
        return VLDStrChunk.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_checked(io: BoundedIO) -> "VLDStrChunk":
        chunk = AVIChunk.read_avi_chunk(io)
        if chunk.fourcc != VLD_STR_CHUNK_FOURCC:
            msg = f"Invalid VLDStrChunk fourcc: {chunk.fourcc}"
            raise RuntimeError(msg)
        return VLDStrChunk.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: AVIChunk) -> "VLDStrChunk":
        return VLDStrChunk.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        value: str,
    ) -> bytes:
        final_data: bytes = value.encode()

        return AVIChunk.write_to_buffer_avi_chunk(VLD_STR_CHUNK_FOURCC, final_data)

    def __str__(self: Self) -> str:
        return f"<VLDStrChunk parent: {AVIChunk.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


ChunkJsonValue = SerializableDict | SerializableDictValue


@final
class VLDJsonChunk(AVIChunk, FinalAVIChunk):
    data: ChunkJsonValue

    def __init__(
        self: Self,
        parent: AVIChunk,
        data: ChunkJsonValue,
    ) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.data = data

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: AVIChunk,
    ) -> "VLDJsonChunk":
        # custom chunk
        # just contains a json payload

        with io.r_ctx(force_entire_read=True) as f:
            data_raw = f.read(parent.span.payload_span.size)

            data = json.loads(data_raw.decode())

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return VLDJsonChunk(parent, data)

    @staticmethod
    def read(io: BoundedIO) -> "VLDJsonChunk":
        chunk = AVIChunk.read_avi_chunk(io)
        return VLDJsonChunk.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: AVIChunk) -> "VLDJsonChunk":
        return VLDJsonChunk.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        data: ChunkJsonValue,
    ) -> bytes:
        byte_data: bytes = json.dumps(data).encode()

        return AVIChunk.write_to_buffer_avi_chunk(VLD_JSON_CHUNK_FOURCC, byte_data)

    def __str__(self: Self) -> str:
        return f"<VLDJsonChunk parent: {AVIChunk.__str__(self)} data: {self.data}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class VLDUUIDChunk(AVIChunk, FinalAVIChunk):
    uuid: UUID

    def __init__(self: Self, parent: AVIChunk, uuid: UUID) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.uuid = uuid

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: AVIChunk,
    ) -> "VLDUUIDChunk":
        # custom chunk
        # just contains an UUID

        with io.r_ctx(force_entire_read=True) as f:
            if parent.span.payload_span.size != 16:
                msg = f"VLDUUIDChunk has not the correct payload size: {parent.span.payload_span.size}"
                raise RuntimeError(msg)

            uuid_raw = f.read(16)

            uuid = uuid_from_bytes(AVI_BYTE_ORDER, uuid_raw)

            parent.span.add_header(16)
            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return VLDUUIDChunk(parent, uuid)

    @staticmethod
    def read(io: BoundedIO) -> "VLDUUIDChunk":
        chunk = AVIChunk.read_avi_chunk(io)
        return VLDUUIDChunk.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: AVIChunk) -> "VLDUUIDChunk":
        return VLDUUIDChunk.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        uuid: UUID,
    ) -> bytes:
        data: bytes = uuid_to_bytes(AVI_BYTE_ORDER, uuid)

        return AVIChunk.write_to_buffer_avi_chunk(VLD_UUID_FOURCC, data)

    def __str__(self: Self) -> str:
        return f"<VLDUUIDChunk parent: {AVIChunk.__str__(self)} uuid: {self.uuid}>"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass
class VLDKeyValueValueStr:
    value: str


@dataclass
class VLDKeyValueValueUUID:
    uuid: UUID


@dataclass
class VLDKeyValueValueJSON:
    data: ChunkJsonValue


VLDKeyValueValue = VLDKeyValueValueStr | VLDKeyValueValueUUID | VLDKeyValueValueJSON


@final
class VLDKeyValueChunk(AVIChunk, FinalAVIChunk):
    key: str
    value: VLDKeyValueValue

    def __init__(
        self: Self,
        parent: AVIChunk,
        key: str,
        value: VLDKeyValueValue,
    ) -> None:
        super().__init__(parent.fourcc, parent.span, is_list=False)

        self.key = key
        self.value = value

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: AVIChunk,
    ) -> "VLDKeyValueChunk":
        # custom chunk
        # contains a key and a value

        # format:
        # one VLDStrChunk
        # the sub chunk, can be any of the available chunk types

        key_chunk = VLDStrChunk.read_checked(parent.payload_io(io))

        parent.span.add_header(key_chunk.span.total.size)

        sub_chunk = AVIChunk.read_avi_chunk(parent.payload_io(io))

        value: VLDKeyValueValue
        value_chunk: AVIChunk

        match sub_chunk.fourcc.value:
            case SupportedChunks.VLD_STR:
                value_chunk = VLDStrChunk.read_from_parent(
                    sub_chunk.payload_io(io),
                    sub_chunk,
                )
                value = VLDKeyValueValueStr(value_chunk.value)
            case SupportedChunks.VLD_JSON:
                value_chunk = VLDJsonChunk.read_from_parent(
                    sub_chunk.payload_io(io),
                    sub_chunk,
                )
                value = VLDKeyValueValueJSON(value_chunk.data)
            case SupportedChunks.VLD_UUID:
                value_chunk = VLDUUIDChunk.read_from_parent(
                    sub_chunk.payload_io(io),
                    sub_chunk,
                )
                value = VLDKeyValueValueUUID(value_chunk.uuid)
            case _:
                msg = f"Invalid sub chunk in VLDKeyValueChunk: {sub_chunk.fourcc}"
                raise RuntimeError(msg)

        parent.span.add_header(value_chunk.span.total.size)

        if parent.span.payload_span.size != 0:
            msg = f"AppleItunesItemBox isn't fully filled by the data box: {parent.span.payload_span.size} leftover data"
            raise RuntimeError(msg)

        return VLDKeyValueChunk(parent, key_chunk.value, value)

    @staticmethod
    def read(io: BoundedIO) -> "VLDKeyValueChunk":
        chunk = AVIChunk.read_avi_chunk(io)
        return VLDKeyValueChunk.__read_impl(chunk.payload_io(io), chunk)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: AVIChunk) -> "VLDKeyValueChunk":
        return VLDKeyValueChunk.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        key: str,
        value: VLDKeyValueValue,
    ) -> bytes:
        buf = BytesIO()

        key_bytes = VLDStrChunk.write_to_buffer(
            key,
        )

        buf.write(key_bytes)

        data_bytes: bytes

        if isinstance(value, VLDKeyValueValueStr):
            data_bytes = VLDStrChunk.write_to_buffer(value.value)
        elif isinstance(value, VLDKeyValueValueJSON):
            data_bytes = VLDJsonChunk.write_to_buffer(value.data)
        elif isinstance(value, VLDKeyValueValueUUID):
            data_bytes = VLDUUIDChunk.write_to_buffer(value.uuid)
        else:
            assert_never(value)

        buf.write(data_bytes)

        final_data = buf.getvalue()

        return AVIChunk.write_to_buffer_avi_chunk(VLD_KEY_VALUE_FOURCC, final_data)

    def __str__(self: Self) -> str:
        return f"<VLDKeyValueChunk parent: {AVIChunk.__str__(self)} key: {self.key} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


class SupportedChunks:
    RIFF = RIFF_FOURCC
    LIST = LIST_FOURCC

    STRH = STRH_FOURCC

    VLD_STR = VLD_STR_CHUNK_FOURCC
    VLD_JSON = VLD_JSON_CHUNK_FOURCC
    VLD_UUID = VLD_UUID_FOURCC
    VLD_KEY_VALUE = VLD_KEY_VALUE_FOURCC


def read_chunk(io: BoundedIO) -> AVIChunk:
    chunk = AVIChunk.read_avi_chunk(io)

    match chunk.fourcc.value:
        case SupportedChunks.RIFF:
            return AVIList.read_avi_list_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.LIST:
            return AVIList.read_avi_list_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.STRH:
            return AVIStreamHeader.read_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.VLD_STR:
            return VLDStrChunk.read_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.VLD_JSON:
            return VLDJsonChunk.read_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.VLD_UUID:
            return VLDUUIDChunk.read_from_parent(chunk.payload_io(io), chunk)
        case SupportedChunks.VLD_KEY_VALUE:
            return VLDKeyValueChunk.read_from_parent(chunk.payload_io(io), chunk)
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


@dataclass
class VLDCustomSubChunk:
    data: VLDKeyValueValue


@dataclass
class VLDKnownStrSubChunk:
    fourcc: FOURCC
    value: str


@dataclass
class VLDCustomKeyValueEntry:
    key: str
    data: VLDKeyValueValue


@dataclass
class VLDUnknownStrSubChunk:
    fourcc: FOURCC
    data: bytes


VLDSubChunkType = (
    VLDCustomSubChunk
    | VLDKnownStrSubChunk
    | VLDUnknownStrSubChunk
    | VLDCustomKeyValueEntry
)


class INFOChunkBuilder:
    __sub_chunks: dict[FOURCC | str, VLDSubChunkType]

    def __init__(self: Self) -> None:
        self.__sub_chunks = {}

    @staticmethod
    def _key_str_impl(sub_chunk: VLDSubChunkType) -> FOURCC | str:
        # note: this is never serialized, it is only to detect duplicates in internal regeneration from an old info chunk, so this doesn't have to match the serialization behavior, but it's close, as the string is unique then

        if isinstance(sub_chunk, VLDKnownStrSubChunk):
            return sub_chunk.fourcc

        if isinstance(sub_chunk, VLDUnknownStrSubChunk):
            return sub_chunk.fourcc

        if isinstance(sub_chunk, VLDCustomSubChunk):
            if isinstance(sub_chunk.data, VLDKeyValueValueStr):
                return VLD_STR_CHUNK_FOURCC
            if isinstance(sub_chunk.data, VLDKeyValueValueUUID):
                return VLD_UUID_FOURCC
            if isinstance(sub_chunk.data, VLDKeyValueValueJSON):
                return VLD_JSON_CHUNK_FOURCC

            assert_never(sub_chunk.data)

        if isinstance(sub_chunk, VLDCustomKeyValueEntry):
            return sub_chunk.key

        assert_never(sub_chunk)

    def add_sub_chunk(
        self: Self,
        sub_chunk: VLDSubChunkType,
        duplicate_behavior: Literal["overwrite", "error", "ignore"],
    ) -> None:

        key = INFOChunkBuilder._key_str_impl(sub_chunk)

        if self.__sub_chunks.get(key, None) is not None:
            if duplicate_behavior == "error":
                msg: str = f"Trying to add duplicate sub chunk key: {key}"
                raise RuntimeError(msg)

            if duplicate_behavior == "overwrite":
                self.__sub_chunks[key] = sub_chunk
            elif duplicate_behavior == "ignore":
                pass
            else:
                assert_never(duplicate_behavior)
        else:
            self.__sub_chunks[key] = sub_chunk

    @staticmethod
    def __render_sub_chunk_impl(
        sub_chunk: VLDCustomSubChunk | VLDKnownStrSubChunk | VLDUnknownStrSubChunk,
    ) -> bytes:
        if isinstance(sub_chunk, VLDKnownStrSubChunk):
            return AVIChunk.write_to_buffer_avi_chunk(
                sub_chunk.fourcc,
                sub_chunk.value.encode(),
            )
        if isinstance(sub_chunk, VLDUnknownStrSubChunk):
            return AVIChunk.write_to_buffer_avi_chunk(
                sub_chunk.fourcc,
                sub_chunk.data,
            )

        if isinstance(sub_chunk, VLDCustomSubChunk):
            if isinstance(sub_chunk.data, VLDKeyValueValueStr):
                return VLDStrChunk.write_to_buffer(sub_chunk.data.value)
            if isinstance(sub_chunk.data, VLDKeyValueValueUUID):
                return VLDUUIDChunk.write_to_buffer(sub_chunk.data.uuid)
            if isinstance(sub_chunk.data, VLDKeyValueValueJSON):
                return VLDJsonChunk.write_to_buffer(sub_chunk.data.data)

            assert_never(sub_chunk.data)

        assert_never(sub_chunk)

    def build(self: Self) -> bytes:

        sub_chunk_data: list[bytes] = [
            VLDKeyValueChunk.write_to_buffer(
                INFO_LIST_ID_NAME,
                VLDKeyValueValueUUID(INFO_LIST_ID_NAME_ID),
            ),
        ]

        list_data: dict[str, VLDKeyValueValue] = {}

        for sub_chunk in self.__sub_chunks.values():
            if isinstance(sub_chunk, VLDCustomKeyValueEntry):
                list_data = merge_dicts(
                    list_data,
                    {sub_chunk.key: sub_chunk.data},
                    "error",
                )
            else:
                sub_chunk_data.append(self.__render_sub_chunk_impl(sub_chunk))

        if len(list_data) != 0:
            sub_chunk_data.append(
                AVIList.write_to_buffer_avi_list(
                    LIST_FOURCC,
                    VLD_LIST_FOURCC,
                    [
                        VLDKeyValueChunk.write_to_buffer(key=key, value=value)
                        for key, value in list_data.items()
                    ],
                ),
            )

        return AVIList.write_to_buffer_avi_list(
            LIST_FOURCC,
            INFO_FOURCC,
            sub_chunk_data,
        )


INFO_LIST_ID_NAME: str = "id:video_language_detect:info_list_name"

INFO_LIST_ID_NAME_ID: UUID = UUID(hex="90e175d1-efdb-4144-a214-ebfab6258be0")


ReadInfoChunkValue = (
    VLDKnownStrSubChunk | VLDCustomKeyValueEntry | VLDUnknownStrSubChunk
)

ReadInfoChunkValues = list[ReadInfoChunkValue]

InfoValues = Optional[ReadInfoChunkValues]


@dataclass
class ReadMetadataImpl:
    metadata: SerializableDict
    uuid: Optional[UUID]
    unrecognized: list[VLDUnknownStrSubChunk]


class AVIMetadataHandler:
    __info_values: InfoValues
    __our_chunks: list[AVIChunk]

    def __init__(
        self: Self,
        info_values: InfoValues,
        our_chunks: list[AVIChunk],
    ) -> None:
        self.__info_values = info_values
        self.__our_chunks = our_chunks

    def remove_old_metadata(self: Self, f: BinaryIO) -> None:
        # delete old metadata
        if len(self.__our_chunks) != 0:
            f.truncate(self.__our_chunks[0].span.total.start)

    def __write_metadata_toplevel_info(
        self: Self,
        f: BinaryIO,
        tags: MetadataTags,
    ) -> None:
        # NOTE: using top level INFO chunks

        # they can appear unlimited times, mots readers just use values from sub-chunks, and then the last one they encounter

        info_chunk = INFOChunkBuilder()

        if self.__info_values is not None:
            # restore the old chunks
            for info_value in self.__info_values:
                info_chunk.add_sub_chunk(
                    info_value,
                    duplicate_behavior="error",
                )

        # add or overwrite chunks, if not present, so that the new data gets written all the time, except uuid, that is never replaced
        info_chunk.add_sub_chunk(
            VLDKnownStrSubChunk(
                fourcc=ICMT_FOURCC,
                value=tags.comment,
            ),
            duplicate_behavior="overwrite",
        )

        for key, value in tags.metadata.items():
            info_chunk.add_sub_chunk(
                VLDCustomKeyValueEntry(
                    key=key,
                    data=VLDKeyValueValueJSON(data=value),
                ),
                duplicate_behavior="overwrite",
            )

        # Note, these are not necessary in sync, which is bad, but that should never happen
        info_chunk.add_sub_chunk(
            VLDCustomKeyValueEntry(
                key=TaggerDomain.UUID_RAW_KEY_FREEFORM.name,
                data=VLDKeyValueValueUUID(uuid=tags.uuid),
            ),
            duplicate_behavior="ignore",
        )

        info_chunk.add_sub_chunk(
            VLDCustomKeyValueEntry(
                key=TaggerDomain.UUID_HEX_KEY_FREEFORM.name,
                data=VLDKeyValueValueStr(value=uuid_to_str(tags.uuid)),
            ),
            duplicate_behavior="ignore",
        )

        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)

        top_level_chunks = list(avi_iter_chunks(f, SimpleSpan(0, filesize)))

        if len(top_level_chunks) != 1:
            msg = f"Expected only one RIFF top level chunk, but got {len(top_level_chunks)}"
            raise RuntimeError(msg)

        top_level_chunk = top_level_chunks[0]

        if not isinstance(top_level_chunk, AVIList):
            msg = f"Expected one LIST chunk at the top level, but got {top_level_chunk}"
            raise TypeError(msg)

        if top_level_chunk.fourcc != RIFF_FOURCC:
            msg = f"Expected a RIFF top level chunk, but got {top_level_chunk.fourcc}"
            raise TypeError(msg)

        info_list_data = info_chunk.build()
        top_level_chunk.add_content_afterwards_avi_list(f, info_list_data)

        f.flush()

    def write_new_metadata(self: Self, f: BinaryIO, tags: MetadataTags) -> None:
        f.seek(0, 2)

        self.__write_metadata_toplevel_info(f, tags)

        f.flush()

    def read_metadata(
        self: Self,
        f: BinaryIO,
    ) -> ReadMetadataImpl:
        info_chunk: Optional[AVIList] = None

        for chunk in self.__our_chunks:
            if chunk.fourcc == LIST_FOURCC:
                if not isinstance(chunk, AVIList):
                    msg = "Invalid AVIList: type not dispatched to correct class"
                    raise TypeError(msg)

                if chunk.type == INFO_FOURCC:
                    if info_chunk is not None:
                        msg = f"Duplicate 'INFO' chunk from us at the top level, only one allowed: {chunk}"
                        raise RuntimeError(msg)

                    info_chunk = chunk
                    continue

            msg = f"Invalid chunk for metadata found: {type(chunk)} {chunk}"
            raise TypeError(msg)

        if info_chunk is None:
            return ReadMetadataImpl({}, None, [])

        result_toplevel_info = self.__read_info_chunk_info(info_chunk, f)

        result = ReadMetadataImpl({}, None, [])

        for value in result_toplevel_info:
            if isinstance(value, VLDKnownStrSubChunk):
                if (
                    KNOWN_INFO_SUBCHUNK_FOURCCS.get(value.fourcc, None)  # noqa: SIM910
                    is None
                ):
                    msg = f"Invalid VLDKnownStrSubChunk chunk, can't be transformed to a string key: {value.fourcc}"
                    raise RuntimeError(msg)

                result.metadata = merge_dicts(
                    result.metadata,
                    {KNOWN_INFO_SUBCHUNK_FOURCCS[value.fourcc]: value.value},
                    "error",
                )
            elif isinstance(value, VLDUnknownStrSubChunk):
                result.unrecognized.append(value)
            elif isinstance(value, VLDCustomKeyValueEntry):
                if value.key in [
                    TaggerDomain.UUID_RAW_KEY_FREEFORM.name,
                    TaggerDomain.UUID_HEX_KEY_FREEFORM.name,
                ]:
                    uuid: UUID

                    if value.key == TaggerDomain.UUID_RAW_KEY_FREEFORM.name:
                        if not isinstance(value.data, VLDKeyValueValueUUID):
                            msg = f"Invalid uuid (raw) key type: {type(value.data)} {value.data}"
                            raise RuntimeError(msg)

                        uuid = value.data.uuid
                    else:
                        if not isinstance(value.data, VLDKeyValueValueStr):
                            msg = f"Invalid uuid (str) key type: {type(value.data)} {value.data}"
                            raise RuntimeError(msg)

                        uuid = uuid_from_str(value.data.value)

                    if result.uuid is not None:
                        if result.uuid != uuid:
                            msg = f"Duplicate uuid tag read, that are not the same: {uuid}"
                            raise RuntimeError(msg)
                    else:
                        result.uuid = uuid
                else:
                    result.metadata = merge_dicts(
                        result.metadata,
                        {
                            "metadata": merge_dicts(
                                cast(
                                    dict[str, Any],
                                    result.metadata.get("metadata", {}),
                                ),
                                {value.key: value.data},
                                "error",
                            ),
                        },
                        "overwrite",
                    )

            else:
                assert_never(value)

        return result

    @staticmethod
    def __read_info_chunk_info(
        info_chunk: AVIList,
        f: BinaryIO,
    ) -> ReadInfoChunkValues:

        result: ReadInfoChunkValues = []

        for chunk in avi_iter_chunks(f, info_chunk.span.payload_span):
            if isinstance(chunk, VLDKeyValueChunk):
                if chunk.key == INFO_LIST_ID_NAME:
                    if chunk.value != VLDKeyValueValueUUID(INFO_LIST_ID_NAME_ID):
                        msg = f"Invalid list id name for our INFO list identifier: {chunk.value}"
                        raise RuntimeError(msg)

                    continue

                msg = f"Invalid VLDKeyValueChunk: not allowed at the top level: {chunk}"
                raise RuntimeError(msg)
            elif isinstance(  # noqa: RET506
                chunk,
                (VLDStrChunk, VLDJsonChunk, VLDUUIDChunk),
            ):
                msg = f"Invalid Vld<Value>Chunk: not allowed at the top level: {chunk}"
                raise TypeError(msg)
            elif isinstance(chunk, AVIList):
                if chunk.fourcc != LIST_FOURCC or chunk.type != VLD_LIST_FOURCC:
                    msg = f"Invalid list inside INFO list: {chunk}"
                    raise RuntimeError(msg)

                for sub_chunk in avi_iter_chunks(f, chunk.span.payload_span):
                    if isinstance(sub_chunk, VLDKeyValueChunk):
                        result.append(
                            VLDCustomKeyValueEntry(
                                key=sub_chunk.key,
                                data=sub_chunk.value,
                            ),
                        )
                    elif isinstance(
                        sub_chunk,
                        (VLDStrChunk, VLDJsonChunk, VLDUUIDChunk),
                    ):
                        msg = f"Invalid Vld<Value>Chunk: not allowed at the vld list level: {sub_chunk}"
                        raise TypeError(msg)
                    elif isinstance(sub_chunk, AVIList):
                        msg = f"Invalid AVI list: not allowed at the vld list level: {sub_chunk}"
                        raise TypeError(msg)
                    else:
                        msg = f"Invalid Normal AVI chunk: not allowed at the vld list level: {sub_chunk}"
                        raise TypeError(msg)

            else:
                data_raw: bytes
                with BoundedIO.get_new(f, chunk.span.payload_span).r_ctx(
                    force_entire_read=True,
                ):
                    data_raw = f.read(chunk.span.payload_span.size)

                if chunk.fourcc in KNOWN_INFO_SUBCHUNK_FOURCCS:
                    result.append(
                        VLDKnownStrSubChunk(
                            fourcc=chunk.fourcc,
                            value=data_raw.decode(),
                        ),
                    )
                else:
                    result.append(
                        VLDUnknownStrSubChunk(fourcc=chunk.fourcc, data=data_raw),
                    )

        return result

    @staticmethod
    def get_metadata_handler(
        f: BinaryIO,
    ) -> "AVIMetadataHandler":

        info_values: InfoValues = None

        def info_chunk_is_written_by_us(chunk: AVIList) -> bool:
            nonlocal info_values

            children_chunks = list(avi_iter_chunks(f, chunk.span.payload_span))

            is_our_chunk = False

            for children_chunk in children_chunks:
                if is_our_chunk:
                    break

                if children_chunk.fourcc == VLD_KEY_VALUE_FOURCC:
                    if not isinstance(children_chunk, VLDKeyValueChunk):
                        msg = "Invalid VLDKeyValueChunk: type not dispatched to correct class"
                        raise TypeError(msg)

                    if (
                        children_chunk.key == INFO_LIST_ID_NAME
                        and children_chunk.value
                        == VLDKeyValueValueUUID(INFO_LIST_ID_NAME_ID)
                    ):
                        is_our_chunk = True

            if not is_our_chunk:
                return False

            if info_values is not None:
                msg = f"Duplicate 'INFO' chunk from us at the top level, only one allowed: {chunk}"
                raise RuntimeError(msg)

            info_values = AVIMetadataHandler.__read_info_chunk_info(chunk, f)
            return True

        def chunk_is_written_by_us(chunk: AVIChunk) -> bool:
            if chunk.fourcc == LIST_FOURCC:
                if not isinstance(chunk, AVIList):
                    msg = "Invalid AVIList: type not dispatched to correct class"
                    raise TypeError(msg)

                if chunk.type == INFO_FOURCC:
                    return info_chunk_is_written_by_us(chunk)

                return False

            return False

        f.seek(0, 2)
        filesize = f.tell()

        f.seek(0)

        top_level_chunks = list(avi_iter_chunks(f, SimpleSpan(0, filesize)))

        if len(top_level_chunks) != 1:
            msg = f"Expected only one RIFF top level chunk, but got {len(top_level_chunks)}"
            raise RuntimeError(msg)

        top_level_chunk = top_level_chunks[0]

        if not isinstance(top_level_chunk, AVIList):
            msg = f"Expected one LIST chunk at the top level, but got {top_level_chunk}"
            raise TypeError(msg)

        if top_level_chunk.fourcc != RIFF_FOURCC:
            msg = f"Expected a RIFF top level chunk, but got {top_level_chunk.fourcc}"
            raise TypeError(msg)

        top_children_chunks: list[AVIChunk] = list(
            avi_iter_chunks(f, top_level_chunk.span.payload_span),
        )

        our_chunks_reversed: list[AVIChunk] = []
        other_chunk_encountered = False
        for chunk in reversed(top_children_chunks):
            if other_chunk_encountered:
                break

            if chunk_is_written_by_us(chunk):
                our_chunks_reversed.append(chunk)
            else:
                other_chunk_encountered = True
                break

        return AVIMetadataHandler(
            info_values,
            list(reversed(our_chunks_reversed)),
        )


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

        # TODO. replace VIDEO_FILE_TAG_UPDATE_BAR_FORMAT everywhere, as we don't use bytes here!
        bar: CounterInterface = self.manager.counter(
            total=float(3),
            desc="update mp4 metadata tags",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        try:
            avi_metadata_handler = AVIMetadataHandler.get_metadata_handler(
                f=self.__writer,
            )

            bar.update(1, force=True)

            avi_metadata_handler.remove_old_metadata(self.__writer)

            bar.update(1, force=True)

            avi_metadata_handler.write_new_metadata(
                self.__writer,
                tags,
            )

            bar.update(1, force=True)

            self.__writer.flush()
        finally:
            bar.close(clear=True)

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

    @override
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead:
        def decode_as_str(value: Any) -> str:
            if not isinstance(value, str):
                msg = f"Invalid type in decode_as_str: {type(value)}"
                raise TypeError(msg)

            return value

        def decode_as_dict(value: Any) -> SerializableDict:
            if not isinstance(value, dict):
                msg = f"Invalid type in decode_as_dict: {type(value)}"
                raise TypeError(msg)

            return value

        avi_metadata_handler = AVIMetadataHandler.get_metadata_handler(
            f=self.__writer,
        )

        metadata_result = avi_metadata_handler.read_metadata(self.__writer)

        result: MetadataTagsRead = MetadataTagsRead(None, None, {}, [])

        if metadata_result.uuid is not None:
            result.uuid = metadata_result.uuid

        for key, value in metadata_result.metadata.items():

            if key == "comment":
                if result.comment is not None:
                    msg = f"Duplicate comment tag read: {value}"
                    raise RuntimeError(msg)

                result.comment = decode_as_str(value)

            elif key == "metadata":
                if len(result.metadata.items()) != 0:
                    msg = f"Duplicate metadata tag read: {value}"
                    raise RuntimeError(msg)

                result.metadata = decode_as_dict(value)
            else:
                result.unrecognized.append(
                    (key, decode_as_str(value)),
                )

        return result


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
