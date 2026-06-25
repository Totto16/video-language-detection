from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Optional, Self, final, override

from content.tagger.parser import BoundedIO, SimpleSpan
from content.tagger.video_tagger import (
    ContextType,
    InspectNotImplemented,
    InspectPrinter,
    InspectPriority,
    VideoTagger,
    VideoTaggerContextCtxGeneric,
    VideoTaggerContextReadable,
    VideoTaggerContextRW,
    VideoTaggerContextWriteable,
)
from helper.decorator import decorate_class
from helper.manager import ManagerInterface
from helper.result import Err, Ok, Result


class MKVDecodeType(Enum):
    Check = "check"
    Normal = "normal"


@dataclass(slots=True, repr=True)
class MKVDecodeOptions:
    strict: bool
    type: MKVDecodeType

    @staticmethod
    def default() -> "MKVDecodeOptions":
        return MKVDecodeOptions(strict=True, type=MKVDecodeType.Normal)


@decorate_class(slots=True)
class FinalEBMLElement:
    __final__ebml_element__ = True


@decorate_class(slots=True)
class NonFinalEBMLElement:
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        is_final = getattr(cls, "__final__ebml_element__", False)

        if not is_final:
            for fn_name in [
                "read",
                "read_from_parent",
            ]:
                if fn_name in cls.__dict__:
                    msg = f"{cls.__name__} defines {fn_name}(), but only final classes may do so"
                    raise TypeError(msg)


@dataclass(slots=True, repr=True)
class Bit:
    value: bool

    @property
    def int(self: Self) -> int:
        return 1 if self.value else 0


@decorate_class(slots=True)
class BitIterator(Iterator[Bit]):
    __data: bytes
    __bit_index: int

    def __init__(self: Self, val: bytes) -> None:
        super().__init__()

        self.__data = val
        self.__bit_index = 0

    @override
    def __next__(self: Self) -> Bit:
        if self.__bit_index >= (8 * len(self.__data)):
            raise StopIteration

        byte_idx = self.__bit_index // 8
        bit_idx = self.__bit_index % 8

        bit = (self.__data[byte_idx] >> (7 - bit_idx)) & 0x01

        if bit not in [0, 1]:
            msg: str = f"Invalid bit calculated: {bit}"
            raise ValueError(msg)

        self.__bit_index = self.__bit_index + 1

        return Bit(bit != 0)

    @property
    def bit_index(self: Self) -> int:
        return self.__bit_index


@decorate_class(slots=True)
class EBMLVarInt:
    __value: int

    def __init__(self: Self, value: int) -> None:
        self.__value = value

    @staticmethod
    def from_io(io: BoundedIO) -> Result[tuple["EBMLVarInt", int], str]:

        with io.r_ctx(force_entire_read=False) as f:

            first_byte = f.read(1)

            if first_byte == b"\x00":
                return Err("The first byte of a VarInt can't be 0x00")

            bits = BitIterator(first_byte)

            for b in bits:
                if b.value:
                    break

            varint_byte_length = bits.bit_index

            if varint_byte_length > 8:
                return Err(
                    f"The final byte length of a VarInt can't be > 8: {varint_byte_length}",
                )

            rest: bytes = b""
            if varint_byte_length > 1:
                rest = f.read(varint_byte_length - 1)

            byte_value = (
                bytes(first_byte[0] & ((1 << (8 - varint_byte_length)) - 1)) + rest
            )

            int_value = int.from_bytes(byte_value, byteorder="big", signed=False)

            return Ok(
                (
                    EBMLVarInt(
                        int_value,
                    ),
                    varint_byte_length,
                ),
            )


@final
@decorate_class(slots=True)
class EBMLElementSpan:
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
            msg = f"Invalid element: size too small: {self.__total.size}"
            raise RuntimeError(msg)

        if self.__total.size < header_size:
            msg = f"Invalid chunk size {self.__total.size} at {self.__total.start}"
            raise RuntimeError(msg)

    varIntTimes2 = "TODO"

    @staticmethod
    def from_ebml_specified_size(
        span: SimpleSpan, header_size: int, todo: varIntTimes2  # size + id?
    ) -> "EBMLElementSpan":
        return EBMLElementSpan(
            SimpleSpan(span.start, span.size + len(todo)), header_size
        )

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

        span = SimpleSpan(interval_start, interval_size)
        if (span.start % 2) != 0 and span.size != 0:
            msg = f"SimpleSpan for AVI is not aligned by the WORD (16 bit): {span}"
            raise RuntimeError(msg)

        return span

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

    def validate_truncation(self: Self, new_total_size: int) -> None:
        # NOTE: ATM only supported for filelevel spans, alias starting at 0!
        if self.__total.start != 0:
            msg = f"validate_truncation only supported for spans starting at 0, but this starts at {self.__total.start }"
            raise RuntimeError(msg)

        last_span = self.__interval_span_impl(len(self.__intervals))

        if last_span.start > new_total_size:
            msg = f"truncation would remove too much of the size: {last_span.start} > {new_total_size}"
            raise RuntimeError(msg)

        self.__total = self.__total.sub_span(new_total_size)

    def __str__(self: Self) -> str:
        header_string = ", ".join(
            str(self.header_span(i)) for i in range(len(self.__intervals))
        )
        return f"<EBMLElementSpan total: {self.__total} header: [ {header_string} ] payload: {self.payload_span}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
class EBMLElement(NonFinalEBMLElement):
    element_id: EBMLVarInt
    span: EBMLElementSpan

    def __init__(
        self: Self,
        element_id: EBMLVarInt,
        span: EBMLElementSpan,
    ) -> None:
        self.element_id = element_id
        self.span = span

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

            if size + 8 > io.span.size:
                msg = f"Invalid AVI Chunk size: It overflows the parent chunk: {size+ 8} > {io.span.size}"
                raise RuntimeError(msg)

            span = EBMLElementSpan.from_avi_specified_size(
                io.span.sub_span(size),
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
        return f"<EBMLElement element_id: {self.element_id} span: {self.span}>"

    def __repr__(self: Self) -> str:
        return str(self)


class EBMLHeader(EBMLElement):
    pass

    def __parse():
        pass
        # The EBML Header MUST contain a single Master Element with an Element Name of EBML and
        # Element ID of 0x1A45DFA3 (see Section 11.2.1); the Master Element may have any number of
        # additional EBML Elements within it. The EBML Header of an EBML Document that uses an
        # EBMLVersion of 1 MUST only contain EBML Elements that are deﬁned as part of this document.
        # Elements within an EBML Header can be at most 4 octets long, except for the EBML Element with
        # Element Name EBML and Element ID 0x1A45DFA3 (see Section 11.2.1); this Element can be up to 8
        # octets long.


class EBMLBody(EBMLElement):
    pass


MKV_FOURCC = "TODO"


@decorate_class(slots=True)
class VideoTaggerAVI(VideoTagger):
    __streams: int
    __types: list[MKV_FOURCC]

    def __init__(
        self: Self,
        file: Path,
        streams: int,
        types: list[MKV_FOURCC],
    ) -> None:
        super().__init__(file)
        self.__streams = streams
        self.__types = types

        streams = 0

    @staticmethod
    def get_handle(file: Path) -> Result["VideoTagger", str]:

        options: MKVDecodeOptions = MKVDecodeOptions(
            strict=False,
            type=MKVDecodeType.Check,
        )

        try:

            with file.open("rb") as f:
                mkv_res = is_mkv_file(f)
                if mkv_res is not None:
                    return Err(mkv_res)

                f.seek(0)

                streams = 0
                types: list[MKV_FOURCC] = [AUDIO_TODO, VIDEO_TODO]

                # read the file, so that we check if we can parse it correctly and that it is an avi file
                for strh in find_strh_chunks_with_type(f, types, options):
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

        @decorate_class(slots=True)
        class VideoTaggerContextCtx(VideoTaggerContextCtxGeneric):

            def __init__(self: Self) -> None:
                super().__init__(file, ctx, manager)

            @override
            def get_context(
                self: Self,
                manager: ManagerInterface,
                writer: BinaryIO,
            ) -> VideoTaggerContextMKV:
                return VideoTaggerContextMKV(manager, file, writer, streams, types)

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

        def is_data_chunk(chunk: AVIChunk) -> bool:
            fourcc = chunk.fourcc

            data_type = fourcc.value[2:4]

            if data_type not in [b"dc", b"wb", "tx"]:
                return False

            return all(bytes([c]).isdigit() for c in fourcc.value[0:2])

        def print_chunk(chunk: AVIChunk, *, depth: int) -> None:

            local_priority = (
                InspectPriority.Important
                if chunk.is_list
                else (
                    InspectPriority.Ignore
                    if is_data_chunk(chunk)
                    else InspectPriority.Normal
                )
            )

            if local_priority.as_int() > priority.as_int():
                return

            name: str = f"{chunk.fourcc}"
            if isinstance(chunk, AVIList):
                name = f"{chunk.fourcc}({chunk.type})"

            element = InspectElement(name, size=chunk.span.total.size)

            printer.element(element, depth)

        with self.file.open(mode="rb") as f:

            def iterate_chunks_recursive(span: SimpleSpan, *, depth: int) -> None:
                for chunk in avi_iter_chunks(f, span):

                    print_chunk(chunk, depth=depth)

                    if (
                        chunk.fourcc == LIST_FOURCC
                        and isinstance(chunk, AVIList)
                        and chunk.type == MOVI_FOURCC
                        and priority.as_int() <= InspectPriority.Normal.as_int()
                    ):
                        # skip movi chunk with maaaany data chunks, but nothing interesting
                        continue

                    if chunk.is_list:
                        iterate_chunks_recursive(
                            chunk.span.payload_span,
                            depth=depth + 1,
                        )

            f.seek(0, 2)
            filesize = f.tell()

            printer.start()
            iterate_chunks_recursive(SimpleSpan(0, filesize), depth=0)
            printer.end()

        return None
