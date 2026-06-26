from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional, Self, final, override

from content.tagger.parser import (
    BoundedIO,
    ByteOrder,
    Float32,
    Float64,
    SimpleSpan,
    Unpacker,
)
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


@dataclass(slots=True, repr=True)
class EBMLDecodeOptions:
    max_id_length: int
    max_size_length: int


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

        if io.span.size < 1:
            return Err("Not enough data for VarInt")

        with io.r_ctx(force_entire_read=False) as f:

            first_byte = f.read(1)

            if first_byte == b"\x00":
                return Err("The first byte of a VarInt can't be 0x00")

            bits = BitIterator(first_byte)

            found_end = False

            for b in bits:
                if b.value:
                    found_end = True
                    break

            if not found_end:
                return Err("more than 8 byte VarInt not supported atm")

            varint_byte_length = bits.bit_index

            if varint_byte_length > 8:
                return Err(
                    f"The final byte length of a VarInt can't be > 8: {varint_byte_length}",
                )

            rest: bytes = b""
            if varint_byte_length > 1:
                if io.span.size < varint_byte_length:
                    return Err(
                        f"Not enough data for VarInt: need {varint_byte_length} bytes but got {io.span.size}",
                    )

                rest = f.read(varint_byte_length - 1)

            byte_value = (
                bytes([first_byte[0] & ((1 << (8 - varint_byte_length)) - 1)]) + rest
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

    def minmum_bytes_required(self: Self) -> int:

        # each byte has 7 bits of information

        acc = self.__value
        amount = 1
        while True:
            acc = acc // 0b10000000

            if acc == 0:
                break

            amount = amount + 1

        return amount

    def is_valid_element_id(self: Self, bytes_used: int) -> Optional[str]:
        # The bits of the
        # VINT_DATA component of the Element ID MUST NOT be all 0 values or all 1 values. The
        # VINT_DATA component of the Element ID MUST be encoded at the shortest valid length. For
        # example, an Element ID with binary encoding of 1011 1111 is valid, whereas an Element ID with
        # binary encoding of 0100 0000 0011 1111 stores a semantically equal VINT_DATA but is invalid,
        # because a shorter VINT encoding is possible. Additionally, an Element ID with binary encoding of
        # 1111 1111 is invalid, since the VINT_DATA section is set to all one values, whereas an Element ID
        # with binary encoding of 0100 0000 0111 1111 stores a semantically equal VINT_DATA and is the
        # shortest-possible VINT encoding.

        if self.__value == 0:
            return "Value is NULL"

        if bytes_used == 0:
            msg = f"IMPLEMENTATION error: bytes_used {bytes_used}"
            raise RuntimeError(msg)

        if self.__value == ((1 << (bytes_used * 7)) - 1):
            return "Value is 0xFF..FF"

        minmum_bytes_required = self.minmum_bytes_required()

        if minmum_bytes_required != bytes_used:
            if minmum_bytes_required > bytes_used:
                msg = f"IMPLEMENTATION ERROR: minmum_bytes_required  calculated incorrectly: {minmum_bytes_required} {self.__value}"
                raise RuntimeError(msg)

            if self.__value == ((1 << ((bytes_used - 1) * 7)) - 1):
                if minmum_bytes_required + 1 == bytes_used:
                    return None
                return f"Value 0xFF..FF encoded incorrectly, must use exactly {minmum_bytes_required +1 } bytes, but used {bytes_used}"

            return f"Value {self.__value} uses too much bytes: {minmum_bytes_required} bytes are the minimum, but used {bytes_used}"

        return None

    @property
    def value(self: Self) -> int:
        return self.__value

    def __str__(self: Self) -> str:
        return f"<VarInt {self.__value}>"

    def __repr__(self: Self) -> str:
        return repr(self.__value)

    def __hash__(self: Self) -> int:
        return hash(("VarInt", self.__value))

    def __int__(self: Self) -> int:
        return self.__value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, EBMLVarInt):
            return self.__value == other.__value

        if isinstance(other, int):
            return self.__value == other

        return False


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
            msg = f"Invalid element size {self.__total.size} at {self.__total.start}"
            raise RuntimeError(msg)

    @staticmethod
    def from_ebml_specified_size(
        span: SimpleSpan,
        header_sizes: tuple[int, int],
    ) -> "EBMLElementSpan":
        header_size = sum(header_sizes)
        return EBMLElementSpan(
            SimpleSpan(span.start, span.size + header_size),
            header_size,
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
        return f"<EBMLElementSpan total: {self.__total} header: [ {header_string} ] payload: {self.payload_span}>"

    def __repr__(self: Self) -> str:
        return str(self)


def vint_max_for_bytes(amount: int) -> int:
    if amount == 0:
        msg = f"Invalid amount: {amount}"
        raise RuntimeError(msg)

    return (1 << (7 * amount)) - 2


VINTMAX: int = vint_max_for_bytes(8)


@decorate_class(slots=True)
class EBMLElement(NonFinalEBMLElement):
    element_id: EBMLVarInt
    span: EBMLElementSpan
    header_sizes: tuple[int, int]
    is_container: bool

    def __init__(
        self: Self,
        element_id: EBMLVarInt,
        span: EBMLElementSpan,
        header_sizes: tuple[int, int],
        *,
        is_container: bool,
    ) -> None:
        self.element_id = element_id
        self.span = span
        self.header_sizes = header_sizes
        self.is_container = is_container

    @staticmethod
    def read_ebml_element(io: BoundedIO, options: EBMLDecodeOptions) -> "EBMLElement":
        # spec: RFC 8794
        # EBML Element structure:
        # element_id | 1-8 bytes | <varint>
        # size   | 1-8 bytes | <varint>
        # ... data (payload or children or both)

        # Note: size is the size after it, so the <size of both varints> bytes less then the whole size

        # class EBMLElement {
        #     VarInt element_id
        #     BarInt data_size
        #     Byte data[data_size]
        # };

        element_id_res = EBMLVarInt.from_io(io)

        if element_id_res.err():
            msg = f"Element ID Parse error: {element_id_res.as_err()}"
            raise RuntimeError(msg)

        element_id, element_id_bytes = element_id_res.as_ok()

        # An Element ID is a Variable-Size Integer. By default, Element IDs are from one octet to four octets
        # in length, although Element IDs of greater lengths MAY be used if the EBMLMaxIDLength
        # Element of the EBML Header is set to a value greater than four (see Section 11.2.4).

        if element_id_bytes > options.max_id_length:
            msg = f"ELement ID VarInt exceeds allowed size of {options.max_id_length}: {element_id_bytes}"
            raise RuntimeError(msg)

        if not element_id.is_valid_element_id(element_id_bytes):
            msg = f"Invalid Element ID: {element_id}"
            raise RuntimeError(msg)

        size_span = io.span.next_span(element_id_bytes)

        data_size_res = EBMLVarInt.from_io(io.new_span_io(size_span))

        if data_size_res.err():
            msg = f"Data Size Parse error: {data_size_res.as_err()}"
            raise RuntimeError(msg)

        data_size, data_size_bytes = data_size_res.as_ok()

        if data_size_bytes > options.max_size_length:
            msg = f"Data Size VarInt exceeds allowed size of {options.max_size_length}: {data_size_bytes}"
            raise RuntimeError(msg)

        if data_size == ((1 << (data_size_bytes * 7)) - 1):
            msg = f"Unknown data size not supported: {data_size} ({data_size_bytes})"
            raise RuntimeError(msg)

        header_sizes = (element_id_bytes, data_size_bytes)

        span = EBMLElementSpan.from_ebml_specified_size(
            io.span.sub_span(data_size.value),
            header_sizes=header_sizes,
        )
        return EBMLElement(element_id, span, header_sizes, is_container=False)

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

    def __parse() -> "TODO":
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


class EBMLDocument:
    pass
    # needs header + body


EBML_NUMBER_BYTE_ORDER_STR: Literal["big"] = "big"
EBML_NUMBER_BYTE_ORDER = ByteOrder.Big


@final
@decorate_class(slots=True)
class EBMLSignedIntegerElement(EBMLElement, FinalEBMLElement):
    value: int

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: int,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLSignedIntegerElement":
        # spec: RFC 8794
        # EBML Signed Integer Element structure:
        # element     | <variable element size> bytes | parent element
        # ... data (0-8 bytes)

        # class EBMLSignedIntegerElement extends EBMLElement {
        #     Byte s_integer_data[0-8]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLSignedIntegerElement(parent, 0)

        if payload_size > 8 or payload_size < 0:
            msg = f"Invalid payload size for EBMLSignedIntegerElement:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            s_integer_value_raw = f.read(payload_size)

            s_int_val = int.from_bytes(
                s_integer_value_raw,
                byteorder=EBML_NUMBER_BYTE_ORDER_STR,
                signed=True,
            )

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLSignedIntegerElement(parent, s_int_val)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLSignedIntegerElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLSignedIntegerElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLSignedIntegerElement":
        return EBMLSignedIntegerElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLSignedIntegerElement parent: {EBMLElement.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLUnsignedIntegerElement(EBMLElement, FinalEBMLElement):
    value: int

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: int,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLUnsignedIntegerElement":
        # spec: RFC 8794
        # EBML Unsigned Integer Element structure:
        # element     | <variable element size> bytes | parent element
        # ... data (0-8 bytes)

        # class EBMLUnsignedIntegerElement extends EBMLElement {
        #     Byte u_integer_data[0-8]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLUnsignedIntegerElement(parent, 0)

        if payload_size > 8 or payload_size < 0:
            msg = (
                f"Invalid payload size for EBMLUnsignedIntegerElement:  {payload_size}"
            )
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            u_integer_value_raw = f.read(payload_size)

            u_int_val = int.from_bytes(
                u_integer_value_raw,
                byteorder=EBML_NUMBER_BYTE_ORDER_STR,
                signed=False,
            )

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLUnsignedIntegerElement(parent, u_int_val)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLUnsignedIntegerElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLUnsignedIntegerElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLUnsignedIntegerElement":
        return EBMLUnsignedIntegerElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLUnsignedIntegerElement parent: {EBMLElement.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLFloatElement(EBMLElement, FinalEBMLElement):
    value: float

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: float,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLFloatElement":
        # spec: RFC 8794
        # EBML Float Element structure:
        # element     | <variable element size> bytes | parent element
        # ... data (0-8 bytes)

        # class EBMLFloatElement extends EBMLElement {
        #     Byte float_data[0-8]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLFloatElement(parent, 0)

        if payload_size not in [4, 8]:
            msg = f"Invalid payload size for EBMLFloatElement:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            float_value_raw = f.read(payload_size)

            float_val: float
            if payload_size == 4:
                float_val = Unpacker.unpack_one(
                    EBML_NUMBER_BYTE_ORDER,
                    Float32(),
                    float_value_raw,
                )
            elif payload_size == 8:
                float_val = Unpacker.unpack_one(
                    EBML_NUMBER_BYTE_ORDER,
                    Float64(),
                    float_value_raw,
                )
            else:
                msg = "Implementation error: float size not checked correctly"
                raise RuntimeError(msg)

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLFloatElement(parent, float_val)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLFloatElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLFloatElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLFloatElement":
        return EBMLFloatElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLFloatElement parent: {EBMLElement.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLStringElement(EBMLElement, FinalEBMLElement):
    value: str

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: str,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLStringElement":
        # spec: RFC 8794
        # EBML String structure:
        # element     | <variable element size> bytes | parent element
        # ... data (* bytes)

        # class EBMLStringElement extends EBMLElement {
        #     Byte str_data[*]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLStringElement(parent, "")

        if payload_size > VINTMAX:
            msg = f"Invalid payload size for EBMLStringElement:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            str_value_raw = f.read(payload_size)

            str_value = str_value_raw.decode("ascii").rstrip("\x00")

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLStringElement(parent, str_value)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLStringElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLStringElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLStringElement":
        return EBMLStringElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLStringElement parent: {EBMLElement.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLUTF8Element(EBMLElement, FinalEBMLElement):
    value: str

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: str,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLUTF8Element":
        # spec: RFC 8794
        # EBML UTF-8 structure:
        # element     | <variable element size> bytes | parent element
        # ... data (* bytes)

        # class EBMLUTF8Element extends EBMLElement {
        #     Byte str_data[*]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLUTF8Element(parent, "")

        if payload_size > VINTMAX:
            msg = f"Invalid payload size for EBMLUTF8Element:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            str_value_raw = f.read(payload_size)

            str_value = str_value_raw.decode("utf-8").rstrip("\x00")

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLUTF8Element(parent, str_value)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLUTF8Element":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLUTF8Element.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLUTF8Element":
        return EBMLUTF8Element.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return (
            f"<EBMLUTF8Element parent: {EBMLElement.__str__(self)} value: {self.value}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


# 2001-01-01T00:00:00.000000000 UTC
EBML_DATE_EPOCH = datetime(2001, 1, 1, 0, 0, 0, tzinfo=UTC)


@final
@decorate_class(slots=True)
class EBMLDateElement(EBMLElement, FinalEBMLElement):
    value: datetime

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: datetime,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLDateElement":
        # spec: RFC 8794
        # EBML Date structure:
        # element     | <variable element size> bytes | parent element
        # ... data (0-8 bytes)

        # class EBMLDateElement extends EBMLElement {
        #     Byte date_data[0-8]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLDateElement(parent, EBML_DATE_EPOCH)

        if payload_size != 8:
            msg = f"Invalid payload size for EBMLDateElement:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            date_int_value_raw = f.read(payload_size)

            # in nanoseconds
            date_int_value = int.from_bytes(
                date_int_value_raw,
                byteorder=EBML_NUMBER_BYTE_ORDER_STR,
                signed=True,
            )

            date_value = EBML_DATE_EPOCH + timedelta(
                microseconds=date_int_value // 1000,
            )

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLDateElement(parent, date_value)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLDateElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLDateElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLDateElement":
        return EBMLDateElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return (
            f"<EBMLDateElement parent: {EBMLElement.__str__(self)} value: {self.value}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


class MKVDecodeType(Enum):
    Check = "check"
    Normal = "normal"


@dataclass(slots=True, repr=True)
class MKVDecodeOptions:
    strict: bool
    type: MKVDecodeType

    @staticmethod
    def default() -> "MKVDecodeOptions":
        return MKVDecodeOptions(
            strict=True,
            type=MKVDecodeType.Normal,
        )


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
