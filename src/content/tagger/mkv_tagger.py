from collections.abc import Generator, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
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

from content.tagger.parser import (
    BoundedIO,
    ByteOrder,
    Float32,
    Float64,
    SimpleSpan,
    Unpacker,
)
from content.tagger.schema.parser import (
    DefaultEmpty,
    DefaultOptions,
    DefaultRequired,
    DocType,
    EBMLAdvancedElementType,
    EBMLAdvancedElementTypeAbstract,
    EBMLAdvancedElementTypeBinary,
    EBMLAdvancedElementTypeDate,
    EBMLAdvancedElementTypeFloat,
    EBMLAdvancedElementTypeInteger,
    EBMLAdvancedElementTypeMaster,
    EBMLAdvancedElementTypeString,
    EBMLElementDescription,
    EBMLElementDescriptionGeneric,
    EBMLElementType,
    EBMLSpec,
    ebml_read_spec_xml,
    filter_spec_elements,
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

        # spec: RFC 8794
        # chapter 4

        # VINT_WIDTH VINT_MARKER VINT_DATA

        # VINT_WIDTH: 0 bits [0,*]
        # VINT_MARKER: 1 bit [1,1]
        # VINT_DATA: <data>, length determined by VINT_WIDTH

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
        # spec: RFC 8794
        # chapter 5

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
        return f"<VarInt {hex(self.__value)}>"

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
        # chapter 3

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

        # spec: RFC 8794
        # chapter 5

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

        # spec: RFC 8794
        # chapter 6

        # spec: RFC 8794
        # chapter 6.1
        # special values and constraints of the data size

        if data_size_bytes > options.max_size_length:
            msg = f"Data Size VarInt exceeds allowed size of {options.max_size_length}: {data_size_bytes}"
            raise RuntimeError(msg)

        # spec: RFC 8794
        # chapter 6.2

        # unknown data size

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
        # chapter 7.1

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
        # chapter 7.2

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
        # chapter 7.3

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
        # chapter 7.4

        # EBML String Element structure:
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
        # chapter 7.5

        # EBML UTF-8 Element structure:
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
        # chapter 7.6

        # EBML Date Element structure:
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


@decorate_class(slots=True)
class EBMLMasterElement(EBMLElement):
    def __init__(
        self: Self,
        parent: EBMLElement,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=True,
        )

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLMasterElement":
        # spec: RFC 8794
        # chapter 7.7

        # EBML Master Element structure:
        # element     | <variable element size> bytes | parent element
        # ... data (* bytes), children elements

        # class EBMLMasterElement extends EBMLElement {
        #     EBMLElement children[*]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLMasterElement(parent)

        if payload_size > VINTMAX:
            msg = f"Invalid payload size for EBMLMasterElement:  {payload_size}"
            raise RuntimeError(msg)

        return EBMLMasterElement(parent)

    @staticmethod
    def read_ebml_master_element(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLMasterElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLMasterElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_ebml_master_element_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLMasterElement":
        return EBMLMasterElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLMasterElement parent: {EBMLElement.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLBinaryElement(EBMLElement, FinalEBMLElement):
    value: bytes

    def __init__(
        self: Self,
        parent: EBMLElement,
        value: bytes,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            parent.header_sizes,
            is_container=False,
        )

        self.value = value

    @staticmethod
    def __read_impl(io: BoundedIO, parent: EBMLElement) -> "EBMLBinaryElement":
        # spec: RFC 8794
        # chapter 7.8

        # EBML Binary Element structure:
        # element     | <variable element size> bytes | parent element
        # ... data (* bytes)

        # class EBMLBinaryElement extends EBMLElement {
        #     Byte binary_data[*]
        # } ;

        payload_size = parent.span.payload_span.size

        if payload_size == 0:
            return EBMLBinaryElement(parent, b"")

        if payload_size > VINTMAX:
            msg = f"Invalid payload size for EBMLBinaryElement:  {payload_size}"
            raise RuntimeError(msg)

        with io.r_ctx(force_entire_read=True) as f:

            binary_value = f.read(payload_size)

            parent.span.add_header(payload_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return EBMLBinaryElement(parent, binary_value)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
    ) -> "EBMLBinaryElement":
        element = EBMLElement.read_ebml_element(io, options)
        return EBMLBinaryElement.__read_impl(element.payload_io(io), element)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLElement,
    ) -> "EBMLBinaryElement":
        return EBMLBinaryElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLBinaryElement parent: {EBMLElement.__str__(self)} value: {self.value!r}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLEmptyMasterElement(EBMLMasterElement, FinalEBMLElement):

    def __init__(
        self: Self,
        parent: EBMLMasterElement,
    ) -> None:
        super().__init__(
            parent,
        )

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: EBMLMasterElement,
    ) -> "EBMLEmptyMasterElement":
        # empty master element, just skip everything

        return EBMLEmptyMasterElement(parent)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: EBMLMasterElement,
    ) -> "EBMLEmptyMasterElement":
        return EBMLEmptyMasterElement.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<EBMLEmptyMasterElement parent: {EBMLMasterElement.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
class SupportedMasterElements:
    pass


def read_element(  # noqa: PLR0915
    io: BoundedIO,
    options: EBMLDecodeOptions,
    spec: EBMLSpec.EBMLSpecById,
) -> tuple[EBMLElement, EBMLElementDescription]:

    element = EBMLElement.read_ebml_element(io, options)

    element_desc = spec.get(element.element_id.value, None)

    if element_desc is None:
        msg = f"Invalid EBML Element ID, not defined by schema: {element.element_id}"
        raise RuntimeError(msg)

    def assert_valid(res: Result[Any, str]) -> None:
        if res.ok():
            return

        msg = f"Failed to validate element {element_desc.name} of type {element_desc.type.type.name}: {res.as_err()}"
        raise RuntimeError(msg)

    match element_desc.type:
        case EBMLAdvancedElementTypeInteger() as int_type:
            match int_type.type:
                case EBMLElementType.SignedInteger:
                    element = EBMLSignedIntegerElement.read_from_parent(
                        element.payload_io(io),
                        element,
                    )

                    validate_res = int_type.validate(element.value)
                    assert_valid(validate_res)

                    return (element, element_desc)
                case EBMLElementType.UnsignedInteger:
                    element = EBMLUnsignedIntegerElement.read_from_parent(
                        element.payload_io(io),
                        element,
                    )

                    validate_res = int_type.validate(element.value)
                    assert_valid(validate_res)

                    return (element, element_desc)
                case _:
                    assert_never(int_type)

        case EBMLAdvancedElementTypeFloat() as float_type:
            element = EBMLFloatElement.read_from_parent(
                element.payload_io(io),
                element,
            )

            validate_res = float_type.validate(element.value)
            assert_valid(validate_res)

            return (element, element_desc)
        case EBMLAdvancedElementTypeString() as str_type:
            match str_type.type:
                case EBMLElementType.String:
                    element = EBMLStringElement.read_from_parent(
                        element.payload_io(io),
                        element,
                    )

                    validate_res = str_type.validate(element.value)
                    assert_valid(validate_res)

                    return (element, element_desc)
                case EBMLElementType.UTF8:
                    element = EBMLUTF8Element.read_from_parent(
                        element.payload_io(io),
                        element,
                    )

                    validate_res = str_type.validate(element.value)
                    assert_valid(validate_res)

                    return (element, element_desc)
                case _:
                    assert_never(str_type)

        case EBMLAdvancedElementTypeDate() as date_type:
            element = EBMLDateElement.read_from_parent(
                element.payload_io(io),
                element,
            )
            validate_res = date_type.validate(element.value)
            assert_valid(validate_res)

            return (element, element_desc)
        case EBMLAdvancedElementTypeMaster():
            master_element = EBMLMasterElement.read_ebml_master_element_from_parent(
                element.payload_io(io),
                element,
            )
            match element.element_id:
                # TODO: use SupportedMasterElements
                # NOTE: only needed if a master element has some payload, and not just children
                case _:
                    element = EBMLEmptyMasterElement.read_from_parent(
                        master_element.payload_io(io),
                        master_element,
                    )
                    return (element, element_desc)
        case EBMLAdvancedElementTypeBinary() as binary_type:
            element = EBMLBinaryElement.read_from_parent(
                element.payload_io(io),
                element,
            )

            validate_res = binary_type.validate(element.value)
            assert_valid(validate_res)

            return (element, element_desc)

        case _:
            assert_never(element_desc)


def get_element_value[A](
    element_type: EBMLAdvancedElementTypeAbstract[A],
    element: EBMLElement,
) -> A:

    def cast_to_type[B](_typ: EBMLAdvancedElementTypeAbstract[B], a: B) -> A:
        return cast(A, a)

    match element_type:
        case EBMLAdvancedElementTypeInteger() as int_type:
            match int_type.type:
                case EBMLElementType.SignedInteger:
                    if not isinstance(element, EBMLSignedIntegerElement):
                        msg = "Invalid SignedInteger: type not dispatched to correct class"
                        raise TypeError(msg)

                    return cast_to_type(int_type, element.value)
                case EBMLElementType.UnsignedInteger:
                    if not isinstance(element, EBMLUnsignedIntegerElement):
                        msg = "Invalid UnsignedInteger: type not dispatched to correct class"
                        raise TypeError(msg)

                    return cast_to_type(int_type, element.value)
                case _:
                    assert_never(int_type)

        case EBMLAdvancedElementTypeFloat() as float_type:
            if not isinstance(element, EBMLFloatElement):
                msg = "Invalid Float: type not dispatched to correct class"
                raise TypeError(msg)

            return cast_to_type(float_type, element.value)
        case EBMLAdvancedElementTypeString() as str_type:
            match str_type.type:
                case EBMLElementType.String:
                    if not isinstance(element, EBMLStringElement):
                        msg = "Invalid String: type not dispatched to correct class"
                        raise TypeError(msg)

                    return cast_to_type(str_type, element.value)
                case EBMLElementType.UTF8:
                    if not isinstance(element, EBMLUTF8Element):
                        msg = "Invalid UTF8: type not dispatched to correct class"
                        raise TypeError(msg)

                    return cast_to_type(str_type, element.value)
                case _:
                    assert_never(str_type)

        case EBMLAdvancedElementTypeDate() as date_type:
            if not isinstance(element, EBMLDateElement):
                msg = "Invalid Date: type not dispatched to correct class"
                raise TypeError(msg)

            return cast_to_type(date_type, element.value)
        case EBMLAdvancedElementTypeMaster():
            msg = "Can't get element value for Master Type"
            raise RuntimeError(msg)
        case EBMLAdvancedElementTypeBinary() as binary_type:
            if not isinstance(element, EBMLBinaryElement):
                msg = "Invalid Binary: type not dispatched to correct class"
                raise TypeError(msg)

            return cast_to_type(binary_type, element.value)

        case _:
            msg = f"Invalid element_type: {element_type}"
            raise RuntimeError(msg)


def ebml_iter_elements(
    io: BoundedIO,
    options: EBMLDecodeOptions,
    spec: EBMLSpec,
) -> Generator[tuple[EBMLElement, EBMLElementDescription]]:

    spec_by_id = spec.elements_by_id()

    # TODO: check minOccurs and maxOccurs in iter function

    pos = io.span.start
    end = io.span.end

    while pos < end:
        new_io = io.new_span_io(SimpleSpan(pos, end - pos))
        element, element_desc = read_element(new_io, options, spec_by_id)

        if pos + element.span.total.size > end:
            msg = f"Element {element.element_id} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield (element, element_desc)
        pos += element.span.total.size

    if pos != end:
        msg = f"Element didn't reach to the end of the parent span: {pos} != {end}"
        raise RuntimeError(msg)


EBMLMainSpec = ebml_read_spec_xml("ebml/ebml.xml")


def filter_ebml_global_element(element: EBMLElementDescription) -> bool:
    return element.name in ["Void", "CRC-32"]


# spec: RFC 8794
# chapter 11.2


# EBML Header Elements
EBMLHeaderElementsSpec = filter_spec_elements(
    EBMLMainSpec,
    lambda element: not filter_ebml_global_element(element),
)


def typed_spec[A: (EBMLAdvancedElementType)](
    spec: EBMLElementDescription,
    a: type[A],
) -> EBMLElementDescriptionGeneric[A]:
    if isinstance(spec.type, a):
        return cast(EBMLElementDescriptionGeneric[A], spec)
    msg = (
        f"Expected EBMLElementDescription to be of type {a} but have: {type(spec.type)}"
    )
    raise RuntimeError(msg)


def require_default_spec_value[A](value: A | DefaultOptions) -> A:
    if isinstance(value, (DefaultRequired, DefaultEmpty)):
        msg = f"Required a default value, but have: {value}"
        raise TypeError(msg)

    return value


EBMLHeaderMasterSpec = EBMLHeaderElementsSpec.elements_by_name()["EBML"]
EBMLVersionSpec = typed_spec(
    EBMLHeaderElementsSpec.elements_by_name()["EBMLVersion"],
    EBMLAdvancedElementTypeInteger,
)
EBMLMaxIDLengthSpec = typed_spec(
    EBMLHeaderElementsSpec.elements_by_name()["EBMLMaxIDLength"],
    EBMLAdvancedElementTypeInteger,
)
EBMLMaxSizeLengthSpec = typed_spec(
    EBMLHeaderElementsSpec.elements_by_name()["EBMLMaxSizeLength"],
    EBMLAdvancedElementTypeInteger,
)
EBMLDocTypeSpec = typed_spec(
    EBMLHeaderElementsSpec.elements_by_name()["DocType"],
    EBMLAdvancedElementTypeString,
)
EBMLDocTypeVersionSpec = typed_spec(
    EBMLHeaderElementsSpec.elements_by_name()["DocTypeVersion"],
    EBMLAdvancedElementTypeInteger,
)
# spec: RFC 8794
# chapter 11.3

# EBML Global Elements
# EBML allows some special Elements to be found within more than one parent in an EBML
# Document or optionally at the Root Level of an EBML Body. These Elements are called Global
# Elements. There are two Global Elements that can be found in any EBML Document: the CRC-32
# Element and the Void Element. An EBML Schema MAY add other Global Elements to the format it
# deﬁnes. These extra elements apply only to the EBML Body, not the EBML Header.
# Global Elements are EBML Elements whose EBMLLastParent part of the path has a
# GlobalPlaceholder. Because it is the last Parent part of the path, a Global Element might also have
# EBMLParentPath parts in its path. In this case, the Global Element can only be found within this
# EBMLParentPath path -- i.e., it's not fully "global".
# A Global Element can be found in many Parent Elements, allowing the same number of
# occurrences in each Parent where this Element is found.


EBMLGlobalElementsSpec = filter_spec_elements(
    EBMLMainSpec,
    filter_ebml_global_element,
)


@dataclass(slots=True, repr=True)
class EBMLHeaderOptions:
    options: EBMLDecodeOptions
    doc_type: DocType


@final
@decorate_class(slots=True)
class EBMLHeader(EBMLElement, FinalEBMLElement):
    options: EBMLHeaderOptions

    def __init__(
        self: Self,
        parent: EBMLElement,
        options: EBMLHeaderOptions,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            header_sizes=parent.header_sizes,
            is_container=False,
        )

        self.options = options

    @staticmethod
    def __read_impl(io: BoundedIO) -> "EBMLHeader":
        # spec: RFC 8794
        # chapter 8.1

        # The EBML Header is a declaration that provides processing instructions and identiﬁcation of the
        # EBML Body. The EBML Header of an EBML Document is analogous to the XML Declaration of an
        # XML Document.
        # The EBML Header documents the EBML Schema (also known as the EBML DocType) that is used
        # to semantically interpret the structure and meaning of the EBML Document. Additionally, the
        # EBML Header documents the versions of both EBML and the EBML Schema that were used to
        # write the EBML Document and the versions required to read the EBML Document.
        # The EBML Header MUST contain a single Master Element with an Element Name of EBML and
        # Element ID of 0x1A45DFA3 (see Section 11.2.1); the Master Element may have any number of
        # additional EBML Elements within it. The EBML Header of an EBML Document that uses an
        # EBMLVersion of 1 MUST only contain EBML Elements that are deﬁned as part of this document.
        # Elements within an EBML Header can be at most 4 octets long, except for the EBML Element with
        # Element Name EBML and Element ID 0x1A45DFA3 (see Section 11.2.1); this Element can be up to 8
        # octets long.

        ebml_header_master_options = EBMLDecodeOptions(
            max_id_length=8,
            max_size_length=8,
        )
        element = EBMLElement.read_ebml_element(io, ebml_header_master_options)

        if element.element_id != EBMLHeaderMasterSpec.id:
            msg = f"Invalid EBML Header element ID: {element.element_id}"
            raise RuntimeError(msg)

        version = require_default_spec_value(EBMLVersionSpec.type.default)
        max_id_length = require_default_spec_value(EBMLMaxIDLengthSpec.type.default)
        max_size_length = require_default_spec_value(EBMLMaxSizeLengthSpec.type.default)

        doc_type_str: Optional[str] = None
        doc_type_version = require_default_spec_value(
            EBMLDocTypeVersionSpec.type.default,
        )

        ebml_header_children_options = EBMLDecodeOptions(
            max_id_length=4,
            max_size_length=8,
        )

        children_elements = ebml_iter_elements(
            element.payload_io(io),
            ebml_header_children_options,
            EBMLHeaderElementsSpec,
        )

        for children_element, element_desc in children_elements:
            match element_desc.type:
                case EBMLAdvancedElementTypeInteger() as int_type:
                    int_value = get_element_value(int_type, element)

                    match element_desc.name:
                        case EBMLVersionSpec.name:
                            version = int_value
                        case EBMLMaxIDLengthSpec.name:
                            max_id_length = int_value
                        case EBMLMaxSizeLengthSpec.name:
                            max_size_length = int_value
                        case EBMLDocTypeVersionSpec.name:
                            doc_type_version = int_value
                        case _:
                            # ignore unused values atm
                            pass
                case EBMLAdvancedElementTypeFloat() as float_type:
                    _float_value = get_element_value(float_type, element)

                    match element_desc.name:
                        case _:
                            # ignore unused values atm
                            pass
                case EBMLAdvancedElementTypeString() as str_type:
                    str_value = get_element_value(str_type, element)

                    match element_desc.name:
                        case EBMLDocTypeSpec.name:
                            doc_type_str = str_value
                        case _:
                            # ignore unused values atm
                            pass
                case EBMLAdvancedElementTypeDate() as date_type:
                    _date_value = get_element_value(date_type, element)

                    match element_desc.name:
                        case _:
                            # ignore unused values atm
                            pass
                case EBMLAdvancedElementTypeMaster():
                    msg = f"Master element not allowed in header element: {children_element}"
                    raise RuntimeError(msg)
                case EBMLAdvancedElementTypeBinary() as binary_type:
                    _binary_value = get_element_value(binary_type, element)

                    match element_desc.name:
                        case _:
                            # ignore unused values atm
                            pass
                case _:
                    assert_never(element_desc)

        decode_options: EBMLDecodeOptions = EBMLDecodeOptions(
            max_id_length=max_id_length,
            max_size_length=max_size_length,
        )

        if doc_type_str is None:
            msg = "missing doc_type_str in header"
            raise RuntimeError(msg)

        doc_type: DocType = DocType(type=doc_type_str, version=doc_type_version)

        if version != 1:
            msg = (
                f"Only version 1 of the EBML spec is supported atm, but got: {version}"
            )
            raise RuntimeError(msg)

        options: EBMLHeaderOptions = EBMLHeaderOptions(
            decode_options,
            doc_type,
        )

        return EBMLHeader(element, options)

    @staticmethod
    def read(
        io: BoundedIO,
    ) -> "EBMLHeader":
        return EBMLHeader.__read_impl(io)

    def __str__(self: Self) -> str:
        return (
            f"<EBMLHeader parent: {EBMLElement.__str__(self)} options: {self.options}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLBody(EBMLElement, FinalEBMLElement):

    def __init__(
        self: Self,
        parent: EBMLElement,
    ) -> None:
        super().__init__(
            parent.element_id,
            parent.span,
            header_sizes=parent.header_sizes,
            is_container=True,
        )

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        options: EBMLDecodeOptions,
        spec: EBMLSpec,
    ) -> "EBMLBody":
        # spec: RFC 8794
        # chapter 8.2

        # All data of an EBML Document following the EBML Header is the EBML Body. The end of the
        # EBML Body, as well as the end of the EBML Document that contains the EBML Body, is reached at
        # whichever comes ﬁrst: the beginning of a new EBML Header at the Root Level or the end of the
        # ﬁle. This document deﬁnes precisely which EBML Elements are to be used within the EBML
        # Header but does not name or deﬁne which EBML Elements are to be used within the EBML Body.
        # The deﬁnition of which EBML Elements are to be used within the EBML Body is deﬁned by an
        # EBML Schema.
        # Within the EBML Body, the maximum octet length allowed for any Element ID is set by the
        # EBMLMaxIDLength Element of the EBML Header, and the maximum octet length allowed for any
        # Element Data Size is set by the EBMLMaxSizeLength Element of the EBML Header.

        spec_id = spec.elements_by_id()

        element, element_desc = read_element(io, options, spec_id)

        if element_desc.type.type != EBMLElementType.Master:
            msg = f"Only Ma Master element allowed for a EBML Body element: {element} {element_desc}"
            raise RuntimeError(msg)

        return EBMLBody(element)

    @staticmethod
    def read(
        io: BoundedIO,
        options: EBMLDecodeOptions,
        spec: EBMLSpec,
    ) -> "EBMLBody":
        return EBMLBody.__read_impl(io, options, spec)

    def __str__(self: Self) -> str:
        return f"<EBMLBody parent: {EBMLElement.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


EBMLMKVSpec = ebml_read_spec_xml("mkv/ebml_matroska.xml")


def get_spec_by_doc_type(doc_type: DocType) -> Result[EBMLSpec, str]:

    available_specs: list[EBMLSpec] = [EBMLMKVSpec]

    for spec in available_specs:
        if doc_type.type == spec.doc_type.type:
            if doc_type.version > spec.doc_type.version:
                return Err(
                    f"Unsupported version: max supported version is {spec.doc_type.version}",
                )

            result = EBMLSpec(doc_type=doc_type)

            for elem in EBMLGlobalElementsSpec.elements:
                result.append(elem)

            for element in spec.elements:

                should_include = element.versions.valid(doc_type.version)

                if should_include:
                    result.append(element)

            return Ok(result)

    return Err("No such DocType")


@final
@decorate_class(slots=True)
class EBMLDocument(FinalEBMLElement):
    header: EBMLHeader
    body: EBMLBody
    span: EBMLElementSpan

    def __init__(
        self: Self,
        header: EBMLHeader,
        body: EBMLBody,
        span: EBMLElementSpan,
    ) -> None:
        super().__init__()

        self.header = header
        self.body = body
        self.span = span

    @staticmethod
    def __read_impl(
        io: BoundedIO,
    ) -> "EBMLDocument":
        # spec: RFC 8794
        # chapter 8

        # An EBML Document is composed of only two components, an EBML Header and an EBML Body.
        # An EBML Document MUST start with an EBML Header that declares signiﬁcant characteristics of
        # the entire EBML Body. An EBML Document consists of EBML Elements and MUST NOT contain
        # any data that is not part of an EBML Element.

        header = EBMLHeader.read(io)
        body_io = io.new_span_io(io.span.next_span(header.span.total.size))

        header_options = header.options

        spec_res = get_spec_by_doc_type(header_options.doc_type)

        if spec_res.err():
            msg = f"DocType {header_options.doc_type} is not supported: {spec_res.as_err()}"
            raise RuntimeError(msg)

        spec = spec_res.as_ok()

        body = EBMLBody.read(body_io, header_options.options, spec)

        total_span = SimpleSpan(
            header.span.total.start,
            header.span.total.size + body.span.total.size,
        )
        span = EBMLElementSpan(total_span, header.span.total.size)

        span.add_header(body.span.total.size)

        if span.payload_span.size != 0:
            msg = f"Expected empty last span but got:{span.payload_span.size}"
            raise RuntimeError(msg)

        return EBMLDocument(header, body, span)

    @staticmethod
    def read_from_io(
        io: BoundedIO,
    ) -> "EBMLDocument":
        return EBMLDocument.__read_impl(io)

    def __str__(self: Self) -> str:
        return (
            f"<EBMLDocument span: {self.span} header: {self.header} body: {self.body}>"
        )

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class EBMLStream(FinalEBMLElement):
    documents: list[EBMLDocument]

    def __init__(
        self: Self,
        documents: list[EBMLDocument],
    ) -> None:
        super().__init__()

        self.documents = documents

    @staticmethod
    def __read_from_span_impl(
        io_base: BinaryIO,
        span: SimpleSpan,
    ) -> "EBMLStream":
        # spec: RFC 8794
        # chapter 9

        # An EBML Stream is a ﬁle that consists of one or more EBML Documents that are concatenated
        # together. An occurrence of an EBML Header at the Root Level marks the beginning of an EBML
        # Document.

        documents: list[EBMLDocument] = []

        pos = span.start

        while pos < span.end:
            io = BoundedIO.get_new(io_base, SimpleSpan(pos, span.end - pos))
            document = EBMLDocument.read_from_io(io)

            if pos + document.span.total.size > span.end:
                msg = f"document {document} at {pos} extends past parent boundary"
                raise RuntimeError(msg)

            documents.append(document)
            pos += document.span.total.size

        if pos != span.end:
            msg = f"EBMLDocument didn't reach to the end of the parent span: {pos} != {span.end}"
            raise RuntimeError(msg)

        return EBMLStream(documents)

    @staticmethod
    def __read_from_file_impl(
        f: BinaryIO,
    ) -> "EBMLStream":
        f.seek(0, 2)
        filesize = f.tell()

        span = SimpleSpan(0, filesize)
        return EBMLStream.__read_from_span_impl(f, span)

    @staticmethod
    def read_from_file(
        f: BinaryIO,
    ) -> "EBMLStream":
        return EBMLStream.__read_from_file_impl(f)

    def __str__(self: Self) -> str:
        return f"<EBMLStream documents: {self.documents}>"

    def __repr__(self: Self) -> str:
        return str(self)


def is_mkv_file(
    f: BinaryIO,
) -> Optional[str]:
    f.seek(0)

    try:
        f.seek(0, 2)
        filesize = f.tell()
        header = EBMLHeader.read(BoundedIO.get_new(f, SimpleSpan(0, filesize)))

        header_options = header.options

        spec_res = get_spec_by_doc_type(header_options.doc_type)

        if spec_res.err():
            return f"DocType {header_options.doc_type} is not supported: {spec_res.as_err()}"

        spec = spec_res.as_ok()

        if spec.doc_type.type != EBMLMKVSpec.doc_type.type:
            return "Not a matroska EBML file"

        f.seek(0)
    except (RuntimeError, ValueError, TypeError) as err:
        return str(err)
    return None


@decorate_class(slots=True)
class VideoTaggerMKV(VideoTagger):
    __streams: int
    __types: list[EBMLVarInt]

    def __init__(
        self: Self,
        file: Path,
        streams: int,
        types: list[EBMLVarInt],
    ) -> None:
        super().__init__(file)
        self.__streams = streams
        self.__types = types

        streams = 0

    @staticmethod
    def get_handle(file: Path) -> Result["VideoTagger", str]:

        try:

            with file.open("rb") as f:
                mkv_res = is_mkv_file(f)
                if mkv_res is not None:
                    return Err(mkv_res)

                f.seek(0)

                streams = 0
                types: list[EBMLVarInt] = [AUDIO_TODO, VIDEO_TODO]

                # read the file, so that we check if we can parse it correctly and that it is an mkv file
                # TODO
                # for strh in find_strh_chunks_with_type(f, types, options):
                #     streams = streams + 1
                #     lang = strh.read_language(f)
                #     # check if this lang is valid

                #     if isinstance(lang, str):
                #         msg = _("Invalid language in avi detected: {lang}").format(
                #             lang=lang,
                #         )
                #         return Err(msg)

                return Ok(
                    VideoTaggerMKV(file, streams, types),
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

        def print_element(element: EBMLElement, *, depth: int) -> None:

            local_priority = (
                InspectPriority.Important if element.is_list else InspectPriority.Normal
            )

            if local_priority.as_int() > priority.as_int():
                return

            name: str = f"{element.fourcc}"
            if isinstance(chunk, AVIList):
                name = f"{chunk.fourcc}({chunk.type})"

            element = InspectElement(name, size=chunk.span.total.size)

            printer.element(element, depth)

        with self.file.open(mode="rb") as f:

            def iterate_elements_recursive(span: SimpleSpan, *, depth: int) -> None:
                for element in ebml_iter_elements(f, span):

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
