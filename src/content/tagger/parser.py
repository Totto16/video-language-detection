import os
import struct
import sys
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from enum import StrEnum
from io import UnsupportedOperation
from types import TracebackType
from typing import (
    BinaryIO,
    Literal,
    Optional,
    Protocol,
    Self,
    assert_never,
    cast,
    final,
    override,
)
from uuid import UUID

from helper.translation import get_translator

_ = get_translator()


@final
class SimpleSpan:
    start: int
    size: int

    def __init__(
        self: Self,
        start: int,
        size: int,
    ) -> None:
        self.start = start
        self.size = size

    @property
    def end(self: Self) -> int:
        return self.start + self.size

    def __str__(self: Self) -> str:
        return f"<SimpleSpan start: {self.start} size: {self.size}>"

    def __repr__(self: Self) -> str:
        return str(self)


class BoundedIOReadable(Protocol):
    def read(self: Self, amount: int) -> bytes: ...

    def skip(self: Self, amount: int) -> None: ...


class BoundedIOWriteable(Protocol):
    def write(self: Self, data: bytes) -> None: ...

    def flush(self: Self) -> None: ...


class BoundedIORW(BoundedIOReadable, BoundedIOWriteable):
    pass


class ExclusiveIOBase:
    __f: BinaryIO

    __holder: int

    def __init__(self: Self, io_base: BinaryIO) -> None:
        self.__f = io_base

        self.__holder = id(None)

    def __assert_no_holder(self: Self) -> None:
        if self.__holder != id(None):
            msg = f"ExclusiveIOBase has a holder, this operation is not possible: {self.__holder}"
            raise RuntimeError(msg)

    def __assert_holder(self: Self) -> None:
        if self.__holder == id(None):
            msg = f"ExclusiveIOBase has no holder, this operation is not possible: {self.__holder}"
            raise RuntimeError(msg)

    def size(self: Self) -> int:
        self.__assert_no_holder()
        self.__f.seek(0, 2)
        return self.__f.tell()

    def size_no_seek(self: Self) -> int:
        self.__assert_holder()

        try:
            fileno = self.__f.fileno()
            return os.fstat(fileno).st_size
        except UnsupportedOperation:
            # use double seek instead of fstat
            current_pos = self.__f.tell()
            self.__f.seek(0, 2)
            filesize = self.__f.tell()
            self.__f.seek(current_pos)
            return filesize

    def tell(self: Self) -> int:
        self.__assert_holder()
        return self.__f.tell()

    def read(self: Self, amount: int) -> bytes:
        self.__assert_holder()
        return self.__f.read(amount)

    def write(self: Self, data: bytes) -> int:
        self.__assert_holder()
        return self.__f.write(data)

    def flush(self: Self) -> None:
        self.__assert_holder()
        self.__f.flush()

    def reset_seek(self: Self) -> None:
        self.__assert_no_holder()
        # seeks to the end, so if someone tries to read, it fails immediately (or returns 0 bytes)
        self.__f.seek(0, 2)

    def seek_abs(self: Self, offset: int) -> None:
        self.__assert_holder()
        self.__f.seek(offset)

    def acquire(self: Self, cls: object) -> None:
        self.__assert_no_holder()
        self.__holder = id(cls)

    def release(self: Self, cls: object) -> None:
        self.__assert_holder()
        if self.__holder != id(cls):
            msg = f"ExclusiveIOBase release error: released with different class, than acquired: {self.__holder} != {id(cls)}"
            raise RuntimeError(msg)
        self.__holder = id(None)


class BoundedIO:
    __io: ExclusiveIOBase

    __span: SimpleSpan

    def __init__(self: Self, io: ExclusiveIOBase, span: SimpleSpan) -> None:

        if span.start < 0:
            msg = f"Start negative: {span.start}"
            raise RuntimeError(msg)

        if span.size < 0:
            msg = f"Size negative: {span.size}"
            raise RuntimeError(msg)

        filesize = io.size()

        if span.start > filesize:
            msg = f"Start outside file size: {span.start} > {filesize}"
            raise RuntimeError(msg)

        if span.end > filesize:
            msg = f"End outside file size: {span.end} > {filesize}"
            raise RuntimeError(msg)

        self.__io = io
        self.__span = span

        self.__reset_seek()

    @staticmethod
    def get_new(io_base: BinaryIO, span: SimpleSpan) -> "BoundedIO":
        return BoundedIO(ExclusiveIOBase(io_base), span)

    def __read_exact_bounds_checked(self: Self, amount: int) -> bytes:
        if amount < 0:
            msg = "Invalid checked read, read amount negative"
            raise ValueError(msg)

        current_end = self.__io.tell() + amount
        if current_end > self.__span.end:
            msg = f"Read would overflow bounds [{self.__span.start}, {self.__span.end}]: {current_end} ({current_end - amount} + {amount})"
            raise RuntimeError(msg)

        value = self.__io.read(amount)
        if len(value) != amount:
            msg = f"Read failed to produce {amount} bytes, got {len(value)}"
            raise RuntimeError(msg)

        return value

    def __skip_exact_bounds_checked(self: Self, amount: int) -> None:
        if amount < 0:
            msg = "Invalid checked skip, skip amount negative"
            raise ValueError(msg)

        current_end = self.__io.tell() + amount
        if current_end > self.__span.end:
            msg = f"Skip would overflow bounds [{self.__span.start}, {self.__span.end}]: {current_end} ({current_end - amount}  + {amount})"
            raise RuntimeError(msg)

        self.__io.seek_abs(current_end)

    def __write_exact_bounds_checked(self: Self, data: bytes) -> None:

        current_end = self.__io.tell() + len(data)
        if current_end > self.__span.end:
            msg = f"Write would overflow bounds [{self.__span.start}, {self.__span.end}]: {current_end}"
            raise RuntimeError(msg)

        amount = self.__io.write(data)
        if amount != len(data):
            msg = f"Write failed to write {len(data)} bytes, got {amount}"
            raise RuntimeError(msg)

    def __flush(self: Self) -> None:
        self.__io.flush()

    @property
    def span(self: Self) -> SimpleSpan:
        return self.__span

    def __position_at_start(self: Self) -> None:
        self.__io.seek_abs(self.__span.start)

    def __reset_seek(self: Self) -> None:
        self.__io.reset_seek()

    def r_ctx(
        self: Self,
        *,
        force_entire_read: bool,
    ) -> AbstractContextManager[BoundedIOReadable]:

        def read_impl(amount: int) -> bytes:
            return self.__read_exact_bounds_checked(amount)

        def skip_impl(amount: int) -> None:
            return self.__skip_exact_bounds_checked(amount)

        def enter_impl() -> None:
            self.__io.acquire(self)

            self.__position_at_start()

        def exit_impl(*, have_exception: bool) -> None:
            current_pos = self.__io.tell()

            self.__io.release(self)
            self.__reset_seek()

            if (
                force_entire_read
                and not have_exception
                and current_pos != self.__span.end
            ):
                msg = (
                    f"Not the entire data was read: {current_pos} != {self.__span.end}"
                )
                raise RuntimeError(msg)

        class BoundedIOReadableCtx(AbstractContextManager[BoundedIOReadable]):
            @override
            def __enter__(self: Self) -> BoundedIOReadable:
                class BoundedIOReadableImpl(BoundedIOReadable):
                    def read(self: Self, amount: int) -> bytes:
                        return read_impl(amount)

                    def skip(self: Self, amount: int) -> None:
                        skip_impl(amount)

                enter_impl()

                return BoundedIOReadableImpl()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                exit_impl(have_exception=exc_val is not None)
                return False

        return BoundedIOReadableCtx()

    def rw_ctx(
        self: Self,
        *,
        force_entire_read: bool,
    ) -> AbstractContextManager[BoundedIORW]:

        def read_impl(amount: int) -> bytes:
            return self.__read_exact_bounds_checked(amount)

        def skip_impl(amount: int) -> None:
            return self.__skip_exact_bounds_checked(amount)

        def write_impl(data: bytes) -> None:
            return self.__write_exact_bounds_checked(data)

        def flush_impl() -> None:
            return self.__flush()

        def enter_impl() -> None:
            self.__io.acquire(self)

            self.__position_at_start()

        def exit_impl(*, have_exception: bool) -> None:
            current_pos = self.__io.tell()

            self.__io.release(self)
            self.__reset_seek()

            if (
                force_entire_read
                and not have_exception
                and current_pos != self.__span.end
            ):
                msg = (
                    f"Not the entire data was read: {current_pos} != {self.__span.end}"
                )
                raise RuntimeError(msg)

        class BoundedIORWCtx(AbstractContextManager[BoundedIORW]):
            @override
            def __enter__(self: Self) -> BoundedIORW:
                class BoundedIORWImpl(BoundedIORW):
                    def read(self: Self, amount: int) -> bytes:
                        return read_impl(amount)

                    def skip(self: Self, amount: int) -> None:
                        skip_impl(amount)

                    def write(self: Self, data: bytes) -> None:
                        write_impl(data)

                    def flush(self: Self) -> None:
                        flush_impl()

                enter_impl()

                return BoundedIORWImpl()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool

                exit_impl(have_exception=exc_val is not None)
                return False

        return BoundedIORWCtx()

    def w_ctx(
        self: Self,
    ) -> AbstractContextManager[BoundedIOWriteable]:
        return self.rw_ctx(force_entire_read=False)

    def new_span_io(self: Self, span: SimpleSpan) -> "BoundedIO":
        if span.start < self.__span.start:
            msg = f"Start of new payload io is before parent start: {span.start} <  {self.__span.start}"
            raise RuntimeError(msg)

        if span.end > self.__span.end:
            msg = f"New payload io end overflows parent: {span.end} > {self.__span.end}"
            raise RuntimeError(msg)

        return BoundedIO(self.__io, span)

    def special_checked_filesize(self: Self) -> int:
        filesize = self.__io.size_no_seek()

        if self.__span.end != filesize:
            msg = f"can only span to the filesize end, if the current bound also ends at the end: {self.__span.end} != {filesize}"
            raise RuntimeError(msg)

        return filesize


class ByteOrder(StrEnum):
    NativeNative = "@"
    Native = "="
    Little = "<"
    Big = ">"
    Network = "!"


ISOM_BYTE_ORDER = ByteOrder.Big


def convert_byteorder(order: ByteOrder) -> Literal["little", "big"]:
    match order.value:
        case ByteOrder.NativeNative.value:
            return sys.byteorder
        case ByteOrder.Native.value:
            return sys.byteorder
        case ByteOrder.Little.value:
            return "little"
        case ByteOrder.Big.value:
            return "big"
        case ByteOrder.Network.value:
            return "big"
        case _:
            assert_never(order.value)


def uuid_from_bytes(order: ByteOrder, value: bytes) -> UUID:
    if len(value) != 16:
        msg = f"Invalid length of a UUID: {len(value)}"
        raise RuntimeError(msg)

    byte_order = convert_byteorder(order)

    int_value = int.from_bytes(value, byte_order, signed=False)

    return UUID(int=int_value)


def uuid_to_bytes(order: ByteOrder, uuid: UUID) -> bytes:

    byte_order = convert_byteorder(order)

    result = uuid.int.to_bytes(16, byte_order, signed=False)

    if len(result) != 16:
        msg = f"Invalid length of a UUID: {len(result)}"
        raise RuntimeError(msg)

    return result


class Packable[Type, Underlying = Type](ABC):
    @property
    @abstractmethod
    def pack_str(self: Self) -> str: ...

    @property
    @abstractmethod
    def pack_size(self: Self) -> int: ...

    @abstractmethod
    def to_underlying(self: Self, value: Type) -> Underlying: ...

    @abstractmethod
    def from_underlying(self: Self, value: Underlying) -> Type: ...


class UnsignedInt(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "I"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4

    @override
    def to_underlying(self: Self, value: int) -> int:
        return value

    @override
    def from_underlying(self: Self, value: int) -> int:
        return value


class UnsignedLongLong(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "Q"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 8

    @override
    def to_underlying(self: Self, value: int) -> int:
        return value

    @override
    def from_underlying(self: Self, value: int) -> int:
        return value


class UnsignedShort(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "H"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 2

    @override
    def to_underlying(self: Self, value: int) -> int:
        return value

    @override
    def from_underlying(self: Self, value: int) -> int:
        return value


class Unpacker:

    @staticmethod
    def __unpack_impl[Type, Underlying](
        byte_order: ByteOrder,
        packer: list[Packable[Type, Underlying]],
        value: bytes,
    ) -> tuple[Type, ...]:

        fmts = "".join([pack.pack_str for pack in packer])

        pack_size = sum(pack.pack_size for pack in packer)

        fmt: str = f"{byte_order}{fmts}"

        size = struct.calcsize(fmt)

        if len(value) != size:
            msg = f"Unpacking has wrong input: expected bytes with size {size} but got {len(value)}"
            raise RuntimeError(msg)

        if pack_size != size:
            msg = f"Unpacking implementation error: expected bytes with size {size} but got PACK {pack_size}"
            raise RuntimeError(msg)

        result: tuple[Underlying, ...] = struct.unpack(fmt, value)

        return tuple(
            p.from_underlying(value) for value, p in zip(result, packer, strict=True)
        )

    @staticmethod
    def unpack_sized[Type, Underlying](
        byte_order: ByteOrder,
        packer: list[Packable[Type, Underlying]],
        value: bytes,
        size: int,
    ) -> tuple[Type, ...]:

        if len(packer) != size:
            msg = f"Expected unpack to produce {size} values, but got {len(packer)}"
            raise RuntimeError(msg)

        result = Unpacker.__unpack_impl(byte_order, packer, value)

        if len(result) != size:
            msg = f"Expected unpack to produce {size} values, but got {len(result)}"
            raise RuntimeError(msg)

        return result

    @staticmethod
    def unpack_one[Type, Underlying](
        byte_order: ByteOrder,
        packer: Packable[Type, Underlying],
        value: bytes,
    ) -> Type:
        result = Unpacker.unpack_sized(byte_order, [packer], value, size=1)

        return result[0]

    @staticmethod
    def unpack_two[Type1, Underlying1, Type2, Underlying2](
        byte_order: ByteOrder,
        packer: tuple[Packable[Type1, Underlying1], Packable[Type2, Underlying2]],
        value: bytes,
    ) -> tuple[Type1, Type2]:
        packers: list[Packable[Type1, Underlying1]] = cast(
            list[Packable[Type1, Underlying1]],
            [*packer],
        )
        result: tuple[Type1, ...] = Unpacker.unpack_sized(
            byte_order,
            packers,
            value,
            size=2,
        )

        return cast(tuple[Type1, Type2], result)


class Packer:

    @staticmethod
    def __pack_impl[Type, Underlying](
        byte_order: ByteOrder,
        packer: list[Packable[Type, Underlying]],
        values: list[Type],
        size: int,
    ) -> bytes:

        if len(packer) != len(values):
            msg = f"Expected pack to use {len(packer)} values, but got {len(values)}"
            raise RuntimeError(msg)

        fmts = "".join([pack.pack_str for pack in packer])

        pack_size = sum(pack.pack_size for pack in packer)

        fmt: str = f"{byte_order}{fmts}"

        size = struct.calcsize(fmt)

        if pack_size != size:
            msg = f"Packing implementation error: expected bytes with size {size} but got PACK {pack_size}"
            raise RuntimeError(msg)

        value_underlying: list[Underlying] = [
            p.to_underlying(v) for v, p in zip(values, packer, strict=True)
        ]

        value = struct.pack(fmt, *value_underlying)

        if len(value) != size:
            msg = f"Packing has wrong input: expected bytes with size {size} but got {len(value)}"
            raise RuntimeError(msg)

        return value

    @staticmethod
    def pack_one[Type, Underlying](
        byte_order: ByteOrder,
        packer: Packable[Type, Underlying],
        value: Type,
        size: int,
    ) -> bytes:
        return Packer.__pack_impl(byte_order, [packer], [value], size=size)

    @staticmethod
    def pack_two[Type1, Underlying1, Type2, Underlying2](
        byte_order: ByteOrder,
        packer: tuple[Packable[Type1, Underlying1], Packable[Type2, Underlying2]],
        values: tuple[Type1, Type2],
        size: int,
    ) -> bytes:
        packers: list[Packable[Type1, Underlying1]] = cast(
            list[Packable[Type1, Underlying1]],
            [*packer],
        )

        values_typed: list[Type1] = cast(list[Type1], [*values])

        return Packer.__pack_impl(byte_order, packers, values_typed, size=size)
