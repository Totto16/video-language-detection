import os
import struct
import sys
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from enum import StrEnum
from io import BufferedIOBase
from types import TracebackType
from typing import Literal, Optional, Protocol, Self, assert_never, cast, override
from uuid import UUID

from helper.translation import get_translator

_ = get_translator()


class BoundedIOReadable(Protocol):
    def read(self: Self, amount: int) -> bytes: ...

    def skip(self: Self, amount: int) -> None: ...


class BoundedIOWriteable(Protocol):
    def write(self: Self, data: bytes) -> None: ...

    def flush(self: Self) -> None: ...


class BoundedIORW(BoundedIOWriteable, BoundedIOReadable):
    pass


class ExclusiveIOBase:
    __f: BufferedIOBase

    __holder: int

    def __init__(self: Self, f: BufferedIOBase) -> None:
        self.__f = f

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

    __start: int
    __size: int

    def __init__(self: Self, io: ExclusiveIOBase, start: int, size: int) -> None:

        if start < 0:
            msg = f"Start negative: {start}"
            raise RuntimeError(msg)

        if size < 0:
            msg = f"Size negative: {size}"
            raise RuntimeError(msg)

        filesize = io.size()

        if start > filesize:
            msg = f"Start outside file size: {start} > {filesize}"
            raise RuntimeError(msg)

        end = start + size

        if end > filesize:
            msg = f"End outside file size: {end} > {filesize}"
            raise RuntimeError(msg)

        self.__io = io
        self.__start = start
        self.__size = size

        self.__reset_seek()

    @staticmethod
    def get_new(f: BufferedIOBase, start: int, size: int) -> "BoundedIO":
        return BoundedIO(ExclusiveIOBase(f), start, size)

    def __read_exact_bounds_checked(self: Self, amount: int) -> bytes:
        if amount < 0:
            msg = "Invalid checked read, read amount negative"
            raise ValueError(msg)

        current_end = self.__io.tell() + amount
        if current_end > self.end:
            msg = f"Read would overflow bounds [{self.__start}, {self.end}]: {current_end}"
            raise RuntimeError(msg)

        value = self.__io.read(amount)
        if len(value) != amount:
            msg = f"Read failed to produce {amount} bytes, got {len(value)}"
            raise RuntimeError(msg)

        return value

    def __skip_exact_bounds_checked(self: Self, amount: int) -> None:
        if amount < 0:
            msg = "Invalid checked read, read amount negative"
            raise ValueError(msg)

        current_end = self.__io.tell() + amount
        if current_end > self.end:
            msg = f"Read would overflow bounds [{self.__start}, {self.end}]: {current_end}"
            raise RuntimeError(msg)

        self.__io.seek_abs(current_end)

    def __write_exact_bounds_checked(self: Self, data: bytes) -> None:

        current_end = self.__io.tell() + len(data)
        if current_end > self.end:
            msg = f"Write would overflow bounds [{self.__start}, {self.end}]: {current_end}"
            raise RuntimeError(msg)

        amount = self.__io.write(data)
        if amount != len(data):
            msg = f"Write failed to write {len(data)} bytes, got {amount}"
            raise RuntimeError(msg)

    def __flush(self: Self) -> None:
        self.__io.flush()

    @property
    def end(self: Self) -> int:
        return self.__start + self.__size

    @property
    def start(self: Self) -> int:
        return self.__start

    def __position_at_start(self: Self) -> None:
        self.__io.seek_abs(self.__start)

    def __reset_seek(self: Self) -> None:
        self.__io.reset_seek()

    def r_ctx(
        self: Self,
        *,
        force_entire_read: bool,
    ) -> AbstractContextManager[BoundedIOReadable]:

        parent = self

        class BoundedIOReadableCtx(AbstractContextManager[BoundedIOReadable]):
            @override
            def __enter__(self: Self) -> BoundedIOReadable:
                class BoundedIOReadableImpl(BoundedIOReadable):
                    def read(self: Self, amount: int) -> bytes:
                        return parent.__read_exact_bounds_checked(amount)

                    def skip(self: Self, amount: int) -> None:
                        parent.__skip_exact_bounds_checked(amount)

                parent.__io.acquire(self)

                parent.__position_at_start()

                return BoundedIOReadableImpl()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool

                if force_entire_read:
                    current_pos = parent.__io.tell()
                    if current_pos != parent.end:
                        msg = f"Not the entire data was read: {current_pos} != {parent.end}"
                        raise RuntimeError(msg)

                parent.__io.release(self)
                parent.__reset_seek()

                return False

        return BoundedIOReadableCtx()

    def rw_ctx(
        self: Self,
        *,
        force_entire_read: bool,
    ) -> AbstractContextManager[BoundedIORW]:

        parent = self

        class BoundedIORWCtx(AbstractContextManager[BoundedIORW]):
            @override
            def __enter__(self: Self) -> BoundedIORW:
                class BoundedIORWImpl(BoundedIORW):
                    def read(self: Self, amount: int) -> bytes:
                        return parent.__read_exact_bounds_checked(amount)

                    def skip(self: Self, amount: int) -> None:
                        parent.__skip_exact_bounds_checked(amount)

                    def write(self: Self, data: bytes) -> None:
                        parent.__write_exact_bounds_checked(data)

                    def flush(self: Self) -> None:
                        parent.__flush()

                parent.__io.acquire(self)

                parent.__position_at_start()

                return BoundedIORWImpl()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool

                if force_entire_read:
                    current_pos = parent.__io.tell()
                    if current_pos != parent.end:
                        msg = f"Not the entire data was read: {current_pos} != {parent.end}"
                        raise RuntimeError(msg)

                parent.__io.release(self)
                parent.__reset_seek()

                return False

        return BoundedIORWCtx()

    def w_ctx(
        self: Self,
    ) -> AbstractContextManager[BoundedIOWriteable]:
        return self.rw_ctx(force_entire_read=False)

    def new_payload_io(self: Self, start: int, size: int) -> "BoundedIO":
        if start < self.__start:
            msg = f"Start of new payload io is before parent start: {start} <  {self.__start}"
            raise RuntimeError(msg)

        if start + size != self.end:
            msg = f"New payload io isn't correctly sized, it doesnÄt rech the end of the üarentz: {start + size} != {self.end}"
            raise RuntimeError(msg)

        return BoundedIO(self.__io, start, size)


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
