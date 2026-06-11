import os
import struct
import sys
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from enum import StrEnum
from io import BufferedIOBase, BytesIO
from types import TracebackType
from typing import Literal, Optional, Protocol, Self, assert_never, cast, override
from uuid import UUID

from helper.translation import get_translator

_ = get_translator()


class BoundedReaderReadable(Protocol):
    def read(self: Self, amount: int) -> bytes: ...

    def skip(self: Self, amount: int) -> None: ...


class ExclusiveIOBase:
    __f: BufferedIOBase

    __used: bool

    def __init__(self: Self, f: BufferedIOBase) -> None:
        self.__f = f

        self.__used = False


class BoundedReader:
    __io: ExclusiveIOBase

    __start: int
    __size: int

    def __init__(self: Self, io: ExclusiveIOBase, start: int, size: int) -> None:
        if io.writable():
            msg = "Only readonly streams supported atm"
            raise RuntimeError(msg)

        if start < 0:
            msg = f"Start negative: {start}"
            raise RuntimeError(msg)

        if size < 0:
            msg = f"Size negative: {size}"
            raise RuntimeError(msg)

        io.seek(0, 2)
        filesize = io.tell()

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

        self.__position_at_end()

    @staticmethod
    def get_new(f: BufferedIOBase, start: int, size: int) -> "BoundedReader":
        if f.writable():
            msg = "Only readonly streams supported atm"
            raise RuntimeError(msg)

        return BoundedReader(ExclusiveIOBase(f), start, size)

    def __read_exact_bounds_checked(self: Self, amount: int) -> bytes:
        if not self.__TODO________:
            msg = f"Implementation error: {self.__TODO________} should be True"
            raise RuntimeError(msg)

        if amount < 0:
            msg = "Invalid checked read, read amount negative"
            raise ValueError(msg)

        if self.__io.tell() + amount > self.__end:
            msg = f"Read would overflow bounds [{self.__start}, {self.__end}]: {self.__io.tell() + amount}"
            raise RuntimeError(msg)

        value = self.__io.read(amount)
        if len(value) != amount:
            msg = f"Read failed to produce {amount} bytes, got {len(value)}"
            raise RuntimeError(msg)

        return value

    @property
    def end(self: Self) -> int:
        return self.__start + self.__size

    @property
    def start(self: Self) -> int:
        return self.__start

    def __position_at_start(self: Self) -> None:
        self.__io.seek(self.__start)

    def __position_at_end(self: Self) -> None:
        self.__io.seek(0, 2)

    def ctx(
        self: Self,
        *,
        force_entire_read: bool,
    ) -> AbstractContextManager[BoundedReaderReadable]:

        parent = self

        class BoundedReaderReadableCtx(AbstractContextManager[BoundedReaderReadable]):
            @override
            def __enter__(self: Self) -> BoundedReaderReadable:
                parent.__position_at_start()

                class BoundedReaderReadableImpl(BoundedReaderReadable):
                    def read(self: Self, amount: int) -> bytes:
                        return parent.__read_exact_bounds_checked(amount)

                    def skip(self: Self, amount: int) -> None:
                        parent.__read_exact_bounds_checked(amount)

                if parent.__TODO______:
                    msg = f"Implementation error: {parent.__TODO______} should be False"
                    raise RuntimeError(msg)

                parent.__TODO______ = True

                return BoundedReaderReadableImpl()

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                parent.__position_at_end()

                if not parent.__TODO______:
                    msg = f"Implementation error: {parent.__TODO______} should be True"
                    raise RuntimeError(msg)

                parent.__TODO______ = False

                if force_entire_read:
                    current_pos = parent.__io.tell()
                    if current_pos != parent.__end:
                        msg = f"Not the entire data was read: {current_pos} != {parent.__end}"
                        raise RuntimeError(msg)

                return False

        return BoundedReaderReadableCtx()

    def new_payload_reader(self: Self, start: int, size: int) -> "BoundedReader":
        if start < self.__start:
            raise NotImplementedError("S")

        if start + size != self.__end:
            raise NotImplementedError("S")

        return BoundedReader(self.__io, start, size)


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
