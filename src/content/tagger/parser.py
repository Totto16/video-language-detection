import struct
import sys
from abc import ABC, abstractmethod
from enum import StrEnum
from io import BufferedIOBase
from typing import Any, Literal, Self, assert_never, override
from uuid import UUID

from helper.translation import get_translator

_ = get_translator()


def read_checked(f: BufferedIOBase, amount: int) -> bytes:
    if amount < 0:
        msg = "Invalid checked read, read amount negative"
        raise ValueError(msg)

    value = f.read(amount)
    if len(value) != amount:
        msg = f"Read failed to produce {amount} bytes, got {len(value)}"
        raise RuntimeError(msg)

    return value


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


class Packable[T](ABC):
    @property
    @abstractmethod
    def pack_str(self: Self) -> str: ...

    @property
    @abstractmethod
    def pack_size(self: Self) -> int: ...


class UnsignedInt(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "I"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4


class UnsignedLongLong(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "Q"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 8


class UnsignedShort(Packable[int]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "H"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 2


class Unpacker:

    @staticmethod
    def __unpack_impl(
        byte_order: ByteOrder,
        packer: list[Packable[Any]],
        value: bytes,
    ) -> tuple[Any, ...]:

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

        return struct.unpack(fmt, value)

    @staticmethod
    def unpack_sized(
        byte_order: ByteOrder,
        packer: list[Packable[Any]],
        value: bytes,
        size: int,
    ) -> tuple[Any, ...]:

        if len(packer) != size:
            msg = f"Expected unpack to produce {size} values, but got {len(packer)}"
            raise RuntimeError(msg)

        result = Unpacker.__unpack_impl(byte_order, packer, value)

        if len(result) != size:
            msg = f"Expected unpack to produce {size} values, but got {len(result)}"
            raise RuntimeError(msg)

        return result

    @staticmethod
    def unpack_one[A](
        byte_order: ByteOrder,
        packer: Packable[A],
        value: bytes,
    ) -> A:
        result = Unpacker.unpack_sized(byte_order, [packer], value, size=1)

        return result[0]

    @staticmethod
    def unpack_two[A, B](
        byte_order: ByteOrder,
        packer: tuple[Packable[A], Packable[B]],
        value: bytes,
    ) -> tuple[A, B]:
        return Unpacker.unpack_sized(byte_order, [*packer], value, size=2)


class Packer:

    @staticmethod
    def __pack_impl(
        byte_order: ByteOrder,
        packer: list[Packable[Any]],
        values: list[Any],
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

        value = struct.pack(fmt, *values)

        if len(value) != size:
            msg = f"Packing has wrong input: expected bytes with size {size} but got {len(value)}"
            raise RuntimeError(msg)

        return value

    @staticmethod
    def pack_one[A](
        byte_order: ByteOrder,
        packer: Packable[A],
        value: A,
        size: int,
    ) -> bytes:
        return Packer.__pack_impl(byte_order, [packer], [value], size=size)

    @staticmethod
    def pack_two[A, B](
        byte_order: ByteOrder,
        packer: tuple[Packable[A], Packable[B]],
        values: tuple[A, B],
        size: int,
    ) -> bytes:
        return Packer.__pack_impl(byte_order, [*packer], [*values], size=size)
