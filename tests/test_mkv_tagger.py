from io import BytesIO
from typing import Optional

from pytest_subtests import SubTests
from test_helper import ErrResult, OkResult

from content.tagger.mkv_tagger import BitIterator, EBMLVarInt
from content.tagger.parser import BoundedIO, SimpleSpan


def test_mkv_tagger_bit_iterator(
    subtests: SubTests,
) -> None:
    tests: list[bytes] = [b"\00\xff", bytes([0b10101010, 0b01010101])]

    for byte in tests:
        with subtests.test("BitIterator works correctly"):
            bit_iterator = BitIterator(byte)

            result = 0

            for b in bit_iterator:
                result = (result << 1) + b.int

            byte_val = int.from_bytes(byte, "big", signed=False)

            assert result == byte_val


def test_mkv_tagger_var_int_bytes_required(
    subtests: SubTests,
) -> None:
    tests: list[tuple[int, int]] = [
        (0b1111111, 1),
        (0b10000000, 2),
        (0b11111111111111, 2),
        (0b100000000000000, 3),
        (0xFFFFFFFFFFFFFF, 8),
        (0xFFFFFFFFFFFFFF + 1, 9),
    ]

    for int_val, amount in tests:
        with subtests.test("VarInt required bytes"):
            var_int = EBMLVarInt(int_val)

            assert var_int.minmum_bytes_required() == amount


def test_mkv_tagger_parse_var_int(
    subtests: SubTests,
) -> None:
    tests: list[tuple[bytes, int]] = [
        # spec tests
        (b"\x82", 2),
        (b"\x40\x02", 2),
        (b"\x20\x00\x02", 2),
        (b"\x10\x00\x00\x02", 2),
        # custom tests
        (b"\x01\x00\x00\x00\x00\x00\x00\x02", 2),
    ]

    for byte, result in tests:
        with subtests.test("VarInt parsing"):
            buf_io = BytesIO(byte)
            io = BoundedIO.get_new(
                buf_io,
                span=SimpleSpan(0, len(byte)),
            )
            var_int_res = EBMLVarInt.from_io(io)

            assert var_int_res == OkResult()

            var_int, bytes_used = var_int_res.as_ok()

            assert bytes_used == len(byte)

            assert var_int == result


def test_mkv_tagger_parse_var_int_errors(
    subtests: SubTests,
) -> None:
    tests: list[tuple[bytes, str]] = [
        (b"\x00", "The first byte of a VarInt can't be 0x00"),
        (b"\x01\x00", "Not enough data for VarInt: need 8 bytes but got 2"),
        (b"", "Not enough data for VarInt"),
    ]

    for byte, result in tests:
        with subtests.test("VarInt parsing ERRORS"):
            buf_io = BytesIO(byte)
            io = BoundedIO.get_new(
                buf_io,
                span=SimpleSpan(0, len(byte)),
            )
            var_int_res = EBMLVarInt.from_io(io)

            assert var_int_res == ErrResult()

            err = var_int_res.as_err()

            assert err == result


def test_mkv_tagger_parse_element_id(
    subtests: SubTests,
) -> None:
    tests: list[tuple[bytes, Optional[str]]] = [
        # spec tests
        (b"\x80", "Value is NULL"),
        (b"\x40\x00", "Value is NULL"),
        (b"\x81", None),
        (
            b"\x40\x01",
            "Value 1 uses too much bytes: 1 bytes are the minimum, but used 2",
        ),
        (b"\xaf", None),
        (
            b"\x40\x3f",
            "Value 63 uses too much bytes: 1 bytes are the minimum, but used 2",
        ),
        (b"\xff", "Value is 0xFF..FF"),
        (b"\x40\x7f", None),
        # custom tests
        (
            b"\x20\x00\x7f",
            "Value 127 uses too much bytes: 1 bytes are the minimum, but used 3",
        ),
    ]

    for byte, result in tests:
        with subtests.test("EBML Element ID parsing"):
            buf_io = BytesIO(byte)
            io = BoundedIO.get_new(
                buf_io,
                span=SimpleSpan(0, len(byte)),
            )
            var_int_res = EBMLVarInt.from_io(io)

            assert var_int_res == OkResult()

            var_int, bytes_used = var_int_res.as_ok()

            assert bytes_used == len(byte)

            is_valid_element_id = var_int.is_valid_element_id(bytes_used)

            assert is_valid_element_id == result
