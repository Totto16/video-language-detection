from io import BytesIO
from pathlib import Path
from typing import Optional

from fixtures import TempVideoFiles, mark_as_used, mkv_test_parse_files
from pytest_subtests import SubTests
from test_helper import ErrResult, OkResult

from content.tagger.mkv_tagger import BitIterator, EBMLVarInt
from content.tagger.parser import BoundedIO, SimpleSpan
from content.tagger.schema.parser import (
    EBMLElementDescription,
    EBMLElementType,
    EBMLSchemaRange,
    EBMLSchemaRangeBound,
    EBMLSchemaRangeElem,
    EBMLSchemaRangeNot,
    WrapperInt,
    xml_any_range_result,
)

mark_as_used(mkv_test_parse_files)


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


def test_mkv_tagger_parse_ebml_schema_range(
    subtests: SubTests,
) -> None:
    tests: list[tuple[str, EBMLSchemaRange[int]]] = [
        ("1", EBMLSchemaRange(1)),
        ("    2   ", EBMLSchemaRange(2)),
        ("   not  2   ", EBMLSchemaRange(EBMLSchemaRangeNot(2))),
        (
            "2-4",
            EBMLSchemaRange(
                (
                    EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Inclusive),
                    EBMLSchemaRangeElem(4, EBMLSchemaRangeBound.Inclusive),
                ),
            ),
        ),
        (
            ">=2",
            EBMLSchemaRange(
                (EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Inclusive), None),
            ),
        ),
        (
            ">2",
            EBMLSchemaRange(
                (EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Exclusive), None),
            ),
        ),
        (
            "<2",
            EBMLSchemaRange(
                (None, EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Exclusive)),
            ),
        ),
        (
            "<=2",
            EBMLSchemaRange(
                (None, EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Inclusive)),
            ),
        ),
        (
            ">2,<4",
            EBMLSchemaRange(
                (
                    EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Exclusive),
                    EBMLSchemaRangeElem(4, EBMLSchemaRangeBound.Exclusive),
                ),
            ),
        ),
        (
            ">=2,<=4",
            EBMLSchemaRange(
                (
                    EBMLSchemaRangeElem(2, EBMLSchemaRangeBound.Inclusive),
                    EBMLSchemaRangeElem(4, EBMLSchemaRangeBound.Inclusive),
                ),
            ),
        ),
    ]

    for inp, result in tests:
        with subtests.test("EBML Element Schema Range parsing"):
            value = xml_any_range_result(inp, WrapperInt())

            assert value == OkResult(result), f"Input was {inp}"


def test_mkv_tagger_parse_ebml_schema_range_errors(
    subtests: SubTests,
) -> None:
    tests: list[tuple[str, str]] = [
        ("", "Invalid bound: Invalid bounded number: ''"),
        ("not 1 not", "Invalid number after not: '1not'"),
        ("1-3-4", "Invalid syntax, only one '-' allowed: '1-3-4'"),
        ("h-2", "Invalid starting number: 'h'"),
        ("1-g", "Invalid ending number: 'g'"),
        ("<=r2", "Invalid bound: Invalid bound number: 'r2'"),
        ("2,4", "Invalid starting bound: Invalid bounded number: '2'"),
        ("1,3,4", "Invalid syntax, only one ',' allowed: '1,3,4'"),
        ("h,>2", "Invalid starting bound: Invalid bounded number: 'h'"),
        (">1,g", "Invalid ending bound: Invalid bounded number: 'g'"),
        (
            "<=2,<=4",
            "First boundary has to be the lower boundary: but was: LE: '<=2,<=4'",
        ),
        (
            ">=2,>=4",
            "Second boundary has to be the upper boundary: but was: GE: '>=2,>=4'",
        ),
        ("4-1", "Invalid range order: first number is bigger: 4 > 1"),
    ]

    for inp, result in tests:
        with subtests.test("EBML Element Schema Range parsing errors"):
            value = xml_any_range_result(inp, WrapperInt())

            assert value == ErrResult()

            assert value.as_err() == result


def test_mkv_tagger_todo(
    subtests: SubTests,
    mkv_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, int]] = list(
        zip(
            mkv_test_parse_files.data,
            [42],
            strict=True,
        ),
    )

    for file, result in test_files:
        with subtests.test("video gets parsed correctly"):
            assert file != ""


def test_mkv_tagger_ebml_schema_test_schema_parse(
    subtests: SubTests,
    mkv_test_parse_files: TempVideoFiles,
) -> None:
    # TODO: test schema extraction
    # EBML Header Elements
    EBMLHeaderElements: list[EBMLElementDescription] = [
        EBMLElementDescription(
            name="EBML",
            id=0x1A45DFA3,
            occurrences=1,
            type=EBMLElementType.Master,
            description="Set the EBML characteristics of the data to follow. Each EBML Document has to start with this.",
        ),
    ]
