from collections.abc import Callable
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Self, override

from conftest import FancyEq
from fixtures import TempVideoFiles, mark_as_used, mkv_test_parse_files
from pytest_subtests import SubTests
from test_helper import ErrResult, OkResult

from content.tagger.mkv_tagger import (
    BitIterator,
    EBMLElementID,
    EBMLVarInt,
    is_mkv_file,
)
from content.tagger.parser import BoundedIO, SimpleSpan
from content.tagger.schema.parser import (
    DefaultEmpty,
    DocType,
    EBMLAdvancedElementTypeBinary,
    EBMLAdvancedElementTypeInteger,
    EBMLAdvancedElementTypeMaster,
    EBMLAdvancedElementTypeString,
    EBMLElementDescription,
    EBMLElementDescriptionGeneric,
    EBMLElementIDParsed,
    EBMLElementType,
    EBMLOccurrences,
    EBMLSchemaRange,
    EBMLSchemaRangeBound,
    EBMLSchemaRangeElem,
    EBMLSchemaRangeNot,
    EBMLSpec,
    LengthRange,
    WrapperInt,
    ebml_read_spec_xml,
    xml_any_range_result,
)
from helper.decorator import decorate_class
from helper.result import Err, Ok, Result

if TYPE_CHECKING:
    from pathlib import Path

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
            element_id_res = EBMLElementID.from_io(io)

            assert element_id_res == OkResult()

            element_id, bytes_used = element_id_res.as_ok()

            assert bytes_used == len(byte)

            is_valid_element_id = element_id.is_valid(bytes_used)

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
        (
            ">=-2,<=4",
            EBMLSchemaRange(
                (
                    EBMLSchemaRangeElem(-2, EBMLSchemaRangeBound.Inclusive),
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


def test_mkv_tagger_parsing(
    subtests: SubTests,
    mkv_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, str, int]] = list(
        zip(
            [f for f, _ in mkv_test_parse_files.data],
            [nm for _, nm in mkv_test_parse_files.data],
            [42],
            strict=True,
        ),
    )

    for file, name, result in test_files:
        with subtests.test(f"video gets parsed correctly: {name}"):
            # TODO
            assert file != ""


def test_mkv_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Read would overflow bounds [0, 0]: 8 (0 + 8)"),
        (
            b"helloworld",
            "Invalid MP4 Box size: It overflows the parent box: 1751477356 > 10",
        ),
        (b"ftyp    ", "Atom name not valid b'    '"),
        (b"\x00\x00\x00\x04ftyp", "Invalid box: size too small: 4"),
        (
            b"\x00\x00\x00\x0eftypabcddcba",
            "Read would overflow bounds [8, 14]: 16 (12 + 4)",
        ),
        (
            b"\x00\x00\x00\x10ftypabcddcba",
            "ISOM/MP42 file has valid box, but invalid major_brand: b'abcd'",
        ),
    ]

    for data, err in test_data:
        with subtests.test("invalid video gets detected correctly"):
            io = BytesIO(data)
            res = is_mkv_file(io)

            assert res is not None, "valid mp4 is incorrect here"

            assert res == err, "incorrect error"


@decorate_class(slots=True)
class EBMLTestSpec(FancyEq):
    __spec: EBMLSpec

    def __init__(self: Self, doc_type: DocType) -> None:
        self.__spec = EBMLSpec(doc_type)

    @property
    def spec(self: Self) -> EBMLSpec:
        return self.__spec

    def append(self: Self, element: EBMLElementDescription) -> None:
        self.__spec.append(element)

    def extend(self: Self, elements: list[EBMLElementDescription]) -> None:
        for elem in elements:
            self.__spec.append(elem)

    @staticmethod
    def __is_elem_eq(
        elem1: EBMLElementDescription,
        elem2: EBMLElementDescription,
    ) -> Result[None, list[str]]:

        if elem1.name != elem2.name:
            return Err(
                [
                    "Element name doesn't match:",
                    str(elem1.name),
                    str(elem2.name),
                ],
            )

        if elem1.type.type != elem2.type.type:
            return Err(
                [
                    "Element type doesn't match:",
                    f"Element name: {elem1.name}",
                    str(elem1.type.type),
                    str(elem2.type.type),
                ],
            )

        if elem1.type != elem2.type:
            return Err(
                [
                    "Element type doesn't match:",
                    f"Element name: {elem1.name}",
                    str(elem1.type),
                    str(elem2.type),
                ],
            )

        if elem1.id != elem2.id:
            return Err(
                [
                    "Element id doesn't match:",
                    f"Element name: {elem1.name}",
                    str(elem1.id),
                    str(elem2.id),
                ],
            )

        keys: list[str] = [
            "occurrences",
            "description",
            "unknown_size_allowed",
            "versions",
            "path",
            "recurring",
            "recursive",
        ]

        for key in keys:
            value1 = getattr(elem1, key)
            value2 = getattr(elem2, key)

            if value1 != value2:
                return Err(
                    [
                        f"Element attributes {key} doesn't match:",
                        f"Element name: {elem1.name}",
                        str(value1),
                        str(value2),
                    ],
                )

        return Ok(None)

    def __eq_other(self: Self, other_value: EBMLSpec) -> Result[None, list[str]]:

        if self.spec.doc_type != other_value.doc_type:
            return Err(
                [
                    "DocType doesn't match:",
                    str(self.spec.doc_type),
                    str(other_value.doc_type),
                ],
            )

        if len(self.__spec.elements) != len(other_value.elements):
            return Err(
                [
                    "Length of elements is not eq:",
                    str(len(self.__spec.elements)),
                    str(len(other_value.elements)),
                ],
            )

        for e1, e2 in zip(self.__spec.elements, other_value.elements, strict=True):
            res = EBMLTestSpec.__is_elem_eq(e1, e2)
            if res.err():
                return res

        return Ok(None)

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, EBMLSpec):
            return (True, lambda: self.__eq_other(other))

        if isinstance(other, EBMLTestSpec):
            return (True, lambda: self.__eq_other(other.spec))

        return (False, lambda: Err(["Invalid compare type", str(type(other))]))

    def __eq__(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[1]().ok()

    @override
    def supports_fancy_eq(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[0]

    @override
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]:
        supports_fancy_eq, cb = self.__eq_impl(other)
        assert supports_fancy_eq
        return cb().err_or(None)

    def __str__(self: Self) -> str:
        return "<EBMLTestSpec {self.__spec}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __hash__(self: Self) -> int:
        return hash(("EBMLTestSpec", self.__spec))


def test_mkv_tagger_ebml_schema_test_schema_parse(
    subtests: SubTests,
) -> None:

    with subtests.test("EBML schema parser: MKV schema is correct"):

        def get_mkv_spec() -> EBMLTestSpec:

            result = EBMLTestSpec(doc_type=DocType(type="matroska", version=4))

            return result

        EBMLMKVSpec = ebml_read_spec_xml("mkv/ebml_matroska.xml")

        mkv_spec = get_mkv_spec()

        # TODO: this needs so much boilerplate, but implement a full comparison
        # assert EBMLMKVSpec == mkv_spec.spec

        assert EBMLMKVSpec.doc_type == mkv_spec.spec.doc_type

        assert len(EBMLMKVSpec.elements) == 262
    with subtests.test("EBML schema parser: EBML schema is correct"):

        def get_ebml_spec() -> EBMLTestSpec:

            result = EBMLTestSpec(doc_type=DocType(type="ebml", version=1))

            result.extend(
                [
                    EBMLElementDescriptionGeneric(
                        name="EBML",
                        id=EBMLElementIDParsed.from_checked(440786851),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeMaster(type=EBMLElementType.Master),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="EBMLVersion",
                        id=EBMLElementIDParsed.from_checked(17030),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=1,
                            range=EBMLSchemaRange(EBMLSchemaRangeNot(0)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\EBMLVersion",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="EBMLReadVersion",
                        id=EBMLElementIDParsed.from_checked(17143),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=1,
                            range=EBMLSchemaRange(1),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\EBMLReadVersion",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="EBMLMaxIDLength",
                        id=EBMLElementIDParsed.from_checked(17138),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=4,
                            range=EBMLSchemaRange(
                                (
                                    EBMLSchemaRangeElem(
                                        value=4,
                                        bound=EBMLSchemaRangeBound.Inclusive,
                                    ),
                                    None,
                                ),
                            ),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\EBMLMaxIDLength",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="EBMLMaxSizeLength",
                        id=EBMLElementIDParsed.from_checked(17139),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=8,
                            range=EBMLSchemaRange(EBMLSchemaRangeNot(0)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\EBMLMaxSizeLength",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocType",
                        id=EBMLElementIDParsed.from_checked(17026),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeString(
                            type=EBMLElementType.String,
                            default=DefaultEmpty(),
                            length=LengthRange(
                                EBMLSchemaRange(
                                    (
                                        EBMLSchemaRangeElem(
                                            value=0,
                                            bound=EBMLSchemaRangeBound.Exclusive,
                                        ),
                                        None,
                                    ),
                                ),
                            ),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocType",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocTypeVersion",
                        id=EBMLElementIDParsed.from_checked(17031),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=1,
                            range=EBMLSchemaRange(EBMLSchemaRangeNot(0)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocTypeVersion",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocTypeReadVersion",
                        id=EBMLElementIDParsed.from_checked(17029),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=1,
                            range=EBMLSchemaRange(EBMLSchemaRangeNot(0)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocTypeReadVersion",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocTypeExtension",
                        id=EBMLElementIDParsed.from_checked(17025),
                        occurrences=EBMLOccurrences(
                            EBMLSchemaRange(
                                (
                                    EBMLSchemaRangeElem(
                                        value=0,
                                        bound=EBMLSchemaRangeBound.Inclusive,
                                    ),
                                    None,
                                ),
                            ),
                        ),
                        type=EBMLAdvancedElementTypeMaster(type=EBMLElementType.Master),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocTypeExtension",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocTypeExtensionName",
                        id=EBMLElementIDParsed.from_checked(17027),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeString(
                            type=EBMLElementType.String,
                            default=DefaultEmpty(),
                            length=LengthRange(
                                EBMLSchemaRange(
                                    (
                                        EBMLSchemaRangeElem(
                                            value=0,
                                            bound=EBMLSchemaRangeBound.Exclusive,
                                        ),
                                        None,
                                    ),
                                ),
                            ),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocTypeExtension\\DocTypeExtensionName",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="DocTypeExtensionVersion",
                        id=EBMLElementIDParsed.from_checked(17028),
                        occurrences=EBMLOccurrences(EBMLSchemaRange(1)),
                        type=EBMLAdvancedElementTypeInteger(
                            type=EBMLElementType.UnsignedInteger,
                            default=DefaultEmpty(),
                            range=EBMLSchemaRange(EBMLSchemaRangeNot(0)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\EBML\\DocTypeExtension\\DocTypeExtensionVersion",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="Void",
                        id=EBMLElementIDParsed.from_checked(236),
                        occurrences=EBMLOccurrences(
                            EBMLSchemaRange(
                                (
                                    EBMLSchemaRangeElem(
                                        value=0,
                                        bound=EBMLSchemaRangeBound.Inclusive,
                                    ),
                                    None,
                                ),
                            ),
                        ),
                        type=EBMLAdvancedElementTypeBinary(
                            type=EBMLElementType.Binary,
                            default=DefaultEmpty(),
                            length=None,
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\(-\\)Void",
                        recurring=False,
                        recursive=False,
                    ),
                    EBMLElementDescriptionGeneric(
                        name="CRC-32",
                        id=EBMLElementIDParsed.from_checked(191),
                        occurrences=EBMLOccurrences(
                            EBMLSchemaRange(
                                (
                                    EBMLSchemaRangeElem(
                                        value=0,
                                        bound=EBMLSchemaRangeBound.Inclusive,
                                    ),
                                    EBMLSchemaRangeElem(
                                        value=1,
                                        bound=EBMLSchemaRangeBound.Inclusive,
                                    ),
                                ),
                            ),
                        ),
                        type=EBMLAdvancedElementTypeBinary(
                            type=EBMLElementType.Binary,
                            default=DefaultEmpty(),
                            length=LengthRange(EBMLSchemaRange(4)),
                        ),
                        description=None,
                        unknown_size_allowed=False,
                        versions=EBMLSchemaRange(
                            (
                                EBMLSchemaRangeElem(
                                    value=1,
                                    bound=EBMLSchemaRangeBound.Inclusive,
                                ),
                                None,
                            ),
                        ),
                        path="\\(1-\\)CRC-32",
                        recurring=False,
                        recursive=False,
                    ),
                ],
            )

            return result

        ebml_spec = get_ebml_spec()
        EBMLMainSpec = ebml_read_spec_xml("ebml/ebml.xml")

        assert EBMLMainSpec == ebml_spec
