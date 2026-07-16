from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Self, assert_never, override

from conftest import FancyEq
from fixtures import TempVideoFiles, mark_as_used, mkv_test_parse_files
from pytest_subtests import SubTests
from test_helper import ErrResult, OkResult, TestResult

from content.tagger.mkv_tagger import (
    BitIterator,
    EBMLDecodeOptions,
    EBMLElement,
    EBMLElementID,
    EBMLElementParsed,
    EBMLElementSpan,
    EBMLStream,
    EBMLVarInt,
    ebml_iter_elements,
    is_mkv_file,
)
from content.tagger.parser import BoundedIO, SimpleSpan
from content.tagger.schema.parser import (
    DefaultEmpty,
    DocType,
    EBMLAdvancedElementType,
    EBMLAdvancedElementTypeBinary,
    EBMLAdvancedElementTypeDate,
    EBMLAdvancedElementTypeFloat,
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
from helper.utils import hash_list

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
    tests: list[tuple[bytes, TestResult[None, str]]] = [
        # spec tests
        (b"\x80", ErrResult("Value is NULL")),
        (b"\x40\x00", ErrResult("Value is NULL")),
        (b"\x81", OkResult(None)),
        (
            b"\x40\x01",
            ErrResult(
                "Value 1 uses too much bytes: 1 bytes are the minimum, but used 2",
            ),
        ),
        (b"\xaf", OkResult(None)),
        (
            b"\x40\x3f",
            ErrResult(
                "Value 63 uses too much bytes: 1 bytes are the minimum, but used 2",
            ),
        ),
        (b"\xff", ErrResult("Value is 0xFF..FF")),
        (b"\x40\x7f", OkResult(None)),
        # custom tests
        (
            b"\x20\x00\x7f",
            ErrResult(
                "Value 127 uses too much bytes: 1 bytes are the minimum, but used 3",
            ),
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

            valid_element_id = element_id.is_valid(bytes_used)

            assert result == valid_element_id


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


class PseudoMKVElement(EBMLElementParsed):

    def __init__(
        self: Self,
        element_id: int,
        element_type: EBMLElementType,
        name: str,
        size: int,
    ) -> None:
        element_id_correct = EBMLElementID(EBMLVarInt(element_id), element_id)

        header_sizes: tuple[int, int] = (
            element_id_correct.minmum_bytes_required(),
            EBMLVarInt(size).minmum_bytes_required(),
        )
        header_size = sum(header_sizes)
        element: EBMLElement = EBMLElement(
            element_id=element_id_correct,
            span=EBMLElementSpan.from_ebml_specified_size(
                SimpleSpan(0, size - header_size),
                header_sizes,
            ),
        )

        advanced_element_type: EBMLAdvancedElementType

        match element_type:
            case EBMLElementType.SignedInteger:
                advanced_element_type = EBMLAdvancedElementTypeInteger(
                    type=element_type,
                    default=DefaultEmpty(),
                    range=None,
                )
            case EBMLElementType.UnsignedInteger:
                advanced_element_type = EBMLAdvancedElementTypeInteger(
                    type=element_type,
                    default=DefaultEmpty(),
                    range=None,
                )
            case EBMLElementType.Float:
                advanced_element_type = EBMLAdvancedElementTypeFloat(
                    type=element_type,
                    default=DefaultEmpty(),
                    range=None,
                )
            case EBMLElementType.String:
                advanced_element_type = EBMLAdvancedElementTypeString(
                    type=element_type,
                    default=DefaultEmpty(),
                    length=None,
                )
            case EBMLElementType.UTF8:
                advanced_element_type = EBMLAdvancedElementTypeString(
                    type=element_type,
                    default=DefaultEmpty(),
                    length=None,
                )
            case EBMLElementType.Date:
                advanced_element_type = EBMLAdvancedElementTypeDate(
                    type=element_type,
                    default=DefaultEmpty(),
                )
            case EBMLElementType.Master:
                advanced_element_type = EBMLAdvancedElementTypeMaster(
                    type=element_type,
                )
            case EBMLElementType.Binary:
                advanced_element_type = EBMLAdvancedElementTypeBinary(
                    type=element_type,
                    default=DefaultEmpty(),
                    length=None,
                )
            case _:
                assert_never(element_type)

        element_desc: EBMLElementDescription = EBMLElementDescriptionGeneric(
            name=name,
            id=EBMLElementIDParsed(element_id),
            occurrences=EBMLOccurrences(
                EBMLSchemaRange(
                    (
                        EBMLSchemaRangeElem(0, bound=EBMLSchemaRangeBound.Inclusive),
                        None,
                    ),
                ),
            ),
            type=advanced_element_type,
            description="<Nothing>",
            unknown_size_allowed=False,
            versions=EBMLSchemaRange(42),
            path="/<IGNORE>",
            recurring=False,
            recursive=False,
        )

        super().__init__(element, element_desc)


class PseudoClusterMKVElement(PseudoMKVElement):
    children: int

    def __init__(self: Self, children: int, size: int) -> None:
        super().__init__(0x1F43B675, EBMLElementType.Master, "Cluster", size)
        self.children = children


@decorate_class(slots=True)
class RecursiveElements:
    RecursiveElementsData = list[
        EBMLElementParsed | tuple[EBMLElementParsed, "RecursiveElementsData"]
    ]
    __data: RecursiveElementsData

    def __init__(self: Self, data: RecursiveElementsData) -> None:
        self.__data = data

    def append(
        self: Self,
        val: EBMLElementParsed | tuple[EBMLElementParsed, "RecursiveElements"],
    ) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))  # noqa: SLF001
            return

        self.__data.append(val)

    @property
    def data(self: Self) -> RecursiveElementsData:
        return self.__data

    @staticmethod
    def __single_to_str(
        data: EBMLElementParsed | tuple[EBMLElementParsed, "RecursiveElementsData"],
        depth: int,
        indent_str: str = " ",
    ) -> str:
        if isinstance(data, tuple):
            return f"{(indent_str * depth)}<NestedElements\n{data[0]!s}\n{RecursiveElements.__to_str(data[1], depth=depth+1)}>"

        return f"{(indent_str * depth)}<SimpleElement {data!s}>"

    @staticmethod
    def __to_str(
        data: RecursiveElementsData,
        depth: int,
        indent_str: str = " ",
    ) -> str:

        return (f"\n{(indent_str * depth)}").join(
            RecursiveElements.__single_to_str(dat, depth, indent_str=indent_str)
            for dat in data
        )

    @staticmethod
    def __is_element_eq(
        element1: EBMLElementParsed,
        element2: EBMLElementParsed,
        depth: int,
    ) -> Result[None, list[str]]:
        # pseudo comparison based on pseudo elements, alias just size and type!
        if element1.element.element_id != element2.element.element_id:
            return Err[list[str]](
                [
                    "Element ID of data is not eq:",
                    str(element1.element.element_id),
                    str(element2.element.element_id),
                    f"Depth {depth}",
                    str(element1),
                    str(element2),
                ],
            )

        if element1.element.span.total.size != element2.element.span.total.size:
            return Err[list[str]](
                [
                    "Sizeof data is not eq:",
                    str(element1.element.span.total.size),
                    str(element2.element.span.total.size),
                    f"Depth {depth}",
                    str(element1),
                    str(element2),
                ],
            )

        if element1.desc.name != element2.desc.name:
            return Err(
                [
                    "Element name doesn't match:",
                    str(element1.desc.name),
                    str(element2.desc.name),
                ],
            )

        if element1.desc.type.type != element2.desc.type.type:
            return Err(
                [
                    "Element type doesn't match:",
                    f"Element name: {element1.desc.name}",
                    str(element1.desc.type.type),
                    str(element2.desc.type.type),
                ],
            )

        # TODO: compare value

        return Ok(None)

    @staticmethod
    def __is_elem_eq(
        data1: EBMLElementParsed | tuple[EBMLElementParsed, RecursiveElementsData],
        data2: EBMLElementParsed | tuple[EBMLElementParsed, RecursiveElementsData],
        depth: int,
    ) -> Result[None, list[str]]:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            b1, d1 = data1
            b2, d2 = data2

            res = RecursiveElements.__is_element_eq(b1, b2, depth)
            if res.err():
                return res

            if b1.desc.name == "Cluster":
                if not isinstance(b2, PseudoClusterMKVElement):
                    return Err[list[str]](
                        [
                            "Cluster Master Element on left side of eq, but right side is not correct class:",
                            str(type(b1)),
                            str(type(b2)),
                            f"Depth {depth}",
                            str(b1),
                            str(b2),
                        ],
                    )

                if b2.children != len(d1):
                    return Err[list[str]](
                        [
                            "Sizeof Cluster Master Element children is not eq:",
                            str(b2.children),
                            str(len(d1)),
                            f"Depth {depth}",
                            str(b2),
                            str(d1),
                        ],
                    )

                return Ok(None)

            return RecursiveElements.__eq_impl_both(d1, d2, depth=depth + 1)
        if isinstance(data1, EBMLElementParsed) and isinstance(
            data2,
            EBMLElementParsed,
        ):
            return RecursiveElements.__is_element_eq(data1, data2, depth)

        return Err[list[str]](
            [
                "Type of data is not eq:",
                str(type(data1)),
                str(type(data2)),
                f"Depth {depth}",
                str(data1),
                str(data2),
            ],
        )

    @staticmethod
    def __eq_impl_both(
        data1: RecursiveElementsData,
        data2: RecursiveElementsData,
        depth: int,
    ) -> Result[None, list[str]]:
        if len(data1) != len(data2):
            return Err[list[str]](
                [
                    "Length of data is not eq:",
                    str(len(data1)),
                    str(len(data2)),
                    f"Depth {depth}",
                    str(data1),
                    str(data2),
                ],
            )

        for d1, d2 in zip(data1, data2, strict=True):
            res = RecursiveElements.__is_elem_eq(d1, d2, depth)
            if res.err():
                return res

        return Ok(None)

    def __eq_impl(self: Self, data: RecursiveElementsData) -> Result[None, list[str]]:
        return RecursiveElements.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveElements.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveElements.__to_str(self.__data, 0, "  ")

    def eq_impl(self: Self, other: "RecursiveElements") -> Result[None, list[str]]:
        return self.__eq_impl(other.data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveElements):
            return self.__eq_impl(other.__data).ok()

        return False

    def __hash__(self: Self) -> int:
        return hash(("RecursiveElements", hash_list(self.__data)))


def list_all_elements_recursively(
    f: BinaryIO,
    span: SimpleSpan,
    options: EBMLDecodeOptions,
    spec: EBMLSpec,
) -> RecursiveElements:

    result: RecursiveElements = RecursiveElements([])

    stack: list[tuple[SimpleSpan, RecursiveElements, int]] = [
        (span, result, 0),
    ]

    while stack:
        span, current_target, depth = stack.pop()
        io = BoundedIO.get_new(f, span)

        for element in ebml_iter_elements(
            io,
            options,
            spec,
            depth=depth + 1,
        ):
            if element.desc.type.type == EBMLElementType.Master:
                target: tuple[EBMLElementParsed, RecursiveElements] = (
                    element,
                    RecursiveElements([]),
                )
                current_target.append(target)
                stack.append((element.element.span.payload_span, target[1], depth + 1))
            else:
                current_target.append(element)

    return result


@decorate_class(slots=True)
class MKVElementStructure(FancyEq):
    elements: RecursiveElements

    def __init__(self: Self, elements: RecursiveElements) -> None:
        self.elements = elements

    @staticmethod
    def from_file(
        file: Path,
    ) -> Result["MKVElementStructure", str]:
        try:
            with file.open("rb") as f:
                mkv_res = is_mkv_file(f)

                if mkv_res.err():
                    return Err(mkv_res.as_err())

                stream = EBMLStream.read_from_file(f)

                if len(stream.documents) != 1:
                    return Err(
                        f"Only One MKV EBML document supported atm, but got: {len(stream.documents)}",
                    )

                document = stream.documents[0]

                header_elem: tuple[
                    EBMLElementParsed,
                    RecursiveElements.RecursiveElementsData,
                ] = (EBMLElementParsed(document.header, document.header.desc), [])

                header_options = document.header.options
                spec = document.header.spec_unsafe()

                body_elements = list_all_elements_recursively(
                    f,
                    document.body.span.payload_span,
                    header_options.options,
                    spec,
                )

                body_element = (
                    EBMLElementParsed(document.body, document.body.desc),
                    body_elements.data,
                )

                elements = RecursiveElements([header_elem, body_element])

                return Ok(MKVElementStructure(elements))
        except RuntimeError as err:
            return Err(str(err))

    def __str__(self: Self) -> str:
        return f"<MKVElementStructure elements: {self.elements!s}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, RecursiveElements):
            return (True, lambda: self.elements.eq_impl(other))

        if isinstance(other, MKVElementStructure):
            return (True, lambda: self.elements.eq_impl(other.elements))

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

    def __hash__(self: Self) -> int:
        return hash(("MKVElementStructure", self.elements))


def test_mkv_tagger_parsing(
    subtests: SubTests,
    mkv_test_parse_files: TempVideoFiles,
) -> None:

    structure1 = MKVElementStructure(
        RecursiveElements(
            [
                (
                    PseudoMKVElement(0x1A45DFA3, EBMLElementType.Master, "EBML", 47),
                    [],
                ),
                (
                    PseudoMKVElement(
                        0x18538067,
                        EBMLElementType.Master,
                        "Segment",
                        573019,
                    ),
                    [
                        (
                            PseudoMKVElement(
                                0x114D9B74,
                                EBMLElementType.Master,
                                "SeekHead",
                                72,
                            ),
                            [
                                PseudoMKVElement(
                                    0xBF,
                                    EBMLElementType.Binary,
                                    "CRC-32",
                                    6,
                                ),
                                (
                                    PseudoMKVElement(
                                        0x4DBB,
                                        EBMLElementType.Master,
                                        "Seek",
                                        14,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0x53AB,
                                            EBMLElementType.Binary,
                                            "SeekID",
                                            7,
                                        ),
                                        PseudoMKVElement(
                                            0x53AC,
                                            EBMLElementType.UnsignedInteger,
                                            "SeekPosition",
                                            4,
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0x4DBB,
                                        EBMLElementType.Master,
                                        "Seek",
                                        15,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0x53AB,
                                            EBMLElementType.Binary,
                                            "SeekID",
                                            7,
                                        ),
                                        PseudoMKVElement(
                                            0x53AC,
                                            EBMLElementType.UnsignedInteger,
                                            "SeekPosition",
                                            5,
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0x4DBB,
                                        EBMLElementType.Master,
                                        "Seek",
                                        15,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0x53AB,
                                            EBMLElementType.Binary,
                                            "SeekID",
                                            7,
                                        ),
                                        PseudoMKVElement(
                                            0x53AC,
                                            EBMLElementType.UnsignedInteger,
                                            "SeekPosition",
                                            5,
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0x4DBB,
                                        EBMLElementType.Master,
                                        "Seek",
                                        16,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0x53AB,
                                            EBMLElementType.Binary,
                                            "SeekID",
                                            7,
                                        ),
                                        PseudoMKVElement(
                                            0x53AC,
                                            EBMLElementType.UnsignedInteger,
                                            "SeekPosition",
                                            6,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        PseudoMKVElement(
                            0xEC,
                            EBMLElementType.Binary,
                            "Void",
                            157,
                        ),
                        (
                            PseudoMKVElement(
                                0x1549A966,
                                EBMLElementType.Master,
                                "Info",
                                87,
                            ),
                            [
                                PseudoMKVElement(
                                    0xBF,
                                    EBMLElementType.Binary,
                                    "CRC-32",
                                    6,
                                ),
                                PseudoMKVElement(
                                    0x2AD7B1,
                                    EBMLElementType.UnsignedInteger,
                                    "TimestampScale",
                                    7,
                                ),
                                PseudoMKVElement(
                                    0x4D80,
                                    EBMLElementType.UTF8,
                                    "MuxingApp",
                                    16,
                                ),
                                PseudoMKVElement(
                                    0x5741,
                                    EBMLElementType.UTF8,
                                    "WritingApp",
                                    16,
                                ),
                                PseudoMKVElement(
                                    0x73A4,
                                    EBMLElementType.Binary,
                                    "SegmentUUID",
                                    19,
                                ),
                                PseudoMKVElement(
                                    0x4489,
                                    EBMLElementType.Float,
                                    "Duration",
                                    11,
                                ),
                            ],
                        ),
                        (
                            PseudoMKVElement(
                                0x1654AE6B,
                                EBMLElementType.Master,
                                "Tracks",
                                151,
                            ),
                            [
                                PseudoMKVElement(
                                    0xBF,
                                    EBMLElementType.Binary,
                                    "CRC-32",
                                    6,
                                ),
                                (
                                    PseudoMKVElement(
                                        0xAE,
                                        EBMLElementType.Master,
                                        "TrackEntry",
                                        133,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0xD7,
                                            EBMLElementType.UnsignedInteger,
                                            "TrackNumber",
                                            3,
                                        ),
                                        PseudoMKVElement(
                                            0x73C5,
                                            EBMLElementType.UnsignedInteger,
                                            "TrackUID",
                                            4,
                                        ),
                                        PseudoMKVElement(
                                            0x9C,
                                            EBMLElementType.UnsignedInteger,
                                            "FlagLacing",
                                            3,
                                        ),
                                        PseudoMKVElement(
                                            0x22B59C,
                                            EBMLElementType.String,
                                            "Language",
                                            7,
                                        ),
                                        PseudoMKVElement(
                                            0x86,
                                            EBMLElementType.String,
                                            "CodecID",
                                            17,
                                        ),
                                        PseudoMKVElement(
                                            0x83,
                                            EBMLElementType.UnsignedInteger,
                                            "TrackType",
                                            3,
                                        ),
                                        PseudoMKVElement(
                                            0x23E383,
                                            EBMLElementType.UnsignedInteger,
                                            "DefaultDuration",
                                            8,
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0xE0,
                                                EBMLElementType.Master,
                                                "Video",
                                                35,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0xB0,
                                                    EBMLElementType.UnsignedInteger,
                                                    "PixelWidth",
                                                    4,
                                                ),
                                                PseudoMKVElement(
                                                    0xBA,
                                                    EBMLElementType.UnsignedInteger,
                                                    "PixelHeight",
                                                    4,
                                                ),
                                                PseudoMKVElement(
                                                    0x9A,
                                                    EBMLElementType.UnsignedInteger,
                                                    "FlagInterlaced",
                                                    3,
                                                ),
                                                PseudoMKVElement(
                                                    0x54B2,
                                                    EBMLElementType.UnsignedInteger,
                                                    "DisplayUnit",
                                                    4,
                                                ),
                                                (
                                                    PseudoMKVElement(
                                                        0x55B0,
                                                        EBMLElementType.Master,
                                                        "Colour",
                                                        11,
                                                    ),
                                                    [
                                                        PseudoMKVElement(
                                                            0x55B7,
                                                            EBMLElementType.UnsignedInteger,
                                                            "ChromaSitingHorz",
                                                            4,
                                                        ),
                                                        PseudoMKVElement(
                                                            0x55B8,
                                                            EBMLElementType.UnsignedInteger,
                                                            "ChromaSitingVert",
                                                            4,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        PseudoMKVElement(
                                            0x63A2,
                                            EBMLElementType.Binary,
                                            "CodecPrivate",
                                            44,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        (
                            PseudoMKVElement(
                                0x1254C367,
                                EBMLElementType.Master,
                                "Tags",
                                370,
                            ),
                            [
                                PseudoMKVElement(
                                    0xBF,
                                    EBMLElementType.Binary,
                                    "CRC-32",
                                    6,
                                ),
                                (
                                    PseudoMKVElement(
                                        0x7373,
                                        EBMLElementType.Master,
                                        "Tag",
                                        166,
                                    ),
                                    [
                                        (
                                            PseudoMKVElement(
                                                0x63C0,
                                                EBMLElementType.Master,
                                                "Targets",
                                                10,
                                            ),
                                            [],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                31,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    14,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    7,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                30,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    16,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    4,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                49,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    20,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    19,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                36,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    10,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    16,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0x7373,
                                        EBMLElementType.Master,
                                        "Tag",
                                        118,
                                    ),
                                    [
                                        (
                                            PseudoMKVElement(
                                                0x63C0,
                                                EBMLElementType.Master,
                                                "Targets",
                                                14,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x63C5,
                                                    EBMLElementType.UnsignedInteger,
                                                    "TagTrackUID",
                                                    4,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                49,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    15,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    24,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                45,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    10,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    25,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0x7373,
                                        EBMLElementType.Master,
                                        "Tag",
                                        68,
                                    ),
                                    [
                                        (
                                            PseudoMKVElement(
                                                0x63C0,
                                                EBMLElementType.Master,
                                                "Targets",
                                                14,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x63C5,
                                                    EBMLElementType.UnsignedInteger,
                                                    "TagTrackUID",
                                                    4,
                                                ),
                                            ],
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0x67C8,
                                                EBMLElementType.Master,
                                                "SimpleTag",
                                                44,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0x45A3,
                                                    EBMLElementType.UTF8,
                                                    "TagName",
                                                    11,
                                                ),
                                                PseudoMKVElement(
                                                    0x4487,
                                                    EBMLElementType.UTF8,
                                                    "TagString",
                                                    23,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        (
                            PseudoClusterMKVElement(
                                149,
                                213680,
                            ),
                            [],
                        ),
                        (
                            PseudoClusterMKVElement(
                                105,
                                142373,
                            ),
                            [],
                        ),
                        (
                            PseudoClusterMKVElement(
                                152,
                                216063,
                            ),
                            [],
                        ),
                        (
                            PseudoMKVElement(
                                0x1C53BB6B,
                                EBMLElementType.Master,
                                "Cues",
                                54,
                            ),
                            [
                                PseudoMKVElement(
                                    0xBF,
                                    EBMLElementType.Binary,
                                    "CRC-32",
                                    6,
                                ),
                                (
                                    PseudoMKVElement(
                                        0xBB,
                                        EBMLElementType.Master,
                                        "CuePoint",
                                        17,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0xB3,
                                            EBMLElementType.UnsignedInteger,
                                            "CueTime",
                                            3,
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0xB7,
                                                EBMLElementType.Master,
                                                "CueTrackPositions",
                                                12,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0xF7,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueTrack",
                                                    3,
                                                ),
                                                PseudoMKVElement(
                                                    0xF1,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueClusterPosition",
                                                    4,
                                                ),
                                                PseudoMKVElement(
                                                    0xF0,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueRelativePosition",
                                                    3,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                (
                                    PseudoMKVElement(
                                        0xBB,
                                        EBMLElementType.Master,
                                        "CuePoint",
                                        19,
                                    ),
                                    [
                                        PseudoMKVElement(
                                            0xB3,
                                            EBMLElementType.UnsignedInteger,
                                            "CueTime",
                                            4,
                                        ),
                                        (
                                            PseudoMKVElement(
                                                0xB7,
                                                EBMLElementType.Master,
                                                "CueTrackPositions",
                                                13,
                                            ),
                                            [
                                                PseudoMKVElement(
                                                    0xF7,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueTrack",
                                                    3,
                                                ),
                                                PseudoMKVElement(
                                                    0xF1,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueClusterPosition",
                                                    5,
                                                ),
                                                PseudoMKVElement(
                                                    0xF0,
                                                    EBMLElementType.UnsignedInteger,
                                                    "CueRelativePosition",
                                                    3,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )

    test_files: list[tuple[Path, str, MKVElementStructure]] = list(
        zip(
            [f for f, _ in mkv_test_parse_files.data],
            [nm for _, nm in mkv_test_parse_files.data],
            [structure1],
            strict=True,
        ),
    )

    for file, name, result in test_files:
        with subtests.test(f"video gets parsed correctly: {name}"):
            structure_res = MKVElementStructure.from_file(file)

            assert structure_res == OkResult(), "structure not parsed correctly"

            structure = structure_res.as_ok()

            filesize = file.stat().st_size

            # check elements consistency
            elements_stack: list[
                tuple[SimpleSpan, RecursiveElements.RecursiveElementsData]
            ] = [
                (SimpleSpan(0, filesize), structure.elements.data),
            ]

            while len(elements_stack) != 0:

                elements_span, elements = elements_stack.pop()
                start: int = elements_span.start

                if len(elements) == 0:
                    start = elements_span.end

                for element_data in elements:

                    element: EBMLElementParsed
                    if isinstance(element_data, tuple):
                        assert (
                            element_data[0].desc.type.type == EBMLElementType.Master
                        ), "elements resulting in children have to be a of type master"
                        element = element_data[0]
                        elements_stack.append(
                            (element.element.span.payload_span, element_data[1]),
                        )
                    else:
                        element = element_data

                    assert (
                        element.element.span.total.start == start
                    ), f"Next element start is invalid: {element!s}"

                    start = element.element.span.total.end

                assert (
                    elements_span.end == start
                ), "elements don't reach at the parent end"

            assert structure == result, "Parsing was incorrect"


def test_mkv_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Element ID Parse error: Not enough data for VarInt"),
        (
            b"\x00",
            "Element ID Parse error: The first byte of a VarInt can't be 0x00",
        ),
        (
            b"\x01\x12",
            "Element ID Parse error: Not enough data for VarInt: need 8 bytes but got 2",
        ),
        (
            b"\x40\x02",
            "Invalid Element ID: <EBMLElementID 0x4002>: Value 2 uses too much bytes: 1 bytes are the minimum, but used 2",
        ),
        (
            b"\x82",
            "Data Size Parse error: Not enough data for VarInt",
        ),
        (
            b"\x82\x81\x00",
            "Invalid EBML Header element ID: got <EBMLElementID 0x82> but expected <EBMLElementIDParsed 0x1a45dfa3>",
        ),
        (
            (
                b"\x1a\x45\xdf\xa3"  # EBML header ID
                b"\x80"  # size of master container: 0
            ),
            "Invalid EBML Header: missing DocType in header",
        ),
        (
            (
                b"\x1a\x45\xdf\xa3"  # EBML header ID  # noqa: ISC003
                b"\x8b"  # size of master container: 11
                ## DocType sub-element
                + (
                    b"\x42\x82"  # DocType ID
                    b"\x88"  # size of doctype element: 8
                    b"matroskb"
                )
            ),
            "DocType <DocType type: matroskb version: 1> is not supported: No such DocType",
        ),
        (
            (
                b"\x1a\x45\xdf\xa3"  # EBML header ID  # noqa: ISC003
                b"\x8c"  # size of master container: 12 (wrong size)
                ## DocType sub-element
                + (
                    b"\x42\x82"  # DocType ID
                    b"\x88"  # size of doctype element: 8
                    b"matroskb"
                )
            ),
            "New payload io end overflows parent: 17 > 16",
        ),
    ]

    for data, err in test_data:
        with subtests.test("invalid video gets detected correctly"):
            io = BytesIO(data)
            res = is_mkv_file(io)

            assert ErrResult(err) == res, "incorrect error"


def test_mkv_valid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[bytes] = [
        (
            b"\x1a\x45\xdf\xa3"  # EBML header ID  # noqa: ISC003
            b"\x8b"  # size of master container: 11
            ## DocType sub-element
            + (
                b"\x42\x82"  # DocType ID
                b"\x88"  # size of doctype element: 8
                b"matroska"
            )
        ),
    ]

    for data in test_data:
        with subtests.test("valid video gets detected correctly"):
            io = BytesIO(data)
            res = is_mkv_file(io)

            assert OkResult(None) == res


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

            return EBMLTestSpec(doc_type=DocType(type="matroska", version=4))

        EBMLMKVSpec = ebml_read_spec_xml("mkv/ebml_matroska.xml")  # noqa: N806

        mkv_spec = get_mkv_spec()

        # TODO: this needs so much boilerplate, but implement a full comparison
        # assert EBMLMKVSpec == mkv_spec.spec  # noqa: ERA001

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
        EBMLMainSpec: EBMLSpec = ebml_read_spec_xml("ebml/ebml.xml")  # noqa: N806

        assert EBMLMainSpec == ebml_spec
