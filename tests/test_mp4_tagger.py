from io import BufferedIOBase, BytesIO
from pathlib import Path
from typing import Self

from fixtures import TempVideoFiles, mark_as_used, mp4_test_parse_files
from pytest_subtests import SubTests

from content.tagger.mp4_tagger import (
    EDTS_ATOM_NAME,
    FREE_ATOM_NAME,
    FTYP_ATOM_NAME,
    HDLR_ATOM_NAME,
    MDHD_ATOM_NAME,
    MDIA_ATOM_NAME,
    MINF_ATOM_NAME,
    MOOV_ATOM_NAME,
    TRAK_ATOM_NAME,
    Mp4BoxSpan,
    ISOMAtomName,
    MP4Box,
    is_mp4_file,
    mp4_iter_boxes,
)
from helper.result import Result
from helper.translation import get_translator

mark_as_used(mp4_test_parse_files)

# TODO: force locale in test cases!
_ = get_translator()


class PseudoMp4Box(MP4Box):

    def __init__(self: Self, typ: ISOMAtomName, size: int) -> None:
        super().__init__(typ, span=Mp4BoxSpan(0, size, 8), is_container=False)


class RecursiveBoxes:
    __RecursiveBoxesData = list[MP4Box | tuple[MP4Box, "__RecursiveBoxesData"]]
    __data: __RecursiveBoxesData

    def __init__(self: Self, data: __RecursiveBoxesData) -> None:
        self.__data = data

    def append(self: Self, val: MP4Box | tuple[MP4Box, "RecursiveBoxes"]) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))
            return

        self.__data.append(val)

    def top_boxes(self: Self) -> list[MP4Box]:
        result: list[MP4Box] = []

        for box in self.__data:
            if isinstance(box, tuple):
                result.append(box[0])
                continue

            result.append(box)

        return result

    @staticmethod
    def __single_to_str(
        data: MP4Box | tuple[MP4Box, "__RecursiveBoxesData"],
        depth: int,
        indent_str: str = " ",
    ) -> str:
        if isinstance(data, tuple):
            return f"{(indent_str * depth)}<NestedBoxes\n{data[0]!s}\n{RecursiveBoxes.__to_str(data[1], depth=depth+1)}>"

        return f"{(indent_str * depth)}<SimpleBox {data!s}>"

    @staticmethod
    def __to_str(
        data: __RecursiveBoxesData,
        depth: int,
        indent_str: str = " ",
    ) -> str:

        return (f"\n{(indent_str * depth)}").join(
            RecursiveBoxes.__single_to_str(dat, depth, indent_str=indent_str)
            for dat in data
        )

    @staticmethod
    def __is_box_eq(
        box1: MP4Box,
        box2: MP4Box,
    ) -> bool:
        # pseudo comparison based on pseudo boxes, alias just size and type!
        if box1.type != box2.type:
            return False

        return box1.span.size == box2.span.size

    @staticmethod
    def __is_elem_eq(
        data1: MP4Box | tuple[MP4Box, __RecursiveBoxesData],
        data2: MP4Box | tuple[MP4Box, __RecursiveBoxesData],
        depth: int,
    ) -> bool:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            b1, d1 = data1
            b2, d2 = data2

            if not RecursiveBoxes.__is_box_eq(b1, b2):
                return False

            return RecursiveBoxes.__eq_impl_both(d1, d2, depth=depth + 1)
        if isinstance(data1, MP4Box) and isinstance(data2, MP4Box):
            return RecursiveBoxes.__is_box_eq(data1, data2)

        return False

    @staticmethod
    def __eq_impl_both(
        data1: __RecursiveBoxesData,
        data2: __RecursiveBoxesData,
        depth: int,
    ) -> bool:
        if len(data1) != len(data2):
            return False

        for d1, d2 in zip(data1, data2, strict=True):
            if not RecursiveBoxes.__is_elem_eq(d1, d2, depth):
                return False

        return True

    def __eq_impl(self: Self, data: __RecursiveBoxesData) -> bool:
        return RecursiveBoxes.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "  ")

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveBoxes):
            return self.__eq_impl(other.__data)

        return False


def list_all_boxes_recursively(f: BufferedIOBase) -> RecursiveBoxes:
    f.seek(0, 2)
    filesize = f.tell()

    result: RecursiveBoxes = RecursiveBoxes([])

    stack: list[tuple[int, int, RecursiveBoxes]] = [(0, filesize, result)]

    while stack:
        start, end, current_target = stack.pop()

        for box in mp4_iter_boxes(f, start, end):
            if box.is_container:
                target: tuple[MP4Box, RecursiveBoxes] = (box, RecursiveBoxes([]))
                current_target.append(target)
                stack.append((box.span.payload_start, box.span.end, target[1]))
            else:
                current_target.append(box)

    return result


Mp4BoxStructure__GetResult = Result["Mp4BoxStructure", str]


class Mp4BoxStructure:
    boxes: RecursiveBoxes

    def __init__(self: Self, boxes: RecursiveBoxes) -> None:
        self.boxes = boxes

    @staticmethod
    def from_file(file: Path) -> Mp4BoxStructure__GetResult:
        try:
            with file.open("rb") as f:
                mp4_res = is_mp4_file(f)

                if mp4_res is not None:
                    return Mp4BoxStructure__GetResult.err(mp4_res)

                boxes = list_all_boxes_recursively(f)
                return Mp4BoxStructure__GetResult.ok(Mp4BoxStructure(boxes))
        except RuntimeError as err:
            return Mp4BoxStructure__GetResult.err(str(err))

    def __str__(self: Self) -> str:
        return f"<Mp4BoxStructure boxes: {self.boxes!s}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveBoxes):
            return self.boxes == other

        if isinstance(other, Mp4BoxStructure):
            return self.boxes == other.boxes

        return False


def test_mp4_tagger_parsing(
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, Mp4BoxStructure]] = list(
        zip(
            mp4_test_parse_files.data,
            [
                Mp4BoxStructure(
                    RecursiveBoxes(
                        [
                            PseudoMp4Box(FTYP_ATOM_NAME, 32),
                            (
                                PseudoMp4Box(MOOV_ATOM_NAME, 11824),
                                [
                                    PseudoMp4Box(ISOMAtomName(b"mvhd"), 108),
                                    PseudoMp4Box(ISOMAtomName(b"iods"), 42),
                                    (
                                        PseudoMp4Box(TRAK_ATOM_NAME, 5317),
                                        [
                                            PseudoMp4Box(
                                                ISOMAtomName(
                                                    b"tkhd",
                                                ),
                                                92,
                                            ),
                                            PseudoMp4Box(EDTS_ATOM_NAME, 36),
                                            (
                                                PseudoMp4Box(MDIA_ATOM_NAME, 5181),
                                                [
                                                    PseudoMp4Box(MDHD_ATOM_NAME, 32),
                                                    PseudoMp4Box(HDLR_ATOM_NAME, 54),
                                                    PseudoMp4Box(MINF_ATOM_NAME, 5087),
                                                ],
                                            ),
                                        ],
                                    ),
                                    (
                                        PseudoMp4Box(TRAK_ATOM_NAME, 6349),
                                        [
                                            PseudoMp4Box(
                                                ISOMAtomName(
                                                    b"tkhd",
                                                ),
                                                92,
                                            ),
                                            PseudoMp4Box(EDTS_ATOM_NAME, 36),
                                            (
                                                PseudoMp4Box(MDIA_ATOM_NAME, 6213),
                                                [
                                                    PseudoMp4Box(MDHD_ATOM_NAME, 32),
                                                    PseudoMp4Box(HDLR_ATOM_NAME, 54),
                                                    PseudoMp4Box(MINF_ATOM_NAME, 6119),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            PseudoMp4Box(FREE_ATOM_NAME, 8),
                            PseudoMp4Box(ISOMAtomName(b"mdat"), 1558160),
                        ],
                    )
                ),
            ],
            strict=True,
        )
    )

    for file, result in test_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = Mp4BoxStructure.from_file(file)

            if structure_res.is_err():
                msg = f"structure not parsed correctly: {structure_res.get_err()}"
                raise AssertionError(msg)

            structure = structure_res.get_ok()

            top_boxes = structure.boxes.top_boxes()

            start: int = 0
            for top_box in top_boxes:
                if top_box.span.start != start:
                    msg = f"Next box start is invalid, expected {start} but got {top_box.span.start}: {top_box!s}"
                    raise AssertionError(msg)
                start = top_box.span.end

            file_size = file.stat().st_size

            if file_size != start:
                msg = f"boxes don't reach EOF: size is {file_size} but boxes reach only to {start}"
                raise AssertionError(msg)

            if structure != result:
                msg = f"Parsing was incorrect:\n{structure!s}"
                raise AssertionError(msg)


def test_mp4_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Read failed to produce 8 bytes, got 0"),
        (b"hello world", _("Not a valid ISOM / MP4 file")),
        (b"ftyp    ", "Atom name is not lowercase b'    '"),
        (b"\x00\x00\x00\x04ftyp", "Invalid box: sitze too small: 4"),
        (
            b"\x00\x00\x00\x0eftypabcddcba",
            "Invalid box size: not enough data for complete FileTypeBox: have 6 but need at least 8",
        ),
        (
            b"\x00\x00\x00\x10ftypabcddcba",
            "ISOM/MP42 file has valid box, but invalid major_brand: b'abcd'",
        ),
    ]

    for data, err in test_data:
        with subtests.test("invalid video gets detected correctly"):
            io = BytesIO(data)
            res = is_mp4_file(io)

            assert res is not None, "valid mp4 is incorrect here"

            assert res == err, "incorrect error"
