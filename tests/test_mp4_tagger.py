import json
from collections.abc import Callable
from copy import deepcopy
from io import BufferedIOBase, BytesIO
from pathlib import Path
from typing import Any, Optional, Self, override
from unittest import mock
from uuid import uuid4

from conftest import FancyEq
from fixtures import TempVideoFiles, mark_as_used, mp4_test_parse_files, test_manager
from pytest_subtests import SubTests
from test_helper import OkResult, file_duplicates

from content.language import Language
from content.tagger.mp4_tagger import (
    EDTS_ATOM_NAME,
    FREE_ATOM_NAME,
    FTYP_ATOM_NAME,
    HDLR_ATOM_NAME,
    ILST_ATOM_NAME,
    IODS_ATOM_NAME,
    MDAT_ATOM_NAME,
    MDHD_ATOM_NAME,
    MDIA_ATOM_NAME,
    META_ATOM_NAME,
    MINF_ATOM_NAME,
    MOOV_ATOM_NAME,
    MVHD_ATOM_NAME,
    SOUN_ATOM_NAME,
    TKHD_ATOM_NAME,
    TRAK_ATOM_NAME,
    UDTA_ATOM_NAME,
    VIDE_ATOM_NAME,
    AppleItunesItemBox,
    AppleItunesItemBoxAtomFreeform,
    AppleItunesItemDataContent,
    AppleItunesItemDataType,
    AppleItunesItemFreeformBox,
    ISOMAtomName,
    MP4Box,
    MP4BoxSpan,
    VideoTaggerMP4,
    find_mdhd_boxes_with_type,
    is_mp4_file,
    merge_dicts,
    mp4_iter_boxes,
)
from content.tagger.mutagen_tagger import VideoTaggerMutagen
from content.tagger.parser import SimpleSpan
from content.tagger.video_tagger import MetadataTags, uuid_to_str
from helper.ffprobe import FFProbeResult, ffprobe
from helper.manager import ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

mark_as_used(mp4_test_parse_files)
mark_as_used(test_manager)

# TODO: force locale in test cases!
_ = get_translator()


class PseudoMP4Box(MP4Box):

    def __init__(self: Self, typ: ISOMAtomName, size: int) -> None:
        super().__init__(
            typ,
            span=MP4BoxSpan(SimpleSpan(0, size), 8),
            is_container=False,
        )


class PseudoAppleItunesMP4Box(MP4Box):
    type_indicator: AppleItunesItemDataType
    value: AppleItunesItemDataContent

    def __init__(
        self: Self,
        typ: ISOMAtomName,
        size: int,
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> None:
        super().__init__(
            typ,
            span=MP4BoxSpan(SimpleSpan(0, size), 8),
            is_container=False,
        )

        self.type_indicator = type_indicator
        self.value = value


class PseudoAppleItunesMP4FreeformBox(PseudoAppleItunesMP4Box):
    mean: str
    name: str

    def __init__(
        self: Self,
        size: int,
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
        mean: str,
        name: str,
    ) -> None:
        super().__init__(AppleItunesItemBoxAtomFreeform, size, type_indicator, value)

        self.mean = mean
        self.name = name


class RecursiveBoxes:
    RecursiveBoxesData = list[MP4Box | tuple[MP4Box, "RecursiveBoxesData"]]
    __data: RecursiveBoxesData

    def __init__(self: Self, data: RecursiveBoxesData) -> None:
        self.__data = data

    def append(self: Self, val: MP4Box | tuple[MP4Box, "RecursiveBoxes"]) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))  # noqa: SLF001
            return

        self.__data.append(val)

    @property
    def data(self: Self) -> RecursiveBoxesData:
        return self.__data

    @staticmethod
    def __single_to_str(
        data: MP4Box | tuple[MP4Box, "RecursiveBoxesData"],
        depth: int,
        indent_str: str = " ",
    ) -> str:
        if isinstance(data, tuple):
            return f"{(indent_str * depth)}<NestedBoxes\n{data[0]!s}\n{RecursiveBoxes.__to_str(data[1], depth=depth+1)}>"

        return f"{(indent_str * depth)}<SimpleBox {data!s}>"

    @staticmethod
    def __to_str(
        data: RecursiveBoxesData,
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
        depth: int,
    ) -> Result[None, list[str]]:
        # pseudo comparison based on pseudo boxes, alias just size and type!
        if box1.type != box2.type:
            return Err[list[str]](
                [
                    "Atom type of data is not eq:",
                    str(box1.type),
                    str(box2.type),
                    f"Depth {depth}",
                    str(box1),
                    str(box2),
                ],
            )

        if box1.span.total.size != box2.span.total.size:
            return Err[list[str]](
                [
                    "Sizeof data is not eq:",
                    str(box1.span.total.size),
                    str(box2.span.total.size),
                    f"Depth {depth}",
                    str(box1),
                    str(box2),
                ],
            )

        if isinstance(
            box1,
            (AppleItunesItemBox, AppleItunesItemFreeformBox),
        ):
            if not isinstance(
                box2,
                PseudoAppleItunesMP4Box,
            ):
                return Err[list[str]](
                    [
                        "AppleItunesItemBox on left side of eq, but right side is not correct class:",
                        str(type(box1)),
                        str(type(box2)),
                        f"Depth {depth}",
                        str(box1),
                        str(box2),
                    ],
                )

            if box1.data.type_indicator != box2.type_indicator.value:
                return Err[list[str]](
                    [
                        "AppleItunesItemBox type_indicator is not eq:",
                        str(box1.data.type_indicator),
                        str(box2.type_indicator.value),
                        f"Depth {depth}",
                        str(box1),
                        str(box2),
                    ],
                )

            if box1.data.value != box2.value:
                return Err[list[str]](
                    [
                        "AppleItunesItemBox value is not eq:",
                        str(box1.data.value),
                        str(box2.value),
                        f"Depth {depth}",
                        str(box1),
                        str(box2),
                    ],
                )

            if isinstance(box1, AppleItunesItemFreeformBox):

                if not isinstance(
                    box2,
                    PseudoAppleItunesMP4FreeformBox,
                ):
                    return Err[list[str]](
                        [
                            "AppleItunesItemFreeformBox on left side of eq, but right side is not correct class:",
                            str(type(box1)),
                            str(type(box2)),
                            f"Depth {depth}",
                            str(box1),
                            str(box2),
                        ],
                    )

                if box1.mean.value != box2.mean:
                    return Err[list[str]](
                        [
                            "AppleItunesItemFreeformBox mean is not eq:",
                            str(box1.mean.value),
                            str(box2.mean),
                            f"Depth {depth}",
                            str(box1),
                            str(box2),
                        ],
                    )

                if box1.name.value != box2.name:
                    return Err[list[str]](
                        [
                            "AppleItunesItemFreeformBox name is not eq:",
                            str(box1.name.value),
                            str(box2.name),
                            f"Depth {depth}",
                            str(box1),
                            str(box2),
                        ],
                    )

        return Ok(None)

    @staticmethod
    def __is_elem_eq(
        data1: MP4Box | tuple[MP4Box, RecursiveBoxesData],
        data2: MP4Box | tuple[MP4Box, RecursiveBoxesData],
        depth: int,
    ) -> Result[None, list[str]]:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            b1, d1 = data1
            b2, d2 = data2

            res = RecursiveBoxes.__is_box_eq(b1, b2, depth)
            if res.err():
                return res

            return RecursiveBoxes.__eq_impl_both(d1, d2, depth=depth + 1)
        if isinstance(data1, MP4Box) and isinstance(data2, MP4Box):
            return RecursiveBoxes.__is_box_eq(data1, data2, depth)

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
        data1: RecursiveBoxesData,
        data2: RecursiveBoxesData,
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
            res = RecursiveBoxes.__is_elem_eq(d1, d2, depth)
            if res.err():
                return res

        return Ok(None)

    def __eq_impl(self: Self, data: RecursiveBoxesData) -> Result[None, list[str]]:
        return RecursiveBoxes.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "  ")

    def eq_impl(self: Self, other: "RecursiveBoxes") -> Result[None, list[str]]:
        return self.__eq_impl(other.data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveBoxes):
            return self.__eq_impl(other.__data).ok()

        return False

    def __hash__(self: Self) -> int:
        return hash(*self.__data)


def list_all_boxes_recursively(f: BufferedIOBase) -> RecursiveBoxes:
    f.seek(0, 2)
    filesize = f.tell()

    result: RecursiveBoxes = RecursiveBoxes([])

    stack: list[tuple[SimpleSpan, RecursiveBoxes]] = [(SimpleSpan(0, filesize), result)]

    while stack:
        span, current_target = stack.pop()

        for box in mp4_iter_boxes(f, span):
            if box.is_container:
                target: tuple[MP4Box, RecursiveBoxes] = (box, RecursiveBoxes([]))
                current_target.append(target)
                stack.append((box.span.payload_span, target[1]))
            else:
                current_target.append(box)

    return result


class MP4BoxStructure(FancyEq):
    boxes: RecursiveBoxes

    def __init__(self: Self, boxes: RecursiveBoxes) -> None:
        self.boxes = boxes

    @staticmethod
    def from_file(file: Path) -> Result["MP4BoxStructure", str]:
        try:
            with file.open("rb") as f:
                mp4_res = is_mp4_file(f)

                if mp4_res is not None:
                    return Err(mp4_res)

                boxes = list_all_boxes_recursively(f)
                return Ok(MP4BoxStructure(boxes))
        except RuntimeError as err:
            return Err(str(err))

    def __str__(self: Self) -> str:
        return f"<MP4BoxStructure boxes: {self.boxes!s}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, RecursiveBoxes):
            return (True, lambda: self.boxes.eq_impl(other))

        if isinstance(other, MP4BoxStructure):
            return (True, lambda: self.boxes.eq_impl(other.boxes))

        return (False, lambda: Err([]))

    def __eq__(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[1]().ok()

    @override
    def support_fancy_eq(self: Self, other: object) -> bool:
        return self.__eq_impl(other)[0]

    @override
    def fancy_eq(self: Self, other: object) -> Optional[list[str]]:
        supports_fancy_eq, cb = self.__eq_impl(other)
        assert supports_fancy_eq
        return cb().err_or(None)

    def __hash__(self: Self) -> int:
        return hash(self.boxes)

    def find_boxes(
        self: Self,
        atom_name: ISOMAtomName,
    ) -> RecursiveBoxes.RecursiveBoxesData:

        boxes_stack: list[RecursiveBoxes.RecursiveBoxesData] = [
            self.boxes.data,
        ]

        result: RecursiveBoxes.RecursiveBoxesData = []

        while len(boxes_stack) != 0:

            boxes = boxes_stack.pop()
            for box_data in boxes:

                box: MP4Box
                if isinstance(box_data, tuple):
                    assert box_data[
                        0
                    ].is_container, "boxes resulting in children have to be a container"
                    box = box_data[0]
                    boxes_stack.append(
                        box_data[1],
                    )
                else:
                    box = box_data

                if box.type == atom_name:
                    result.append(box_data)

        return result


def test_mp4_tagger_parsing(
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
) -> None:

    structure1 = MP4BoxStructure(
        RecursiveBoxes(
            [
                PseudoMP4Box(FTYP_ATOM_NAME, 32),
                (
                    PseudoMP4Box(MOOV_ATOM_NAME, 11824),
                    [
                        PseudoMP4Box(MVHD_ATOM_NAME, 108),
                        PseudoMP4Box(IODS_ATOM_NAME, 42),
                        (
                            PseudoMP4Box(TRAK_ATOM_NAME, 5317),
                            [
                                PseudoMP4Box(
                                    TKHD_ATOM_NAME,
                                    92,
                                ),
                                PseudoMP4Box(EDTS_ATOM_NAME, 36),
                                (
                                    PseudoMP4Box(MDIA_ATOM_NAME, 5181),
                                    [
                                        PseudoMP4Box(MDHD_ATOM_NAME, 32),
                                        PseudoMP4Box(HDLR_ATOM_NAME, 54),
                                        PseudoMP4Box(MINF_ATOM_NAME, 5087),
                                    ],
                                ),
                            ],
                        ),
                        (
                            PseudoMP4Box(TRAK_ATOM_NAME, 6349),
                            [
                                PseudoMP4Box(
                                    TKHD_ATOM_NAME,
                                    92,
                                ),
                                PseudoMP4Box(EDTS_ATOM_NAME, 36),
                                (
                                    PseudoMP4Box(MDIA_ATOM_NAME, 6213),
                                    [
                                        PseudoMP4Box(MDHD_ATOM_NAME, 32),
                                        PseudoMP4Box(HDLR_ATOM_NAME, 54),
                                        PseudoMP4Box(MINF_ATOM_NAME, 6119),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                PseudoMP4Box(FREE_ATOM_NAME, 8),
                PseudoMP4Box(MDAT_ATOM_NAME, 1558160),
            ],
        ),
    )

    structure2 = MP4BoxStructure(
        RecursiveBoxes(
            [
                PseudoMP4Box(FTYP_ATOM_NAME, 32),
                (
                    PseudoMP4Box(MOOV_ATOM_NAME, 3888),
                    [
                        PseudoMP4Box(MVHD_ATOM_NAME, 108),
                        (
                            PseudoMP4Box(TRAK_ATOM_NAME, 3378),
                            [
                                PseudoMP4Box(TKHD_ATOM_NAME, 92),
                                PseudoMP4Box(EDTS_ATOM_NAME, 36),
                                (
                                    PseudoMP4Box(MDIA_ATOM_NAME, 3242),
                                    [
                                        PseudoMP4Box(MDHD_ATOM_NAME, 32),
                                        PseudoMP4Box(HDLR_ATOM_NAME, 55),
                                        PseudoMP4Box(MINF_ATOM_NAME, 3147),
                                    ],
                                ),
                            ],
                        ),
                        (
                            PseudoMP4Box(UDTA_ATOM_NAME, 394),
                            [
                                (
                                    PseudoMP4Box(META_ATOM_NAME, 386),
                                    [
                                        (
                                            PseudoMP4Box(ILST_ATOM_NAME, 341),
                                            [
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9nam"),
                                                    57,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Big Buck Bunny, Sunflower version",
                                                ),
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9ART"),
                                                    76,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Blender Foundation 2008, Janus Bager Kristensen 2013",
                                                ),
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9wrt"),
                                                    41,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Sacha Goedegebure",
                                                ),
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9too"),
                                                    37,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Lavf58.63.100",
                                                ),
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9cmt"),
                                                    89,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Creative Commons Attribution 3.0 - http://bbb3d.renderfarming.net",
                                                ),
                                                PseudoAppleItunesMP4Box(
                                                    ISOMAtomName(b"\xa9gen"),
                                                    33,
                                                    AppleItunesItemDataType.UTF8,
                                                    "Animation",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                PseudoMP4Box(FREE_ATOM_NAME, 8),
                PseudoMP4Box(MDAT_ATOM_NAME, 1041617),
            ],
        ),
    )

    test_files: list[tuple[Path, MP4BoxStructure]] = list(
        zip(
            mp4_test_parse_files.data,
            [structure1, structure2],
            strict=True,
        ),
    )

    for file, result in test_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = MP4BoxStructure.from_file(file)

            assert structure_res == OkResult(), "structure not parsed correctly"

            structure = structure_res.as_ok()

            filesize = file.stat().st_size

            # check box consistency
            boxes_stack: list[tuple[SimpleSpan, RecursiveBoxes.RecursiveBoxesData]] = [
                (SimpleSpan(0, filesize), structure.boxes.data),
            ]

            while len(boxes_stack) != 0:

                boxes_span, boxes = boxes_stack.pop()
                start: int = boxes_span.start
                for box_data in boxes:

                    box: MP4Box
                    if isinstance(box_data, tuple):
                        assert box_data[
                            0
                        ].is_container, (
                            "boxes resulting in children have to be a container"
                        )
                        box = box_data[0]
                        boxes_stack.append(
                            (box.span.payload_span, box_data[1]),
                        )
                    else:
                        box = box_data

                    assert (
                        box.span.total.start == start
                    ), f"Next box start is invalid: {box!s}"

                    start = box.span.total.end

                assert boxes_span.end == start, "boxes don't reach at the parent end"

            assert structure == result, "Parsing was incorrect"


def test_mp4_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Read would overflow bounds [0, 0]: 8 (0 + 8)"),
        (b"helloworld", _("Not a valid ISOM / MP4 file")),
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
            res = is_mp4_file(io)

            assert res is not None, "valid mp4 is incorrect here"

            assert res == err, "incorrect error"


def test_mp4_tagger_language_patching(
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, Language, Language]] = list(
        zip(
            mp4_test_parse_files.data,
            [Language.get_default(), Language.get_default()],
            [
                Language.from_values_unsafe("de", "German"),
                Language.from_values_unsafe("en", "English"),
            ],
            strict=True,
        ),
    )

    types: list[ISOMAtomName] = [SOUN_ATOM_NAME, VIDE_ATOM_NAME]

    for file, old_lang, new_language in test_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = MP4BoxStructure.from_file(file)

            assert structure_res == OkResult(), "structure not parsed correctly"

            with file.open("rb+") as f:
                for mdhd in find_mdhd_boxes_with_type(f, types):
                    old_file_lang = mdhd.read_language(f)

                    assert old_lang.short == old_file_lang, "Old language should match"

                    mdhd.patch_language(f, new_language.short)

            # validate language
            with file.open("rb") as f:
                for mdhd in find_mdhd_boxes_with_type(f, types):
                    old_file_lang = mdhd.read_language(f)

                    assert (
                        new_language.short == old_file_lang
                    ), "New language should be written"

            # validate with ffprobe
            ffprobe_res = ffprobe(file)

            assert ffprobe_res == OkResult(), "FFProbe error"

            for stream in ffprobe_res.as_ok().audio_streams():
                assert stream.is_audio()
                assert stream.raw["tags"]["language"] == new_language.short


def keys_that_are_not_none(dict1: dict[str, Any]) -> list[str]:
    return [key for key, value in dict1.items() if value is not None]


def mp4_has_already_udta_box(file: Path) -> bool:
    structure_res = MP4BoxStructure.from_file(file)

    assert structure_res == OkResult(), "structure not parsed correctly"

    structure = structure_res.as_ok()

    udta_boxes = structure.find_boxes(UDTA_ATOM_NAME)

    return len(udta_boxes) != 0


def test_mp4_tagger_metadata_tags_mutagen(  # noqa: PLR0915
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
    test_manager: ManagerInterface,
) -> None:

    def get_raw_ffprobe_tags(
        result: FFProbeResult,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        val = deepcopy(result.file_info.raw)
        # delete things that might change, but are insignificant for metadata
        del val["size"]
        del val["bit_rate"]

        metadata: dict[str, Any] = {"comment": None, "metadata": None}

        if val.get("tags", None) is not None:
            tags: dict[str, Any] = val["tags"]
            if tags.get("comment", None) is not None:  # noqa: SIM910
                metadata["comment"] = tags["comment"]
                del val["tags"]["comment"]

            for key, value in [*tags.items()]:
                if key.startswith("video_language_detect"):
                    if metadata.get("metadata", None) is None:  # noqa: SIM910
                        metadata["metadata"] = {}

                    metadata["metadata"][key] = value
                    del val["tags"][key]

        return (val, metadata)

    with file_duplicates(mp4_test_parse_files.data) as data:
        test_files: list[tuple[Path, MetadataTags]] = list(
            zip(
                data,
                [
                    MetadataTags(
                        comment="Test comment 1",
                        uuid=uuid4(),
                        language=Language.get_default(),
                        metadata={
                            "test": "str",
                            "dict": {"key1": "value1", "int1": 1414},
                        },
                    ),
                    MetadataTags(
                        comment="Test comment 2",
                        uuid=uuid4(),
                        language=Language.get_default(),
                        metadata={
                            "test": "str",
                            "dict": {"key2": "value2", "int2": 1321},
                        },
                    ),
                ],
                strict=True,
            ),
        )

        for file, tags in test_files:
            with subtests.test("video gets tagged correctly"):
                # mutagen reqrite the udta, if it is already present, otherwise it creates its own, which is not recognized by ffprobe
                # the reason for that is, that it writes the udta box before any trak box, so ffprobe ignores custom tags alias freeform keys
                # see: https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/23427
                is_recognized_by_ffprobe = mp4_has_already_udta_box(file)

                tagger_res = VideoTaggerMutagen.get_handle(file)

                assert tagger_res == OkResult(), "video tagger handle err"

                tagger = tagger_res.as_ok()

                with tagger.writer(manager=test_manager) as w:
                    early_tags = w.get_tags()

                    assert early_tags.uuid is None, "uuid can't be found yet"
                    assert [
                        *early_tags.metadata.items(),
                    ] == [], "no metadata tags can be found already"

                    ffprobe_early_tags = ffprobe(file)

                    assert ffprobe_early_tags == OkResult(), "FFProbe error"

                    raw_early_tags, ffprobe_metadata_early = get_raw_ffprobe_tags(
                        ffprobe_early_tags.as_ok(),
                    )

                    assert (
                        keys_that_are_not_none(ffprobe_metadata_early) == ["comment"]
                        or keys_that_are_not_none(ffprobe_metadata_early) == []
                    ), "raw ffprobe metadata is empty at start"

                    w.write_tags(tags)

                    next_tags = w.get_tags()

                    assert next_tags.uuid == tags.uuid, "UUID was written correctly"
                    assert (
                        next_tags.comment == tags.comment
                    ), "Comment was written correctly"
                    assert (
                        next_tags.unrecognized == early_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert (
                        next_tags.metadata == tags.metadata
                    ), "Metadata was written correctly"

                    ffprobe_next_tags = ffprobe(file)

                    assert ffprobe_next_tags == OkResult(), "FFProbe error"

                    raw_next_tags, ffprobe_metadata_next = get_raw_ffprobe_tags(
                        ffprobe_next_tags.as_ok(),
                    )

                    assert raw_next_tags == raw_early_tags

                    assert ffprobe_metadata_next["comment"] == tags.comment

                    if is_recognized_by_ffprobe:
                        assert ffprobe_metadata_next.get("metadata", None) is not None
                    else:
                        assert ffprobe_metadata_next.get("metadata", None) is None

                    if ffprobe_metadata_next.get("metadata", None) is not None:
                        assert ffprobe_metadata_next["metadata"] == merge_dicts(
                            {
                                f"video_language_detect:{key}": json.dumps(value)
                                for key, value in tags.metadata.items()
                            },
                            {
                                "video_language_detect_uuid:raw": mock.ANY,
                                "video_language_detect_uuid:hex": uuid_to_str(
                                    tags.uuid,
                                ),
                            },
                            "error",
                        )

                    # write again, test that the uuid doesn't get overwritten and that the new data overwrites the old data

                    new_tags = MetadataTags(
                        comment=tags.comment + " - NEW",
                        uuid=uuid4(),
                        language=tags.language,
                        metadata=merge_dicts(
                            tags.metadata,
                            {"new": "a new tag"},
                            "error",
                        ),
                    )

                    assert new_tags.uuid != tags.uuid, "UUID should be unique"

                    w.write_tags(new_tags)

                    write_again_tags = w.get_tags()

                    assert (
                        write_again_tags.uuid == next_tags.uuid
                    ), "UUID was not overwritten"
                    assert write_again_tags.comment == (
                        tags.comment + " - NEW"
                    ), "Comment was overwritten correctly"
                    assert (
                        write_again_tags.unrecognized == next_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert write_again_tags.metadata == merge_dicts(
                        tags.metadata,
                        {"new": "a new tag"},
                        "error",
                    ), "Metadata was overwritten correctly"

                    ffprobe_next_tags = ffprobe(file)

                    assert ffprobe_next_tags == OkResult(), "FFProbe error"

                    ffprobe_again_tags = ffprobe(file)

                    assert ffprobe_again_tags == OkResult(), "FFProbe error"

                    raw_again_tags, ffprobe_metadata_again = get_raw_ffprobe_tags(
                        ffprobe_again_tags.as_ok(),
                    )

                    assert raw_again_tags == raw_early_tags

                    assert ffprobe_metadata_again["comment"] == (
                        tags.comment + " - NEW"
                    )

                    if is_recognized_by_ffprobe:
                        assert ffprobe_metadata_again.get("metadata", None) is not None
                    else:
                        assert ffprobe_metadata_again.get("metadata", None) is None

                    if ffprobe_metadata_again.get("metadata", None) is not None:
                        assert ffprobe_metadata_again["metadata"] == merge_dicts(
                            {
                                f"video_language_detect:{key}": json.dumps(value)
                                for key, value in tags.metadata.items()
                            },
                            {
                                "video_language_detect_uuid:raw": ffprobe_metadata_next[
                                    "metadata"
                                ]["video_language_detect_uuid:raw"],
                                "video_language_detect_uuid:hex": uuid_to_str(
                                    tags.uuid,
                                ),
                                "video_language_detect:new": '"a new tag"',
                            },
                            "error",
                        )


def test_mp4_tagger_metadata_tags_custom(
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
    test_manager: ManagerInterface,
) -> None:

    with file_duplicates(mp4_test_parse_files.data) as data:
        test_files: list[tuple[Path, MetadataTags]] = list(
            zip(
                data,
                [
                    MetadataTags(
                        comment="Test comment 1",
                        uuid=uuid4(),
                        language=Language.get_default(),
                        metadata={
                            "test": "str",
                            "dict": {"key1": "value1", "int1": 1414},
                        },
                    ),
                    MetadataTags(
                        comment="Test comment 2",
                        uuid=uuid4(),
                        language=Language.get_default(),
                        metadata={
                            "test": "str",
                            "dict": {"key2": "value2", "int2": 1321},
                        },
                    ),
                ],
                strict=True,
            ),
        )

        for file, tags in test_files:
            with subtests.test("video gets tagged correctly"):
                tagger_res = VideoTaggerMP4.get_handle(file)

                assert tagger_res == OkResult(), "video tagger handle err"

                tagger = tagger_res.as_ok()

                with tagger.writer(manager=test_manager) as w:
                    early_tags = w.get_tags()

                    assert early_tags.uuid is None, "uuid can't be found yet"
                    assert [
                        *early_tags.metadata.items(),
                    ] == [], "no metadata tags can be found already"

                    w.write_tags(tags)

                    next_tags = w.get_tags()

                    assert next_tags.uuid == tags.uuid, "UUID was written correctly"
                    assert (
                        next_tags.comment == tags.comment
                    ), "Comment was written correctly"
                    assert (
                        next_tags.unrecognized == early_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert (
                        next_tags.metadata == tags.metadata
                    ), "Metadata was written correctly"

                    # write again, test that the uuid doesn't get overwritten and that the new data overwrites the old data

                    new_tags = MetadataTags(
                        comment=tags.comment + " - NEW",
                        uuid=uuid4(),
                        language=tags.language,
                        metadata=merge_dicts(
                            tags.metadata,
                            {"new": "a new tag"},
                            "error",
                        ),
                    )

                    assert new_tags.uuid != tags.uuid, "UUID should be unique"

                    w.write_tags(new_tags)

                    write_again_tags = w.get_tags()

                    assert (
                        write_again_tags.uuid == next_tags.uuid
                    ), "UUID was not overwritten"
                    assert write_again_tags.comment == (
                        tags.comment + " - NEW"
                    ), "Comment was overwritten correctly"
                    assert (
                        write_again_tags.unrecognized == next_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert write_again_tags.metadata == merge_dicts(
                        tags.metadata,
                        {"new": "a new tag"},
                        "error",
                    ), "Metadata was overwritten correctly"


def test_mp4_metadata_tags_apple_custom(
    subtests: SubTests,
    mp4_test_parse_files: TempVideoFiles,
    test_manager: ManagerInterface,
) -> None:

    with file_duplicates(mp4_test_parse_files.data) as data:

        metadatas = [
            MetadataTags(
                comment="Test comment (1)",
                uuid=uuid4(),
                language=Language.get_default(),
                metadata={
                    "test": "str value 1",
                    "dict": {"key1": "value1", "int1": 14141},
                },
            ),
            MetadataTags(
                comment="Test comment (2)",
                uuid=uuid4(),
                language=Language.get_default(),
                metadata={
                    "test": "str value 2",
                    "dict": {"key2": "value2", "int2": 13213},
                },
            ),
        ]

        test_files: list[tuple[Path, MetadataTags, MP4BoxStructure]] = list(
            zip(
                data,
                metadatas,
                [
                    MP4BoxStructure(
                        RecursiveBoxes(
                            [
                                (
                                    PseudoMP4Box(UDTA_ATOM_NAME, 3148),
                                    [
                                        (
                                            PseudoMP4Box(META_ATOM_NAME, 3140),
                                            [
                                                (
                                                    PseudoMP4Box(
                                                        ILST_ATOM_NAME,
                                                        494,
                                                    ),
                                                    [
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            99,
                                                            AppleItunesItemDataType.UTF8,
                                                            json.dumps(
                                                                metadatas[0].metadata[
                                                                    "test"
                                                                ],
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect:test",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            122,
                                                            AppleItunesItemDataType.UTF8,
                                                            uuid_to_str(
                                                                metadatas[0].uuid,
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect_uuid:hex",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            119,
                                                            AppleItunesItemDataType.UTF8,
                                                            json.dumps(
                                                                metadatas[0].metadata[
                                                                    "dict"
                                                                ],
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect:dict",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            106,
                                                            AppleItunesItemDataType.UUID,
                                                            metadatas[0].uuid,
                                                            "lt.totto.vld",
                                                            "video_language_detect_uuid:raw",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9cmt"),
                                                            40,
                                                            AppleItunesItemDataType.UTF8,
                                                            metadatas[0].comment,
                                                        ),
                                                    ],
                                                ),
                                                PseudoMP4Box(FREE_ATOM_NAME, 2601),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    MP4BoxStructure(
                        RecursiveBoxes(
                            [
                                (
                                    PseudoMP4Box(UDTA_ATOM_NAME, 2864),
                                    [
                                        (
                                            PseudoMP4Box(META_ATOM_NAME, 2856),
                                            [
                                                (
                                                    PseudoMP4Box(
                                                        ILST_ATOM_NAME,
                                                        738,
                                                    ),
                                                    [
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9nam"),
                                                            57,
                                                            AppleItunesItemDataType.UTF8,
                                                            "Big Buck Bunny, Sunflower version",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9ART"),
                                                            76,
                                                            AppleItunesItemDataType.UTF8,
                                                            "Blender Foundation 2008, Janus Bager Kristensen 2013",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9wrt"),
                                                            41,
                                                            AppleItunesItemDataType.UTF8,
                                                            "Sacha Goedegebure",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9gen"),
                                                            33,
                                                            AppleItunesItemDataType.UTF8,
                                                            "Animation",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9too"),
                                                            37,
                                                            AppleItunesItemDataType.UTF8,
                                                            "Lavf58.63.100",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            99,
                                                            AppleItunesItemDataType.UTF8,
                                                            json.dumps(
                                                                metadatas[1].metadata[
                                                                    "test"
                                                                ],
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect:test",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            122,
                                                            AppleItunesItemDataType.UTF8,
                                                            uuid_to_str(
                                                                metadatas[1].uuid,
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect_uuid:hex",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            119,
                                                            AppleItunesItemDataType.UTF8,
                                                            json.dumps(
                                                                metadatas[1].metadata[
                                                                    "dict"
                                                                ],
                                                            ),
                                                            "lt.totto.vld",
                                                            "video_language_detect:dict",
                                                        ),
                                                        PseudoAppleItunesMP4FreeformBox(
                                                            106,
                                                            AppleItunesItemDataType.UUID,
                                                            metadatas[1].uuid,
                                                            "lt.totto.vld",
                                                            "video_language_detect_uuid:raw",
                                                        ),
                                                        PseudoAppleItunesMP4Box(
                                                            ISOMAtomName(b"\xa9cmt"),
                                                            40,
                                                            AppleItunesItemDataType.UTF8,
                                                            metadatas[1].comment,
                                                        ),
                                                    ],
                                                ),
                                                PseudoMP4Box(FREE_ATOM_NAME, 2073),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                ],
                strict=True,
            ),
        )

        for file, tags, apple_boxes in test_files:
            with subtests.test("video gets tagged correctly"):
                tagger_res = VideoTaggerMutagen.get_handle(file)

                assert tagger_res == OkResult(), "video tagger handle err"

                tagger = tagger_res.as_ok()

                with tagger.writer(manager=test_manager) as w:
                    early_tags = w.get_tags()

                    assert early_tags.uuid is None, "uuid can't be found yet"
                    assert [
                        *early_tags.metadata.items(),
                    ] == [], "no metadata tags can be found already"

                    w.write_tags(tags)

                    next_tags = w.get_tags()

                    assert next_tags.uuid == tags.uuid, "UUID was written correctly"
                    assert (
                        next_tags.comment == tags.comment
                    ), "Comment was written correctly"
                    assert (
                        next_tags.unrecognized == early_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert (
                        next_tags.metadata == tags.metadata
                    ), "Metadata was written correctly"

                # test the parsing of these apple tags

                structure_res = MP4BoxStructure.from_file(file)
                assert structure_res == OkResult(), "structure not parsed correctly"

                structure = structure_res.as_ok()

                udta_boxes = structure.find_boxes(UDTA_ATOM_NAME)

                assert len(udta_boxes) == 1

                udta_box = MP4BoxStructure(RecursiveBoxes(udta_boxes))

                assert udta_box == apple_boxes
