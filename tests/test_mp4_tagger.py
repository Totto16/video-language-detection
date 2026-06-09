import json
from copy import deepcopy
from io import BufferedIOBase, BytesIO
from pathlib import Path
from typing import Any, Self
from unittest import mock
from uuid import uuid4

from fixtures import TempVideoFiles, mark_as_used, mp4_test_parse_files, test_manager
from pytest_subtests import SubTests
from test_helper import file_duplicates

from content.language import Language
from content.tagger.mp4_tagger import (
    EDTS_ATOM_NAME,
    FREE_ATOM_NAME,
    FTYP_ATOM_NAME,
    HDLR_ATOM_NAME,
    MDHD_ATOM_NAME,
    MDIA_ATOM_NAME,
    META_ATOM_NAME,
    MINF_ATOM_NAME,
    MOOV_ATOM_NAME,
    SOUN_ATOM_NAME,
    TRAK_ATOM_NAME,
    UDTA_ATOM_NAME,
    VIDE_ATOM_NAME,
    ISOMAtomName,
    MP4Box,
    MP4BoxSpan,
    VideoTaggerMP4,
    find_mdhd_boxes_with_type,
    is_mp4_file,
    mp4_iter_boxes,
)
from content.tagger.mutagen_tagger import VideoTaggerMutagen
from content.tagger.video_tagger import MetadataTags
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
        super().__init__(typ, span=MP4BoxSpan(0, size, 8), is_container=False)


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
            return f"{(indent_str * depth)}<NestedBoxes\n{data[0]!s}\n{RecursiveBoxes.__to_str(data[1], depth=depth+1)}(>)"

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
    ) -> bool:
        # pseudo comparison based on pseudo boxes, alias just size and type!
        if box1.type != box2.type:
            return False

        return box1.span.size == box2.span.size

    @staticmethod
    def __is_elem_eq(
        data1: MP4Box | tuple[MP4Box, RecursiveBoxesData],
        data2: MP4Box | tuple[MP4Box, RecursiveBoxesData],
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
        data1: RecursiveBoxesData,
        data2: RecursiveBoxesData,
        depth: int,
    ) -> bool:
        if len(data1) != len(data2):
            return False

        for d1, d2 in zip(data1, data2, strict=True):
            if not RecursiveBoxes.__is_elem_eq(d1, d2, depth):
                return False

        return True

    def __eq_impl(self: Self, data: RecursiveBoxesData) -> bool:
        return RecursiveBoxes.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveBoxes.__to_str(self.__data, 0, "  ")

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveBoxes):
            return self.__eq_impl(other.__data)

        return False

    def __hash__(self: Self) -> int:
        return hash(*self.__data)


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


class MP4BoxStructure:
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

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveBoxes):
            return self.boxes == other

        if isinstance(other, MP4BoxStructure):
            return self.boxes == other.boxes

        return False

    def __hash__(self: Self) -> int:
        return hash(self.boxes)


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
                        PseudoMP4Box(ISOMAtomName(b"mvhd"), 108),
                        PseudoMP4Box(ISOMAtomName(b"iods"), 42),
                        (
                            PseudoMP4Box(TRAK_ATOM_NAME, 5317),
                            [
                                PseudoMP4Box(
                                    ISOMAtomName(
                                        b"tkhd",
                                    ),
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
                                    ISOMAtomName(
                                        b"tkhd",
                                    ),
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
                PseudoMP4Box(ISOMAtomName(b"mdat"), 1558160),
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
                        PseudoMP4Box(ISOMAtomName(b"mvhd"), 108),
                        (
                            PseudoMP4Box(TRAK_ATOM_NAME, 3378),
                            [
                                PseudoMP4Box(ISOMAtomName(b"tkhd"), 92),
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
                                PseudoMP4Box(META_ATOM_NAME, 386),
                            ],
                        ),
                    ],
                ),
                PseudoMP4Box(FREE_ATOM_NAME, 8),
                PseudoMP4Box(ISOMAtomName(b"mdat"), 1041617),
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

            if structure_res.err():
                msg = f"structure not parsed correctly: {structure_res.as_err()}"
                raise AssertionError(msg)

            structure = structure_res.as_ok()

            # check box consistency
            boxes_stack: list[tuple[int, int, RecursiveBoxes.RecursiveBoxesData]] = [
                (0, file.stat().st_size, structure.boxes.data),
            ]

            while len(boxes_stack) != 0:

                boxes_start, boxes_end, boxes = boxes_stack.pop()
                start: int = boxes_start
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
                            (box.span.payload_start, box.span.end, box_data[1]),
                        )
                    else:
                        box = box_data

                    if box.span.start != start:
                        msg = f"Next box start is invalid, expected {start} but got {box.span.start}: {box!s}"
                        raise AssertionError(msg)

                    start = box.span.end

                if boxes_end != start:
                    msg = f"boxes don't reach at the parent end: size is {boxes_end} but boxes reach only to {start}"
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

            if structure_res.err():
                msg = f"structure not parsed correctly: {structure_res.as_err()}"
                raise AssertionError(msg)

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


def merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for key, value in dict1.items():
        res[key] = value  # noqa: PERF403

    for key, value in dict2.items():
        res[key] = value  # noqa: PERF403

    return res


def keys_that_are_not_none(dict1: dict[str, Any]) -> list[str]:
    return [key for key, value in dict1.items() if value is not None]


def test_mp4_tagger_metadata_tags_mutagen(
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
        test_files: list[tuple[Path, MetadataTags, bool]] = list(
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
                [False, True],
                strict=True,
            ),
        )

        for file, tags, is_recognized_by_ffprobe in test_files:
            with subtests.test("video gets tagged correctly"):
                tagger_res = VideoTaggerMutagen.get_handle(file)

                if tagger_res.err():
                    msg = f"video tagger handle err: {tagger_res.as_err()}"
                    raise AssertionError(msg)

                tagger = tagger_res.as_ok()

                with tagger.writer(manager=test_manager) as w:
                    early_tags = w.get_tags()

                    assert early_tags.uuid is None, "uuid can't be found yet"
                    assert [
                        *early_tags.metadata.items(),
                    ] == [], "no metadata tags can be found already"

                    ffprobe_early_tags = ffprobe(file)

                    if ffprobe_early_tags.err():
                        msg = f"FFProbe error: {ffprobe_early_tags.as_err()}"
                        raise AssertionError(msg)

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

                    if ffprobe_next_tags.err():
                        msg = f"FFProbe error: {ffprobe_next_tags.as_err()}"
                        raise AssertionError(msg)

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
                                "video_language_detect_uuid:file_uuid": mock.ANY,
                            },
                        )

                    # write again, test that the uuid doesn't get overwritten

                    new_tags = MetadataTags(
                        comment=tags.comment,
                        uuid=uuid4(),
                        language=tags.language,
                        metadata=tags.metadata,
                    )

                    assert new_tags.uuid != tags.uuid, "UUID should be unique"

                    w.write_tags(new_tags)

                    write_again_tags = w.get_tags()

                    assert (
                        next_tags.uuid == write_again_tags.uuid
                    ), "UUID was not overwritten"
                    assert (
                        next_tags.comment == write_again_tags.comment
                    ), "Comment was written correctly"
                    assert (
                        next_tags.unrecognized == write_again_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert (
                        next_tags.metadata == write_again_tags.metadata
                    ), "Metadata was written correctly"

                    ffprobe_next_tags = ffprobe(file)

                    if ffprobe_next_tags.err():
                        msg = f"FFProbe error: {ffprobe_next_tags.as_err()}"
                        raise AssertionError(msg)

                    ffprobe_again_tags = ffprobe(file)

                    if ffprobe_again_tags.err():
                        msg = f"FFProbe error: {ffprobe_again_tags.as_err()}"
                        raise AssertionError(msg)

                    raw_again_tags, ffprobe_metadata_again = get_raw_ffprobe_tags(
                        ffprobe_again_tags.as_ok(),
                    )

                    assert raw_again_tags == raw_early_tags

                    assert ffprobe_metadata_again["comment"] == tags.comment

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
                                "video_language_detect_uuid:file_uuid": ffprobe_metadata_next[
                                    "metadata"
                                ][
                                    "video_language_detect_uuid:file_uuid"
                                ],
                            },
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

                if tagger_res.err():
                    msg = f"video tagger handle err: {tagger_res.as_err()}"
                    raise AssertionError(msg)

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

                    # write again, test that the uuid doesn't get overwritten

                    new_tags = MetadataTags(
                        comment=tags.comment,
                        uuid=uuid4(),
                        language=tags.language,
                        metadata=tags.metadata,
                    )

                    assert new_tags.uuid != tags.uuid, "UUID should be unique"

                    w.write_tags(new_tags)

                    write_again_tags = w.get_tags()

                    assert (
                        next_tags.uuid == write_again_tags.uuid
                    ), "UUID was not overwritten"
                    assert (
                        next_tags.comment == write_again_tags.comment
                    ), "Comment was written correctly"
                    assert (
                        next_tags.unrecognized == write_again_tags.unrecognized
                    ), "no new unrecognized tags"
                    assert (
                        next_tags.metadata == write_again_tags.metadata
                    ), "Metadata was written correctly"
