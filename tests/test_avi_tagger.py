import json
from collections.abc import Callable
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional, Self, cast, override
from uuid import uuid4

from conftest import FancyEq
from fixtures import TempVideoFiles, avi_test_parse_files, mark_as_used, test_manager
from pytest_subtests import SubTests
from test_helper import OkResult, file_duplicates

from content.language import Language
from content.tagger.avi_tagger import (
    AUDS_FOURCC,
    AVI__FOURCC,
    FOURCC,
    LIST_FOURCC,
    RIFF_FOURCC,
    VIDS_FOURCC,
    VLD_FFMPEG_RAW_STRING_JSON_CHUNK_FOURCC,
    VLD_KEY_VALUE_FOURCC,
    AVIChunk,
    AVIChunkSpan,
    AVIList,
    VideoTaggerAVI,
    avi_iter_chunks,
    find_strh_chunks_with_type,
    is_avi_file,
)
from content.tagger.parser import SimpleSpan
from content.tagger.utils import merge_dicts
from content.tagger.video_tagger import MetadataTags, uuid_to_str
from helper.decorator import decorate_class
from helper.ffprobe import FFProbeResult, ffprobe
from helper.manager import ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

mark_as_used(avi_test_parse_files)
mark_as_used(test_manager)

# TODO: force locale in test cases!
_ = get_translator()

@decorate_class(slots=True)
class PseudoAVIChunk(AVIChunk):

    def __init__(self: Self, fourcc: FOURCC, size: int) -> None:
        super().__init__(
            fourcc,
            span=AVIChunkSpan(SimpleSpan(0, size), 8),
            is_list=False,
        )

@decorate_class(slots=True)
class PseudoAVIList(AVIList):

    def __init__(self: Self, fourcc: FOURCC, size: int, typ: FOURCC) -> None:
        super().__init__(PseudoAVIChunk(fourcc, size), typ)

@decorate_class(slots=True)
class PseudoMOVIChunk(PseudoAVIList):
    children: int

    def __init__(self: Self, children: int, size: int) -> None:
        super().__init__(LIST_FOURCC, size, FOURCC(b"movi"))
        self.children = children

@decorate_class(slots=True)
class RecursiveChunks:
    RecursiveChunkData = list[AVIChunk | tuple[AVIList, "RecursiveChunkData"]]
    __data: RecursiveChunkData

    def __init__(self: Self, data: RecursiveChunkData) -> None:
        self.__data = data

    def append(self: Self, val: AVIChunk | tuple[AVIList, RecursiveChunks]) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))  # noqa: SLF001
            return

        self.__data.append(val)

    @property
    def data(self: Self) -> RecursiveChunkData:
        return self.__data

    @staticmethod
    def __single_to_str(
        data: AVIChunk | tuple[AVIChunk, RecursiveChunkData],
        depth: int,
        indent_str: str = " ",
    ) -> str:
        if isinstance(data, tuple) and isinstance(data[0], AVIList):
            if data[0].type == FOURCC(b"movi"):
                return f"{(indent_str * depth)}<MoviChunk children: {len(data[1])} span: {data[0].span}>"

            return f"{(indent_str * depth)}<NestedChunks\n{data[0]!s}\n{RecursiveChunks.__to_str(data[1], depth=depth+1)}>"

        return f"{(indent_str * depth)}<SimpleChunk {data!s}>"

    @staticmethod
    def __to_str(
        data: RecursiveChunkData,
        depth: int,
        indent_str: str = " ",
    ) -> str:

        return (f"\n{(indent_str * depth)}").join(
            RecursiveChunks.__single_to_str(dat, depth, indent_str=indent_str)
            for dat in data
        )

    @staticmethod
    def __is_chunk_eq(
        chunk1: AVIChunk,
        chunk2: AVIChunk,
        depth: int,
    ) -> Result[None, list[str]]:
        # pseudo comparison based on pseudo chunks, alias just size and type! (and list type if it is one)
        if chunk1.fourcc != chunk2.fourcc:
            return Err[list[str]](
                [
                    "FOURCC of data is not eq:",
                    str(chunk1.fourcc),
                    str(chunk2.fourcc),
                    f"Depth {depth}",
                    str(chunk1),
                    str(chunk2),
                ],
            )

        if chunk1.span.total.size != chunk2.span.total.size:
            return Err[list[str]](
                [
                    "Sizeof data is not eq:",
                    str(chunk1.span.total.size),
                    str(chunk2.span.total.size),
                    f"Depth {depth}",
                    str(chunk1),
                    str(chunk2),
                ],
            )
        if isinstance(chunk1, AVIList):
            if not isinstance(chunk2, AVIList):
                return Err[list[str]](
                    [
                        "AVIList on left side of eq, but right side is not correct class:",
                        str(type(chunk1)),
                        str(type(chunk2)),
                        f"Depth {depth}",
                        str(chunk1),
                        str(chunk2),
                    ],
                )

            if chunk1.type != chunk2.type:
                return Err[list[str]](
                    [
                        "FOURCC of list type is not eq:",
                        str(chunk1.type),
                        str(chunk2.type),
                        f"Depth {depth}",
                        str(chunk1),
                        str(chunk2),
                    ],
                )

            return Ok(None)

        if isinstance(chunk2, AVIList):
            return Err[list[str]](
                [
                    "AVIList on right side of eq, but left side is not correct class:",
                    str(type(chunk1)),
                    str(type(chunk2)),
                    f"Depth {depth}",
                    str(chunk1),
                    str(chunk2),
                ],
            )

        return Ok(None)

    @staticmethod
    def __is_elem_eq(
        data1: AVIChunk | tuple[AVIChunk, RecursiveChunkData],
        data2: AVIChunk | tuple[AVIChunk, RecursiveChunkData],
        depth: int,
    ) -> Result[None, list[str]]:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            c1, d1 = data1
            c2, d2 = data2

            res = RecursiveChunks.__is_chunk_eq(c1, c2, depth)
            if res.err():
                return res

            if isinstance(c1, AVIList) and c1.type == FOURCC(b"movi"):
                if not isinstance(c2, PseudoMOVIChunk):
                    return Err[list[str]](
                        [
                            "MOVIChunk on left side of eq, but right side is not correct class:",
                            str(type(c1)),
                            str(type(c2)),
                            f"Depth {depth}",
                            str(c1),
                            str(c2),
                        ],
                    )

                if c2.children != len(d1):
                    return Err[list[str]](
                        [
                            "Sizeof MOVIChunk children is not eq:",
                            str(c2.children),
                            str(len(d1)),
                            f"Depth {depth}",
                            str(c2),
                            str(d1),
                        ],
                    )

                return Ok(None)

            return RecursiveChunks.__eq_impl_both(d1, d2, depth=depth + 1)

        if isinstance(data1, AVIChunk) and isinstance(data2, AVIChunk):
            return RecursiveChunks.__is_chunk_eq(data1, data2, depth)

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
        data1: RecursiveChunkData,
        data2: RecursiveChunkData,
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
            res = RecursiveChunks.__is_elem_eq(d1, d2, depth)
            if res.err():
                return res

        return Ok(None)

    def __eq_impl(self: Self, data: RecursiveChunkData) -> Result[None, list[str]]:
        return RecursiveChunks.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveChunks.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveChunks.__to_str(self.__data, 0, "  ")

    def eq_impl(self: Self, other: RecursiveChunks) -> Result[None, list[str]]:
        return self.__eq_impl(other.data)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveChunks):
            return self.__eq_impl(other.__data).ok()

        return False

    def __hash__(self: Self) -> int:
        return hash(*self.__data)


def list_all_chunks_recursively(f: BinaryIO) -> RecursiveChunks:
    f.seek(0, 2)
    filesize = f.tell()

    result: RecursiveChunks = RecursiveChunks([])

    stack: list[tuple[SimpleSpan, RecursiveChunks]] = [
        (SimpleSpan(0, filesize), result),
    ]

    while stack:
        span, current_target = stack.pop()

        for chunk in avi_iter_chunks(f, span):
            if chunk.is_list:
                if not isinstance(chunk, AVIList):
                    msg = "Invalid AVIList: type not dispatched to correct class"
                    raise ValueError(msg)

                target: tuple[AVIList, RecursiveChunks] = (chunk, RecursiveChunks([]))
                current_target.append(target)
                stack.append((chunk.span.payload_span, target[1]))
            else:
                current_target.append(chunk)

    return result

@decorate_class(slots=True)
class AVIChunkStructure(FancyEq):
    chunks: RecursiveChunks

    def __init__(self: Self, chunks: RecursiveChunks) -> None:
        self.chunks = chunks

    @staticmethod
    def from_file(file: Path) -> Result[AVIChunkStructure, str]:
        try:
            with file.open("rb") as f:
                avi_res = is_avi_file(f)

                if avi_res is not None:
                    return Err(avi_res)

                chunks = list_all_chunks_recursively(f)
                return Ok(AVIChunkStructure(chunks))
        except RuntimeError as err:
            return Err(str(err))

    def __str__(self: Self) -> str:
        return f"<AVIChunkStructure chunks: {self.chunks!s}>"

    def __repr__(self: Self) -> str:
        return str(self)

    def __eq_impl(
        self: Self,
        other: object,
    ) -> tuple[bool, Callable[[], Result[None, list[str]]]]:
        if isinstance(other, RecursiveChunks):
            return (True, lambda: self.chunks.eq_impl(other))

        if isinstance(other, AVIChunkStructure):
            return (True, lambda: self.chunks.eq_impl(other.chunks))

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
        return hash(self.chunks)


def test_avi_tagger_parsing(
    subtests: SubTests,
    avi_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, AVIChunkStructure]] = list(
        zip(
            avi_test_parse_files.data,
            [
                AVIChunkStructure(
                    RecursiveChunks(
                        [
                            (
                                PseudoAVIList(RIFF_FOURCC, 742478, AVI__FOURCC),
                                [
                                    (
                                        PseudoAVIList(
                                            LIST_FOURCC,
                                            8902,
                                            FOURCC(b"hdrl"),
                                        ),
                                        [
                                            PseudoAVIChunk(FOURCC(b"avih"), 64),
                                            (
                                                PseudoAVIList(
                                                    LIST_FOURCC,
                                                    4328,
                                                    FOURCC(b"strl"),
                                                ),
                                                [
                                                    PseudoAVIChunk(FOURCC(b"strh"), 64),
                                                    PseudoAVIChunk(FOURCC(b"strf"), 48),
                                                    PseudoAVIChunk(
                                                        FOURCC(
                                                            b"JUNK",
                                                        ),
                                                        4128,
                                                    ),
                                                    PseudoAVIChunk(FOURCC(b"vprp"), 76),
                                                ],
                                            ),
                                            (
                                                PseudoAVIList(
                                                    LIST_FOURCC,
                                                    4230,
                                                    FOURCC(b"strl"),
                                                ),
                                                [
                                                    PseudoAVIChunk(FOURCC(b"strh"), 64),
                                                    PseudoAVIChunk(FOURCC(b"strf"), 26),
                                                    PseudoAVIChunk(
                                                        FOURCC(
                                                            b"JUNK",
                                                        ),
                                                        4128,
                                                    ),
                                                ],
                                            ),
                                            PseudoAVIChunk(
                                                FOURCC(
                                                    b"JUNK",
                                                ),
                                                268,
                                            ),
                                        ],
                                    ),
                                    (
                                        PseudoAVIList(LIST_FOURCC, 34, FOURCC(b"INFO")),
                                        [
                                            PseudoAVIChunk(FOURCC(b"ISFT"), 22),
                                        ],
                                    ),
                                    PseudoAVIChunk(FOURCC(b"JUNK"), 1024),
                                    (PseudoMOVIChunk(2336, 695122), []),
                                    PseudoAVIChunk(FOURCC(b"idx1"), 37384),
                                ],
                            ),
                        ],
                    ),
                ),
            ],
            strict=True,
        ),
    )

    for file, result in test_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = AVIChunkStructure.from_file(file)

            assert structure_res == OkResult(), "structure not parsed correctly"

            structure = structure_res.as_ok()

            filesize = file.stat().st_size
            # check chunk consistency
            chunks_stack: list[
                tuple[SimpleSpan, RecursiveChunks.RecursiveChunkData]
            ] = [
                (SimpleSpan(0, filesize), structure.chunks.data),
            ]

            while len(chunks_stack) != 0:

                chunks_span, chunks = chunks_stack.pop()
                start: int = chunks_span.start
                for chunk_data in chunks:

                    chunk: AVIChunk
                    if isinstance(chunk_data, tuple):
                        assert chunk_data[
                            0
                        ].is_list, "chunks resulting in children have to be a list"
                        chunk = chunk_data[0]
                        chunks_stack.append(
                            (chunk.span.payload_span, chunk_data[1]),
                        )
                    else:
                        chunk = chunk_data

                    assert (
                        chunk.span.total.start == start
                    ), f"Next chunk start is invalid: {chunk!s}"

                    start = chunk.span.total.end

                    # adjust padding
                    if (start % 2) != 0:
                        start += 1

                assert chunks_span.end == start, "chunks don't reach at the parent end"

            assert structure == result, "Parsing was incorrect"


def test_avi_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Read would overflow bounds [0, 0]: 8 (0 + 8)"),
        (
            b"HELLO WORLD",
            "Invalid AVI Chunk size: It overflows the parent chunk: 1331109975 > 11",
        ),
        (
            b"LIST\x22\x00\x00\x00WORLD",
            "Invalid AVI Chunk size: It overflows the parent chunk: 42 > 13",
        ),
        (b"LIST\x02\x00\x00\x00WO", "Read would overflow bounds [8, 10]: 12 (8 + 4)"),
        (
            b"LIST\x04\x00\x00\x00HELO",
            "RIFF/AVI file has valid chunk, but it is not the correct starting chunk: b'LIST'",
        ),
        (
            b"RIFF\x04\x00\x00\x00LD12",
            "RIFF/AVI file has valid chunk, but it is not the correct starting chunk, list type invalid: b'LD12'",
        ),
    ]

    for data, err in test_data:
        with subtests.test("invalid video gets detected correctly"):
            io = BytesIO(data)
            res = is_avi_file(io)

            assert res is not None, "valid avi is incorrect here"

            assert res == err, "incorrect error"


def test_avi_tagger_language_patching(
    subtests: SubTests,
    avi_test_parse_files: TempVideoFiles,
) -> None:

    test_files: list[tuple[Path, Language, Language]] = list(
        zip(
            avi_test_parse_files.data,
            [Language.get_default()],
            [Language.from_values_unsafe("de", "German")],
            strict=True,
        ),
    )

    types: list[FOURCC] = [AUDS_FOURCC, VIDS_FOURCC]

    for file, old_lang, new_language in test_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = AVIChunkStructure.from_file(file)

            assert structure_res == OkResult(), "structure not parsed correctly"

            with file.open("rb+") as f:
                for strh in find_strh_chunks_with_type(f, types):
                    old_file_lang = strh.read_language(f)

                    assert old_lang.short == old_file_lang, "Old language should match"

                    strh.patch_language(f, new_language.short)

            # validate language
            with file.open("rb") as f:
                for strh in find_strh_chunks_with_type(f, types):
                    old_file_lang = strh.read_language(f)

                    assert (
                        new_language.short == old_file_lang
                    ), "New language should be written"


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

            if key in [LIST_FOURCC.value.decode(), VLD_KEY_VALUE_FOURCC.value.decode()]:
                if metadata.get("artifacts", None) is None:  # noqa: SIM910
                    metadata["artifacts"] = {}

                metadata["artifacts"][key] = value
                del val["tags"][key]

            if key == VLD_FFMPEG_RAW_STRING_JSON_CHUNK_FOURCC.value.decode():
                if metadata.get("metadata", None) is None:  # noqa: SIM910
                    metadata["metadata"] = {}

                if metadata.get("errors", None) is None:  # noqa: SIM910
                    metadata["errors"] = []

                raw_value = json.loads(value)

                if isinstance(raw_value, dict):
                    for d_key, d_val in raw_value.items():
                        if d_key == "metadata":
                            if not isinstance(d_val, dict):
                                msg = f"Implementation error: not a dict: {d_val}"
                                raise RuntimeError(msg)
                            metadata["metadata"] = merge_dicts(
                                metadata["metadata"],
                                {
                                    f"video_language_detect:{i_k}": i_v
                                    for i_k, i_v in d_val.items()
                                },
                                "error",
                            )
                        elif d_key == "uuid_hex":
                            metadata["metadata"] = merge_dicts(
                                metadata["metadata"],
                                {"video_language_detect_uuid:hex": d_val},
                                "error",
                            )
                        else:
                            metadata["errors"].append({d_key: d_val})

                else:
                    metadata["errors"].append(raw_value)

                del val["tags"][key]

    return (val, metadata)


def keys_that_are_not_none(dict1: dict[str, Any]) -> list[str]:
    return [key for key, value in dict1.items() if value is not None]


def test_avi_tagger_metadata_tags_custom(
    subtests: SubTests,
    avi_test_parse_files: TempVideoFiles,
    test_manager: ManagerInterface,
) -> None:

    with file_duplicates(avi_test_parse_files.data) as data:
        test_files: list[tuple[Path, MetadataTags]] = list(
            zip(
                data,
                [
                    MetadataTags(
                        comment="Test comment 1",
                        uuid=uuid4(),
                        metadata={
                            "test": "str",
                            "dict": {"key1": "value1", "int1": 1414},
                        },
                    ),
                ],
                strict=True,
            ),
        )

        for file, tags in test_files:
            with subtests.test("video gets tagged correctly"):
                tagger_res = VideoTaggerAVI.get_handle(file)

                assert tagger_res == OkResult(), "video tagger handle err"

                tagger = tagger_res.as_ok()

                with tagger.rw_ctx(manager=test_manager) as ctx:
                    early_tags = ctx.get_tags()

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

                    ctx.write_tags(tags)

                    next_tags = ctx.get_tags()

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

                    assert keys_that_are_not_none(ffprobe_metadata_next) == [
                        "comment",
                        "metadata",
                        "artifacts",
                        "errors",
                    ], "raw ffprobe metadata is correct later on"

                    assert ffprobe_metadata_next["comment"] == tags.comment

                    assert ffprobe_metadata_next["artifacts"] == {
                        "LIST": "vldlvldk\x1a",
                        "vldk": "vlds'",
                    }

                    assert ffprobe_metadata_next["errors"] == []

                    assert ffprobe_metadata_next.get("metadata", None) is not None

                    assert ffprobe_metadata_next["metadata"] == merge_dicts(
                        cast(
                            dict[str, Any],
                            {
                                f"video_language_detect:{key}": value
                                for key, value in tags.metadata.items()
                            },
                        ),
                        {
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
                        metadata=merge_dicts(
                            tags.metadata,
                            {"new": "a new tag"},
                            "error",
                        ),
                    )

                    assert new_tags.uuid != tags.uuid, "UUID should be unique"

                    ctx.write_tags(new_tags)

                    write_again_tags = ctx.get_tags()

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
                        cast(dict[str, Any], {"new": "a new tag"}),
                        "error",
                    ), "Metadata was overwritten correctly"

                    ffprobe_again_tags = ffprobe(file)

                    assert ffprobe_again_tags == OkResult(), "FFProbe error"

                    raw_again_tags, ffprobe_metadata_again = get_raw_ffprobe_tags(
                        ffprobe_again_tags.as_ok(),
                    )

                    assert raw_again_tags == raw_early_tags

                    assert keys_that_are_not_none(ffprobe_metadata_next) == [
                        "comment",
                        "metadata",
                        "artifacts",
                        "errors",
                    ], "raw ffprobe metadata is correct later on"

                    assert ffprobe_metadata_again["comment"] == (
                        tags.comment + " - NEW"
                    )

                    assert ffprobe_metadata_next["artifacts"] == {
                        "LIST": "vldlvldk\x1a",
                        "vldk": "vlds'",
                    }

                    assert ffprobe_metadata_next["errors"] == []

                    assert ffprobe_metadata_again.get("metadata", None) is not None

                    assert ffprobe_metadata_again["metadata"] == merge_dicts(
                        {
                            f"video_language_detect:{key}": value
                            for key, value in tags.metadata.items()
                        },
                        cast(
                            dict[str, Any],
                            {
                                "video_language_detect_uuid:hex": uuid_to_str(
                                    tags.uuid,
                                ),
                                "video_language_detect:new": "a new tag",
                            },
                        ),
                        "error",
                    )
