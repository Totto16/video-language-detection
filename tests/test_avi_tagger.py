from io import BufferedIOBase, BytesIO
from pathlib import Path
from typing import Self

from fixtures import TempVideoFiles, avi_test_parse_files, mark_as_used
from pytest_subtests import SubTests
from test_helper import OkResult

from content.language import Language
from content.tagger.avi_tagger import (
    AUDS_FOURCC,
    AVI__FOURCC,
    FOURCC,
    LIST_FOURCC,
    RIFF_FOURCC,
    VIDS_FOURCC,
    AVIChunk,
    AVIChunkSpan,
    AVIList,
    avi_iter_chunks,
    find_strh_chunks_with_type,
    is_avi_file,
)
from helper.result import Err, Ok, Result
from helper.translation import get_translator

mark_as_used(avi_test_parse_files)


# TODO: force locale in test cases!
_ = get_translator()


class PseudoAVIChunk(AVIChunk):

    def __init__(self: Self, fourcc: FOURCC, size: int) -> None:
        super().__init__(fourcc, span=AVIChunkSpan(0, size, 8), is_list=False)


class PseudoAVIList(AVIList):

    def __init__(self: Self, fourcc: FOURCC, size: int, typ: FOURCC) -> None:
        super().__init__(PseudoAVIChunk(fourcc, size), typ)


class PseudoMOVIChunk(PseudoAVIList):
    children: int

    def __init__(self: Self, children: int, size: int) -> None:
        super().__init__(LIST_FOURCC, size, FOURCC(b"movi"))
        self.children = children


class RecursiveChunks:
    RecursiveChunkData = list[AVIChunk | tuple[AVIList, "RecursiveChunkData"]]
    __data: RecursiveChunkData

    def __init__(self: Self, data: RecursiveChunkData) -> None:
        self.__data = data

    def append(self: Self, val: AVIChunk | tuple[AVIList, "RecursiveChunks"]) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))  # noqa: SLF001
            return

        self.__data.append(val)

    @property
    def data(self: Self) -> RecursiveChunkData:
        return self.__data

    @staticmethod
    def __single_to_str(
        data: AVIChunk | tuple[AVIChunk, "RecursiveChunkData"],
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
    ) -> bool:
        # pseudo comparison based on pseudo chunks, alias just size and type! (and list type if it is one)
        if chunk1.fourcc != chunk2.fourcc:
            return False

        if chunk1.span.size != chunk2.span.size:
            return False

        if isinstance(chunk1, AVIList) and isinstance(chunk2, AVIList):
            return chunk1.type == chunk2.type

        return not (isinstance(chunk1, AVIList) or isinstance(chunk2, AVIList))

    @staticmethod
    def __is_elem_eq(
        data1: AVIChunk | tuple[AVIChunk, RecursiveChunkData],
        data2: AVIChunk | tuple[AVIChunk, RecursiveChunkData],
        depth: int,
    ) -> bool:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            c1, d1 = data1
            c2, d2 = data2

            if not RecursiveChunks.__is_chunk_eq(c1, c2):
                return False

            if isinstance(c1, AVIList) and c1.type == FOURCC(b"movi"):
                if not isinstance(c2, PseudoMOVIChunk):
                    msg = "Found MOVI Chunk without matching PseudoMOVIChunk"
                    raise RuntimeError(msg)

                return c2.children == len(d1)

            return RecursiveChunks.__eq_impl_both(d1, d2, depth=depth + 1)
        if isinstance(data1, AVIChunk) and isinstance(data2, AVIChunk):
            return RecursiveChunks.__is_chunk_eq(data1, data2)

        return False

    @staticmethod
    def __eq_impl_both(
        data1: RecursiveChunkData,
        data2: RecursiveChunkData,
        depth: int,
    ) -> bool:
        if len(data1) != len(data2):
            return False

        for d1, d2 in zip(data1, data2, strict=True):
            if not RecursiveChunks.__is_elem_eq(d1, d2, depth):
                return False

        return True

    def __eq_impl(self: Self, data: RecursiveChunkData) -> bool:
        return RecursiveChunks.__eq_impl_both(self.__data, data, depth=0)

    def __str__(self: Self) -> str:
        return RecursiveChunks.__to_str(self.__data, 0, "\t")

    def __repr__(self: Self) -> str:
        return RecursiveChunks.__to_str(self.__data, 0, "  ")

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveChunks):
            return self.__eq_impl(other.__data)

        return False

    def __hash__(self: Self) -> int:
        return hash(*self.__data)


def list_all_chunks_recursively(f: BufferedIOBase) -> RecursiveChunks:
    f.seek(0, 2)
    filesize = f.tell()

    result: RecursiveChunks = RecursiveChunks([])

    stack: list[tuple[int, int, RecursiveChunks]] = [(0, filesize, result)]

    while stack:
        start, end, current_target = stack.pop()

        for chunk in avi_iter_chunks(f, start, end):
            if chunk.is_list:
                if not isinstance(chunk, AVIList):
                    msg = "Invalid AVIList: type not dispatched to correct class"
                    raise ValueError(msg)

                target: tuple[AVIList, RecursiveChunks] = (chunk, RecursiveChunks([]))
                current_target.append(target)
                stack.append((chunk.span.payload_start, chunk.span.end, target[1]))
            else:
                current_target.append(chunk)

    return result


class AVIChunkStructure:
    chunks: RecursiveChunks

    def __init__(self: Self, chunks: RecursiveChunks) -> None:
        self.chunks = chunks

    @staticmethod
    def from_file(file: Path) -> Result["AVIChunkStructure", str]:
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

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, RecursiveChunks):
            return self.chunks == other

        if isinstance(other, AVIChunkStructure):
            return self.chunks == other.chunks

        return False

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

            # check chunk consistency
            chunks_stack: list[tuple[int, int, RecursiveChunks.RecursiveChunkData]] = [
                (0, file.stat().st_size, structure.chunks.data),
            ]

            while len(chunks_stack) != 0:

                chunks_start, chunks_end, chunks = chunks_stack.pop()
                start: int = chunks_start
                for chunk_data in chunks:

                    chunk: AVIChunk
                    if isinstance(chunk_data, tuple):
                        assert chunk_data[
                            0
                        ].is_list, "chunks resulting in children have to be a list"
                        chunk = chunk_data[0]
                        chunks_stack.append(
                            (chunk.span.payload_start, chunk.span.end, chunk_data[1]),
                        )
                    else:
                        chunk = chunk_data

                    assert (
                        chunk.span.start == start
                    ), f"Next chunk start is invalid: {chunk!s}"

                    start = chunk.span.end

                    # adjust padding
                    if (start % 2) != 0:
                        start += 1

                assert chunks_end == start, "chunks don't reach at the parent end"

            assert structure == result, "Parsing was incorrect"


def test_avi_invalid_bytes(
    subtests: SubTests,
) -> None:

    test_data: list[tuple[bytes, str]] = [
        (b"", "Read failed to produce 8 bytes, got 0"),
        (b"HELLO WORLD", "Not a valid RIFF / AVI file"),
        (b"LIST WORLD", "Read failed to produce 4 bytes, got 2"),
        (
            b"LIST WORLD12",
            "RIFF/AVI file has valid chunk, but it is not the correct starting chunk: b'LIST'",
        ),
        (
            b"RIFF WORLD12",
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
