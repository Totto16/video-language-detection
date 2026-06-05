from io import BufferedIOBase, BytesIO
from typing import Self

from pytest_subtests import SubTests

from content.tagger.avi_tagger import AVIChunk, AVIList, avi_iter_chunks, is_avi_file
from helper.translation import get_translator

# TODO: force locale in test cases!
_ = get_translator()


class RecursiveChunks:
    __RecursiveChunkData = list[AVIChunk | tuple[AVIChunk, "__RecursiveChunkData"]]
    __data: __RecursiveChunkData

    def __init__(self: Self, data: __RecursiveChunkData) -> None:
        self.__data = data

    def append(self: Self, val: AVIChunk | tuple[AVIChunk, "RecursiveChunks"]) -> None:
        if isinstance(val, tuple):
            self.__data.append((val[0], val[1].__data))  # noqa: SLF001
            return

        self.__data.append(val)

    def top_chunks(self: Self) -> list[AVIChunk]:
        result: list[AVIChunk] = []

        for chunk in self.__data:
            if isinstance(chunk, tuple):
                result.append(chunk[0])
                continue

            result.append(chunk)

        return result

    @staticmethod
    def __single_to_str(
        data: AVIChunk | tuple[AVIChunk, "__RecursiveChunkData"],
        depth: int,
        indent_str: str = " ",
    ) -> str:
        if isinstance(data, tuple):
            return f"{(indent_str * depth)}<NestedBoxes\n{data[0]!s}\n{RecursiveChunks.__to_str(data[1], depth=depth+1)}>"

        return f"{(indent_str * depth)}<SimpleBox {data!s}>"

    @staticmethod
    def __to_str(
        data: __RecursiveChunkData,
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
        data1: AVIChunk | tuple[AVIChunk, __RecursiveChunkData],
        data2: AVIChunk | tuple[AVIChunk, __RecursiveChunkData],
        depth: int,
    ) -> bool:
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            c1, d1 = data1
            c2, d2 = data2

            if not RecursiveChunks.__is_chunk_eq(c1, c2):
                return False

            return RecursiveChunks.__eq_impl_both(d1, d2, depth=depth + 1)
        if isinstance(data1, AVIChunk) and isinstance(data2, AVIChunk):
            return RecursiveChunks.__is_chunk_eq(data1, data2)

        return False

    @staticmethod
    def __eq_impl_both(
        data1: __RecursiveChunkData,
        data2: __RecursiveChunkData,
        depth: int,
    ) -> bool:
        if len(data1) != len(data2):
            return False

        for d1, d2 in zip(data1, data2, strict=True):
            if not RecursiveChunks.__is_elem_eq(d1, d2, depth):
                return False

        return True

    def __eq_impl(self: Self, data: __RecursiveChunkData) -> bool:
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
                target: tuple[AVIChunk, RecursiveChunks] = (chunk, RecursiveChunks([]))
                current_target.append(target)
                stack.append((chunk.span.payload_start, chunk.span.end, target[1]))
            else:
                current_target.append(chunk)

    return result


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
