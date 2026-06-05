from io import BufferedIOBase
from pathlib import Path
from typing import Self

from fixtures import mark_as_used, temp_mp4_files
from pytest_subtests import SubTests

from content.tagger.mp4_tagger import (
    MP4Box,
    is_mp4_file,
    mp4_iter_boxes,
)
from helper.result import Result

mark_as_used(temp_mp4_files)

Mp4BoxStructure__GetResult = Result["Mp4BoxStructure", str]


RecursiveBoxes = list[MP4Box | tuple[MP4Box, "RecursiveBoxes"]]


def list_all_boxes_recursively(f: BufferedIOBase) -> RecursiveBoxes:
    f.seek(0, 2)
    filesize = f.tell()

    result: RecursiveBoxes = []

    stack: list[tuple[int, int, RecursiveBoxes]] = [(0, filesize, result)]

    while stack:
        start, end, current_target = stack.pop()

        for box in mp4_iter_boxes(f, start, end):
            if box.container:
                target: tuple[MP4Box, RecursiveBoxes] = (box, [])
                current_target.append(target)
                stack.append((box.span.payload_start, box.span.end, target[1]))
            else:
                current_target.append(box)

    return result


class Mp4BoxStructure:
    boxes: RecursiveBoxes

    def __init__(self: Self, boxes: RecursiveBoxes) -> None:
        self.boxes = boxes

    @staticmethod
    def from_file(file: Path) -> Mp4BoxStructure__GetResult:
        with file.open("rb") as f:
            mp4_res = is_mp4_file(f)

            if mp4_res is not None:
                return Mp4BoxStructure__GetResult.err(mp4_res)

            boxes = list_all_boxes_recursively(f)
            return Mp4BoxStructure__GetResult.ok(Mp4BoxStructure(boxes))

        return Mp4BoxStructure__GetResult.err("TODO")


def test_mp4_tagger_parsing(
    subtests: SubTests,
) -> None:

    todo_files: list[tuple[Path, Mp4BoxStructure]] = [
        (
            Path(
                "/home/totto/Code/video-language-detection/file_example_MP4_480_1_5MG.mp4"
            ),
            Mp4BoxStructure([]),
        ),
    ]

    for file, result in todo_files:
        with subtests.test("video gets parsed correctly"):
            structure_res = Mp4BoxStructure.from_file(file)

            assert structure_res.is_ok(), "structure parsed correctly"

            structure = structure_res.get_ok()
            
            print(structure)

            assert structure == result, "Parsing was correct"
