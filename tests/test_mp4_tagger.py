from fixtures import TempMp4Files, mark_as_used, temp_mp4_files
from pytest_subtests import SubTests

mark_as_used(temp_mp4_files)

def test_mp4_tagger(
    subtests: SubTests,
    temp_mp4_files: TempMp4Files,
) -> None:
    for video in temp_mp4_files.data:
        with subtests.test("video gets parsed correctly"):
            assert video is None, "TODO"
