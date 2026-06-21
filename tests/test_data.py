import tempfile
from pathlib import Path

import pytest
from pytest_subtests import SubTests
from test_helper import re_exact_string

from helper.base import load_from_file
from main import AllContent


def test_data_parse_fails(subtests: SubTests) -> None:

    with subtests.test("file doesn't exist"):

        def load() -> None:
            load_from_file(Path("not_present.json"), AllContent)

        with pytest.raises(
            FileNotFoundError,
            match=re_exact_string(
                "[Errno 2] No such file or directory: 'not_present.json'",
            ),
        ):
            load()

    with subtests.test("invalid extension"):

        def load() -> None:
            with tempfile.NamedTemporaryFile(
                suffix=".txt",
            ) as f:
                load_from_file(Path(f.name), AllContent)

        with pytest.raises(
            RuntimeError,
            match=re_exact_string("Data not loadable from 'txt' file!"),
        ):
            load()
