import pytest
from pytest_subtests import SubTests
from test_helper import re_exact_string

from helper.base import load_from_file
from main import AllContent


def test_data_parse_fails(subtests: SubTests) -> None:

    with subtests.test("invalid extension"):

        def load() -> None:
            load_from_file("not_present.txt", AllContent)

        with pytest.raises(
            TypeError,
            match=re_exact_string("TODO"),
        ):
            load()

    with subtests.test("file doesn't exist"):

        def load() -> None:
            load_from_file("not_present.json", AllContent)

        with pytest.raises(
            TypeError,
            match=re_exact_string("TODO"),
        ):
            load()
