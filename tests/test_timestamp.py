import pytest

from helper.timestamp import parse_int_safely


def test_int_parsing() -> None:
    assert parse_int_safely("1") == 1

    assert parse_int_safely("1.0") is None

    assert parse_int_safely("test") is None


def test_raw_int_parse():
    with pytest.raises(ValueError):
        int("hello")
