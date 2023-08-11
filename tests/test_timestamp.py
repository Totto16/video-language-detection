import pytest
from helper.timestamp import parse_int_safely


def test_int_parsing_correct() -> None:
    raw_ints: list[str] = [
        "1",
        "112411414",
        "358013251367513513515",
        "-1212",
        "0",
        "+1212",
    ]
    for int_num in raw_ints:
        assert parse_int_safely(int_num) == int(
            int_num
        ), f"{int_num} should be parsable as int"


def test_int_parsing_wrong() -> None:
    raw_ints: list[str] = ["1.9", "-+112411414", "Infinity", "NaN", "test", "1t", "z0"]

    for int_num in raw_ints:
        assert (
            parse_int_safely(int_num) is None
        ), f"{int_num} shouldn't be parsable as int"


def test_raw_int_parse() -> None:
    raw_ints: list[str] = ["iunfsaf", "sadsada", "dafdsa", "ds", "1saa", "ds1"]

    for int_num in raw_ints:
        with pytest.raises(
            ValueError,
            match=r"^invalid literal for int\(\) with base 10: '.*'$",
        ):
            int(int_num)
