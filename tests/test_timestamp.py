from datetime import timedelta

import pytest
from helper.timestamp import Timestamp, parse_int_safely
from pytest_subtests import SubTests


def test_int_parsing_correct(subtests: SubTests) -> None:
    raw_ints: list[str] = [
        "1",
        "112411414",
        "358013251367513513515",
        "-1212",
        "0",
        "+1212",
    ]
    for int_num in raw_ints:
        with subtests.test():
            assert parse_int_safely(int_num) == int(
                int_num,
            ), f"{int_num} is parsable as int"


def test_int_parsing_wrong(subtests: SubTests) -> None:
    raw_ints: list[str] = ["1.9", "-+112411414", "Infinity", "NaN", "test", "1t", "z0"]

    for int_num in raw_ints:
        with subtests.test():
            assert parse_int_safely(int_num) is None, f"{int_num} isn't parsable as int"


def test_raw_int_parse(subtests: SubTests) -> None:
    raw_ints: list[str] = ["iunfsaf", "sadsada", "dafdsa", "ds", "1saa", "ds1"]

    for int_num in raw_ints:
        with subtests.test(), pytest.raises(
            ValueError,
            match=r"^invalid literal for int\(\) with base 10: '.*'$",
        ):
            int(int_num)


def test_timedelta_operator_errors(subtests: SubTests) -> None:
    with subtests.test("lt"), pytest.raises(
        TypeError,
        match=r"^'<' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() < True

    with subtests.test("le"), pytest.raises(
        TypeError,
        match=r"^'<=' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() <= True

    with subtests.test("gt"), pytest.raises(
        TypeError,
        match=r"^'>' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() > True

    with subtests.test("ge"), pytest.raises(
        TypeError,
        match=r"^'>=' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() >= True

    ts = Timestamp.zero()
    with subtests.test("iadd"), pytest.raises(
        TypeError,
        match=r"^'\+=' not supported between instances of 'Timestamp' and '.*'$",
    ):
        ts += True

    with subtests.test("sub"), pytest.raises(
        TypeError,
        match=r"^'-' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() - True

    with subtests.test("truediv"), pytest.raises(
        TypeError,
        match=r"^'/' not supported between instances of 'Timestamp' and '.*'$",
    ):
        assert Timestamp.zero() / True


def test_timedelta(subtests: SubTests) -> None:
    raw_data: list[timedelta] = [
        timedelta(),
        timedelta(seconds=132112411),
        timedelta(seconds=32142),
    ]
    for data in raw_data:
        ts = Timestamp(data)
        with subtests.test("lossless storage"):
            assert ts.delta == data, "lossless storage in Timestamp"

        with subtests.test("seconds conversion"):
            assert (
                Timestamp.from_seconds(data.total_seconds()) == ts
            ), "seconds conversion is lossless"

        with subtests.test("minutes conversion"):
            assert (
                Timestamp.from_minutes(data.total_seconds() / 60).minutes == ts.minutes
            ), "Timestamp.from_minutes() is lossless"
            assert ts.minutes == (
                data.total_seconds() / 60.0 + (data.microseconds / (60.0 * 10**6))
            ), "minutes conversion is lossless"

        with subtests.test("!= comparison"):
            assert ts.delta != Timestamp(
                timedelta(seconds=212121),
            ), "!= comparison works"

        with subtests.test("total order"):
            local_ts = Timestamp(timedelta(seconds=12345))
            if ts >= local_ts:
                assert ts > local_ts or ts == local_ts, "total order of timestamps"
                assert not (ts < local_ts), "total order of timestamps"
            if ts <= local_ts:
                assert ts < local_ts or ts == local_ts, "total order of timestamps"
                assert not (ts > local_ts), "total order of timestamps"


def test_timedelta_zero() -> None:
    assert Timestamp.zero().delta == timedelta(seconds=0), "Timestamp.zero should be 0"
    assert (
        Timestamp.zero().delta.total_seconds() == 0
    ), "Timestamp.zero should be 0 seconds"


def test_timedelta_to_float() -> None:
    ts = Timestamp(timedelta(seconds=1314))
    assert float(ts) == ts.minutes, "conversion to float works"


def test_timedelta_abs() -> None:
    ts = Timestamp(timedelta(seconds=-1314))
    assert abs(ts) >= Timestamp.zero(), "abs() works"


def test_timedelta_iadd() -> None:
    r = 131241
    ts = Timestamp(timedelta(seconds=r))
    ts += timedelta(seconds=1)
    assert ts == Timestamp(timedelta(seconds=r + 1)), "+= works"
    ts += Timestamp(timedelta(seconds=1))
    assert ts == Timestamp(timedelta(seconds=r + 2)), "+= works"


def test_timedelta_add() -> None:
    r = 131241
    ts = Timestamp(timedelta(seconds=r))
    assert (ts + timedelta(seconds=1)) == Timestamp(timedelta(seconds=r + 1)), "+ works"


def test_timedelta_sub() -> None:
    r = 131241
    ts = Timestamp(timedelta(seconds=r))
    assert (ts - Timestamp(timedelta(seconds=1))) == Timestamp(
        timedelta(seconds=r - 1),
    ), "- works"
    assert (ts - 1 / 60.0) == Timestamp(timedelta(seconds=r - 1)), "- works"


def test_timedelta_truediv() -> None:
    r = 60
    ts = Timestamp(timedelta(seconds=r))
    assert (ts / Timestamp(timedelta(seconds=1))) == float(r), "/ works"
    assert (ts / (1 / 60.0)) == float(r), "/ works"


def test_timedelta_format() -> None:
    # TODO
    pass
