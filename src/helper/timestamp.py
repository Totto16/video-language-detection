from datetime import datetime, timedelta
from typing import Never, Optional, Self

from apischema import deserializer, schema, serializer


def parse_int_safely(inp: str) -> Optional[int]:
    try:
        return int(inp)
    except ValueError:
        return None


# TODO check for overflow of hours everywhere!


@schema(pattern=r"^\d{1,2}:\d{1,2}:\d{1,2}$")
class Timestamp:
    __delta: timedelta

    def __init__(self: Self, delta: timedelta) -> None:
        self.__delta = delta

    @property
    def delta(self: Self) -> timedelta:
        return self.__delta

    @property
    def minutes(self: Self) -> float:
        return self.__delta.total_seconds() / 60.0 + (
            self.__delta.microseconds / (60.0 * 10**6)
        )

    @staticmethod
    def zero() -> "Timestamp":
        return Timestamp(timedelta(seconds=0))

    @staticmethod
    def from_minutes(minutes: float) -> "Timestamp":
        return Timestamp(timedelta(minutes=minutes))

    @staticmethod
    def from_seconds(seconds: float) -> "Timestamp":
        return Timestamp(timedelta(seconds=seconds))

    @serializer
    def serialize(self: Self) -> str:
        return str(self)

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "Timestamp":
        # https://stackoverflow.com/questions/4628122/how-to-construct-a-timedelta-object-from-a-simple-string
        t = datetime.strptime(inp, "%H:%M:%S").astimezone()
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        return Timestamp(delta)

    def __str__(self: Self) -> str:
        return str(self.__delta)

    def __repr__(self: Self) -> str:
        return str(self)

    def __format__(self: Self, spec: str) -> str:
        """This is responsible for formatting the Timestamp

        Args:
            spec (str): a format string, if empty or "d" no microseconds will be printed
                        otherwise you can provide an int from 0-5 inclusive, to determine how much you wan't to round
                        the microseconds, you can provide a "n" afterwards, to not have ".00". e.g , it ignores 0's with e.g. "2n"

        Examples:
            f"{timestamp:2}" => "00:01:00.20"
            f"{timestamp:4}" => "00:01:00.2010"

            f"{timestamp2:2}" => "00:05:00.00"
            f"{timestamp2:4}" => "00:05:00"
        """

        delta: timedelta = timedelta(
            seconds=int(self.__delta.total_seconds()),
            microseconds=0,
        )
        ms: int = self.__delta.microseconds

        if spec in ("", "d", "0n", "0"):
            return str(delta)

        def round_to_tens(value: int, tens: int) -> int:
            return int(round(value / (10**tens)))  # noqa: RUF046

        def emit_error(reason: str) -> Never:
            msg = f"Invalid format specifier '{spec}' for object of type 'Timestamp': reason {reason}"
            raise ValueError(msg)

        ignore_zero: bool = False
        if spec.endswith("n"):
            ignore_zero = True
            spec = spec[:-1]

        if ignore_zero and ms == 0:
            return str(delta)

        val: Optional[int] = parse_int_safely(spec)
        if val is None:
            msg = f"Couldn't parse int: '{spec}'"
            emit_error(msg)

        if val > 5 or val < 0:
            msg = f"{val} is out of allowed range 0 <= value <= 5"
            emit_error(msg)
        # val is between 1 and 5 inclusive
        ms = round_to_tens(ms, 6 - val)

        return "{delta}.{ms:0{val}d}".format(
            delta=str(delta),
            ms=ms,
            val=val,
        )

    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta == value.delta

        return False

    def __ne__(self: Self, value: object) -> bool:
        return not self.__eq__(value)

    def __lt__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta < value.delta

        msg = f"'<' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __le__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta <= value.delta

        msg = f"'<=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __gt__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta > value.delta

        msg = f"'>' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __ge__(self: Self, value: object) -> bool:
        if isinstance(value, Timestamp):
            return self.__delta >= value.delta

        msg = f"'>=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __iadd__(self: Self, value: object) -> Self:
        if isinstance(value, Timestamp):
            self.__delta += value.delta
            return self
        if isinstance(value, timedelta):
            self.__delta += value
            return self

        msg = f"'+=' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __add__(self: Self, value: object) -> "Timestamp":
        new_value: Timestamp = Timestamp(self.__delta)
        new_value += value
        return new_value

    def __sub__(self: Self, value: object) -> "Timestamp":
        if isinstance(value, Timestamp):
            result: timedelta = self.__delta - value.delta
            return Timestamp(result)
        if isinstance(value, float):
            result2: Timestamp = self - Timestamp.from_minutes(value)
            return result2

        msg = f"'-' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __abs__(self: Self) -> "Timestamp":
        return Timestamp(abs(self.__delta))

    def __float__(self: Self) -> float:
        return self.minutes

    def __truediv__(self: Self, value: object) -> float:
        if isinstance(value, Timestamp):
            return self.__delta / value.delta
        if isinstance(value, float):
            return self / Timestamp.from_minutes(value)

        msg = f"'/' not supported between instances of 'Timestamp' and '{value.__class__.__name__}'"
        raise TypeError(msg)


@schema(min=1, max=60 * 60 * 24, deprecated=True)
class TimestampFromSecond(Timestamp):
    @deserializer
    @staticmethod
    def deserialize_int(inp: int) -> "Timestamp":
        return Timestamp.from_seconds(inp)


ConfigTimeStamp = TimestampFromSecond | Timestamp
