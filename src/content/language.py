from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class Language:
    short: str
    long: str

    @staticmethod
    def from_str(inp: str) -> Optional["Language"]:
        arr: list[str] = [a.strip() for a in inp.split(":")]
        if len(arr) != 2:
            return None

        return Language(arr[0], arr[1])

    @staticmethod
    def from_str_unsafe(inp: str) -> "Language":
        lan: Optional[Language] = Language.from_str(inp)
        if lan is None:
            msg = f"Couldn't get the Language from str '{inp}'"
            raise RuntimeError(msg)

        return lan

    # this is for episodes, that have no real language, for some special episods of some tv series
    @staticmethod
    def no_language() -> "Language":
        return Language("no_lang", "No Language")

    # TODO: get all languages somehow, from teh classifiere, that supports them all

    # note this is an implementation detail, that shoudl not leak
    @staticmethod
    def __unknown() -> "Language":
        return Language("un", "Unknown")

    @staticmethod
    def get_default() -> "Language":
        return Language.__unknown()

    @staticmethod
    def is_default_value(language: "Language") -> bool:
        return language == Language.__unknown()

    def __str__(self: Self) -> str:
        return self.long

    def __repr__(self: Self) -> str:
        return f"<Language short: {self.short!r} long: {self.long!r}>"

    def __hash__(self: Self) -> int:
        return hash((self.short, self.long))

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Language):
            return self.short == other.short and self.long == other.long

        return False
