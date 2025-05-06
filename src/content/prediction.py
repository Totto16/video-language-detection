import math
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from math import floor
from typing import Optional, Self, TypedDict

from content.language import Language

PredictionType = list[tuple[float, Language]]


class LanguagePercentageDict(TypedDict):
    language: str
    score: float


@dataclass
class PredictionBest:
    accuracy: float
    language: Language

    def __str__(self: Self) -> str:
        return f"{self.language!s} ({self.accuracy:.2%})"

    def __repr__(self: Self) -> str:
        return (
            f"<PredictionBest accuracy: {self.accuracy!r} language: {self.language!r}>"
        )


TRUNCATED_PERCENTILE: float = 0.2


# see: https://en.wikipedia.org/wiki/Mean
class MeanType(Enum):
    arithmetic = "arithmetic"
    geometric = "geometric"
    harmonic = "harmonic"
    truncated = "truncated"


def get_mean(
    mean_type: MeanType,
    values: list[float],
    *,
    normalize_percents: bool = False,
) -> float:
    if normalize_percents:
        percent_mean: float = get_mean(
            mean_type,
            [value * 100.0 for value in values],
            normalize_percents=False,
        )
        return percent_mean / 100.0

    match mean_type:
        case MeanType.arithmetic:
            sum_value: float = sum(values)
            return sum_value / len(values)
        case MeanType.geometric:
            sum_value = reduce(lambda x, y: x * y, values, 1.0)
            return math.pow(sum_value, (1 / len(values)))
        case MeanType.harmonic:
            sum_value = sum(1 / value for value in values)
            return len(values) / sum_value
        case MeanType.truncated:
            start: int = floor(len(values) * TRUNCATED_PERCENTILE)
            end: int = len(values) - start
            new_values: list[float] = [
                value
                for i, value in enumerate(sorted(values))
                if i >= start and i < end
            ]
            return get_mean(
                MeanType.arithmetic,
                new_values,
                normalize_percents=normalize_percents,
            )


class Prediction:
    __data: list[PredictionType]

    def __init__(self: Self, data: Optional[PredictionType] = None) -> None:
        self.__data = []
        if data is not None:
            self.__data.append(data)

    def get_best_list(
        self: Self,
        mean_type: MeanType = MeanType.arithmetic,
    ) -> list[PredictionBest]:
        prob_dict: dict[Language, list[float]] = {}
        for data in self.__data:
            for acc, language in data:
                if prob_dict.get(language) is None:
                    prob_dict[language] = []

                prob_dict[language].append(acc)

        prob: PredictionType = []
        for lan, acc2 in prob_dict.items():
            prob.append((get_mean(mean_type, acc2), lan))

        sorted_prob: PredictionType = sorted(prob, key=lambda x: -x[0])

        return [PredictionBest(*sorted_prob_item) for sorted_prob_item in sorted_prob]

    def get_best(
        self: Self,
        mean_type: MeanType = MeanType.arithmetic,
    ) -> PredictionBest:
        best_list: list[PredictionBest] = self.get_best_list(mean_type)
        if len(best_list) == 0:
            return PredictionBest(0.0, Language.unknown())

        return best_list[0]

    @property
    def data(self: Self) -> list[PredictionType]:
        return self.__data

    def append(self: Self, data: PredictionType) -> None:
        self.__data.append(data)

    def append_other(self: Self, pred: "Prediction") -> None:
        self.__data.extend(pred.data)

    def __iadd__(self: Self, value: object) -> Self:
        if isinstance(value, Prediction):
            self.append_other(value)
            return self

        msg = f"'+=' not supported between instances of 'Prediction' and '{value.__class__.__name__}'"
        raise TypeError(msg)

    def __add__(self: Self, value: object) -> "Prediction":
        new_value = Prediction()
        new_value += self
        new_value += value
        return new_value
