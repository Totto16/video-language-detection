#!/usr/bin/env python3

import dataclasses as dc
from dataclasses import is_dataclass
from enum import Enum
from json import JSONDecoder, JSONEncoder
from pathlib import Path
from typing import (
    Any,
    Optional,
    Self,
    cast,
    get_type_hints,
)

from classifier import Language
from content.base_class import Content
from content.collection_content import CollectionContent, CollectionContentDict
from content.episode_content import EpisodeContent, EpisodeContentDict
from content.general import (
    ContentType,
    EpisodeDescription,
    ScannedFile,
    ScannedFileType,
    SeasonDescription,
    SeriesDescription,
    Stats,
    StatsDict,
)
from content.season_content import SeasonContent, SeasonContentDict
from content.series_content import SeriesContent, SeriesContentDict

VALUE_KEY: str = "__value__"
TYPE_KEY: str = "__type__"


class Encoder(JSONEncoder):
    def default(self: Self, o: Any) -> Any:
        if isinstance(o, Content):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
        if isinstance(o, ScannedFile):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.as_dict(self)}
        if isinstance(o, Path):
            return {TYPE_KEY: "Path", VALUE_KEY: str(o.absolute())}
        if isinstance(o, Enum):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: o.name}
        if isinstance(o, Stats):
            return {TYPE_KEY: "Stats", VALUE_KEY: o.as_dict()}
        if is_dataclass(o):
            return {TYPE_KEY: o.__class__.__name__, VALUE_KEY: dc.asdict(o)}

        return super().default(o)


class Decoder(JSONDecoder):
    def __init__(self: Self) -> None:
        def object_hook(dct: dict[str, Any] | Any) -> Any:
            if not isinstance(dct, dict):
                return dct

            _type: Optional[str] = cast(Optional[str], dct.get(TYPE_KEY))
            if _type is None:
                return dct

            def pipe(dct: Any, keys: list[str]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key in keys:
                    result[key] = object_hook(dct[key])

                return result

            value: Any = dct[VALUE_KEY]
            match _type:
                # Enums
                case "ScannedFileType":
                    return ScannedFileType(value)
                case "ContentType":
                    return ContentType(value)

                # dataclass
                case "ScannedFile":
                    return ScannedFile(**value)
                case "SeriesDescription":
                    return SeriesDescription(**value)
                case "SeasonDescription":
                    return SeasonDescription(**value)
                case "EpisodeDescription":
                    return EpisodeDescription(**value)
                case "Language":
                    return Language(**value)

                # other classes
                case "Stats":
                    stats_dict: StatsDict = cast(StatsDict, value)
                    return Stats.from_dict(stats_dict)
                case "Path":
                    path_value: str = cast(str, value)
                    return Path(path_value)

                # Content classes
                case "SeriesContent":
                    series_content_dict: SeriesContentDict = cast(
                        SeriesContentDict,
                        pipe(value, list(get_type_hints(SeriesContentDict).keys())),
                    )
                    return SeriesContent.from_dict(series_content_dict)
                case "CollectionContent":
                    collection_content_dict: CollectionContentDict = cast(
                        CollectionContentDict,
                        pipe(value, list(get_type_hints(CollectionContentDict).keys())),
                    )
                    return CollectionContent.from_dict(collection_content_dict)
                case "EpisodeContent":
                    episode_content_dict: EpisodeContentDict = cast(
                        EpisodeContentDict,
                        pipe(value, list(get_type_hints(EpisodeContentDict).keys())),
                    )
                    return EpisodeContent.from_dict(episode_content_dict)
                case "SeasonContent":
                    season_content_dict: SeasonContentDict = cast(
                        SeasonContentDict,
                        pipe(value, list(get_type_hints(SeasonContentDict).keys())),
                    )
                    return SeasonContent.from_dict(season_content_dict)

                # error
                case _:
                    raise TypeError(
                        f"Object of type {_type} is not JSON de-serializable",
                    )

        super().__init__(object_hook=object_hook)
