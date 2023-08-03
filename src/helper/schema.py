import json
from dataclasses import dataclass, field
from os import makedirs
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from apischema import alias
from apischema.json_schema import (
    deserialization_schema,
    serialization_schema,
)
from classifier import Language
from content.base_class import Content
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.general import (
    CollectionDescription,
    ContentType,
    EpisodeDescription,
    ScannedFile,
    SeasonDescription,
    SeriesDescription,
)
from content.season_content import SeasonContent
from content.series_content import SeriesContent

if TYPE_CHECKING:
    from collections.abc import Mapping


def generate_json_schema(file_path: Path, any_type: Any) -> None:
    result: Mapping[str, Any] = deserialization_schema(
        any_type,
        additional_properties=False,
        all_refs=True,
    )

    result2 = serialization_schema(any_type, additional_properties=False, all_refs=True)

    if result != result2:
        raise RuntimeError("Deserialization and Serialization scheme mismatch")

    if not file_path.parent.exists():
        makedirs(file_path.parent)

    with open(file_path, "w") as file:
        json_content: str = json.dumps(result, indent=4)
        file.write(json_content)


@dataclass(slots=True, repr=True)
class ContentSchema:
    type_: ContentType = field(metadata=alias("type"))
    scanned_file: ScannedFile


@dataclass(slots=True, repr=True)
class EpisodeContentSchema(ContentSchema):
    description: EpisodeDescription
    language: Language


@dataclass(slots=True, repr=True)
class SeasonContentSchema(ContentSchema):
    description: SeasonDescription
    episodes: list[EpisodeContentSchema]


@dataclass(slots=True, repr=True)
class SeriesContentSchema(ContentSchema):
    description: SeriesDescription
    seasons: list[SeasonContentSchema]


@dataclass(slots=True, repr=True)
class CollectionContentSchema(ContentSchema):
    description: CollectionDescription
    series: list[SeriesContentSchema]


AllContentSchemas = (
    EpisodeContentSchema
    | SeasonContentSchema
    | SeriesContentSchema
    | CollectionContentSchema
)


def convert_contents(
    inp: list[Content],
) -> list[AllContentSchemas]:
    def convert_one(ct: Content) -> ContentSchema:
        if isinstance(ct, CollectionContent):
            series = cast(list[SeriesContentSchema], list(map(convert_one, ct.series)))
            return CollectionContentSchema(
                ContentType.collection,
                ct.scanned_file,
                ct.description,
                series,
            )
        if isinstance(ct, SeriesContent):
            seasons = cast(
                list[SeasonContentSchema],
                list(map(convert_one, ct.seasons)),
            )
            return SeriesContentSchema(
                ContentType.series,
                ct.scanned_file,
                ct.description,
                seasons,
            )
        if isinstance(ct, SeasonContent):
            episodes = cast(
                list[EpisodeContentSchema],
                list(map(convert_one, ct.episodes)),
            )
            return SeasonContentSchema(
                ContentType.season,
                ct.scanned_file,
                ct.description,
                episodes,
            )
        if isinstance(ct, EpisodeContent):
            return EpisodeContentSchema(
                ContentType.episode,
                ct.scanned_file,
                ct.description,
                ct.language,
            )

        raise RuntimeError(f"Not recognized class {type(ct)}")

    return cast(list[AllContentSchemas], list(map(convert_one, inp)))


AllContent = EpisodeContent | SeasonContent | SeriesContent | CollectionContent


def convert_contents_reverse(
    inp: list[AllContentSchemas],
) -> list[AllContent]:
    def convert_one(ct: ContentSchema) -> Content:
        if isinstance(ct, CollectionContentSchema):
            series = cast(list[SeriesContent], list(map(convert_one, ct.series)))
            return CollectionContent(
                ContentType.collection,
                ct.scanned_file,
                ct.description,
                series,
            )
        if isinstance(ct, SeriesContentSchema):
            seasons = cast(
                list[SeasonContent],
                list(map(convert_one, ct.seasons)),
            )
            return SeriesContent(
                ContentType.series,
                ct.scanned_file,
                ct.description,
                seasons,
            )
        if isinstance(ct, SeasonContentSchema):
            episodes = cast(
                list[EpisodeContent],
                list(map(convert_one, ct.episodes)),
            )
            return SeasonContent(
                ContentType.season,
                ct.scanned_file,
                ct.description,
                episodes,
            )
        if isinstance(ct, EpisodeContentSchema):
            return EpisodeContent(
                ContentType.episode,
                ct.scanned_file,
                ct.description,
                ct.language,
            )

        raise RuntimeError(f"Not recognized class {type(ct)}")

    return cast(list[AllContent], list(map(convert_one, inp)))
