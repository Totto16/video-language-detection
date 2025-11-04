import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

from config import SchemaConfig
from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from helper.apischema import EmitType, OneOf, get_schema
from helper.translation import get_translator

if TYPE_CHECKING:
    from collections.abc import Mapping


_ = get_translator()


AllContent = Annotated[
    EpisodeContent | SeasonContent | SeriesContent | CollectionContent,
    OneOf,
]


def generate_schema(
    file_path: Path,
    any_type: Any,
    *,
    emit_type: Optional[EmitType] = None,
) -> None:
    result: Mapping[str, Any] = get_schema(
        any_type,
        additional_properties=False,
        all_refs=True,
        emit_type=emit_type,
    )

    if not file_path.parent.exists():
        Path(file_path).parent.mkdir(parents=True)

    with file_path.open(mode="w") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def generate_schemas(folder: Path) -> None:
    generate_schema(
        folder / "content_list_schema.json",
        list[AllContent],
        emit_type="deserialize",
    )
    generate_schema(
        folder / "config_schema.json",
        SchemaConfig,
        emit_type="deserialize",
    )
