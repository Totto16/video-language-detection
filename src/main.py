import json
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import jsonschema.validators

from content.collection_content import CollectionContent
from content.episode_content import EpisodeContent
from content.season_content import SeasonContent
from content.series_content import SeriesContent
from helper.apischema import EmitType, OneOf, get_schema
from helper.config import SchemaConfig
import jsonschema

from helper.log import get_logger
from helper.translation import get_translator

if TYPE_CHECKING:
    from collections.abc import Mapping


AllContent = Annotated[
    EpisodeContent | SeasonContent | SeriesContent | CollectionContent,
    OneOf,
]

logger: Logger = get_logger()
_ = get_translator()


def validate_metaschema(file_path: Path) -> None:

    # trying to emulate: check_jsonschema
    # check-jsonschema --check-metaschema <file_path>

    try:

        meta_schema: Any
        with file_path.open("r") as f:
            meta_schema = json.load(f)

        validator = jsonschema.validators.validator_for(meta_schema)

        validator.check_schema(meta_schema)
    except jsonschema.SchemaError as e:
        logger.error(_("Invalid metaschema: {err}").format(err=e))  # noqa: TRY400


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

    validate_metaschema(file_path)


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
