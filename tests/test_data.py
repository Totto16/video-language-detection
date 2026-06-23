from collections.abc import Mapping
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytest_subtests import SubTests
from helper.apischema import get_schema
from test_helper import re_exact_string

from helper.base import load_from_file
from helper.config import ParsedTargetFileJson, SchemaConfig
from helper.custom_parser import CustomNameParser
from main import AllContent


def test_data_parse_fails(subtests: SubTests) -> None:

    with subtests.test("file doesn't exist"):

        def load() -> None:
            load_from_file(
                ParsedTargetFileJson("json", Path("not_present.json")),
                AllContent,
            )

        with pytest.raises(
            FileNotFoundError,
            match=re_exact_string(
                "[Errno 2] No such file or directory: 'not_present.json'",
            ),
        ):
            load()

    with subtests.test("invalid extension"):

        def load() -> None:
            with tempfile.NamedTemporaryFile(
                suffix=".txt",
            ) as f:
                load_from_file(ParsedTargetFileJson("json", Path(f.name)), AllContent)

        with pytest.raises(
            RuntimeError,
            match=re_exact_string("Data not loadable from 'txt' file!"),
        ):
            load()


def test_custom_name_parser(subtests: SubTests) -> None:
    season_special_names = ["Extras", "Specials", "Special"]

    with subtests.test("series name: invalid"):
        parser = CustomNameParser(season_special_names)

        series_data = parser.parse_series_name("Invalid")

        assert series_data is None

    with subtests.test("series name: valid"):
        parser = CustomNameParser(season_special_names)

        series_data = parser.parse_series_name("Cool Series (2042)")

        assert series_data is not None

        assert series_data[0] == "Cool Series"
        assert series_data[1] == 2042

    with subtests.test("season name: invalid"):
        parser = CustomNameParser(season_special_names)

        season_data = parser.parse_season_name("Invalid")

        assert season_data is None

    with subtests.test("season name: valid"):
        parser = CustomNameParser(season_special_names)

        season_data = parser.parse_season_name("Staffel 42")

        assert season_data is not None

        assert season_data[0] == 42

    with subtests.test("season name: special"):
        parser = CustomNameParser(season_special_names)

        season_data = parser.parse_season_name("Special")

        assert season_data is not None

        assert season_data[0] == 0

    with subtests.test("episode name: invalid"):
        parser = CustomNameParser(season_special_names)

        episode_data = parser.parse_episode_name("Invalid")

        assert episode_data is None

    with subtests.test("episode name: valid"):
        parser = CustomNameParser(season_special_names)

        episode_data = parser.parse_episode_name("Episode 01 - Cool Name [S09E01].mp4")

        assert episode_data is not None

        assert episode_data.episode == 1
        assert episode_data.name == "Cool Name"
        assert episode_data.season == 9


def generate_test_schema(any_type: Any) -> Mapping[str, Any]:
    result: Mapping[str, Any] = get_schema(
        any_type,
        additional_properties=False,
        all_refs=True,
        emit_type="deserialize",
    )

    return result


def read_json_file(file: Path) -> Any:
    with file.open(mode="r") as f:
        return json.load(f)


def test_schema_generation(subtests: SubTests) -> None:

    schema_folder = Path(__file__).parent.parent / "schema"

    with subtests.test("validate schema: content schema"):
        schema = generate_test_schema(
            list[AllContent],
        )

        schema_file = schema_folder / "content_list_schema.json"

        schema_content = read_json_file(schema_file)

        assert schema == schema_content

    with subtests.test("validate schema: config schema"):
        schema = generate_test_schema(
            SchemaConfig,
        )

        schema_file = schema_folder / "config_schema.json"

        schema_content = read_json_file(schema_file)

        assert schema == schema_content
