import json
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Annotated, Any, Optional, Self

import yaml
from apischema import ValidationError, deserialize, deserializer, schema, serializer
from prompt_toolkit.keys import KEY_ALIASES, Keys

from classifier import ClassifierOptionsConfig
from content.language_picker import (
    LanguagePickerConfig,
    NoLanguagePickerConfig,
)
from content.metadata.config import MetadataConfig
from content.metadata.interfaces import MissingProviderMetadataConfig
from content.scanner import ConfigScannerConfig, ScannerConfig
from helper.apischema import OneOf
from helper.log import get_logger


@dataclass
class GeneralConfig:
    target_file: str


@dataclass
class GeneralConfigParsed:
    target_file: Path


@dataclass
class ParserConfig:
    root_folder: str
    special: Optional[list[str]]
    video_formats: list[str] = field(
        default_factory=list,
        metadata=schema(min_items=1, unique=True),
    )
    ignore_files: list[str] = field(
        default_factory=list,
        metadata=schema(),
    )
    exception_on_error: Optional[bool] = field(metadata=schema(), default=True)


@dataclass
class ParserConfigParsed:
    root_folder: Path
    special: list[str]
    video_formats: list[str] = field(
        default_factory=list,
        metadata=schema(min_items=1, unique=True),
    )
    ignore_files: list[str] = field(
        default_factory=list,
        metadata=schema(),
    )
    exception_on_error: bool = field(metadata=schema(), default=True)


def one_of_list(values: list[str]) -> Callable[[dict[str, Any]], None]:

    def modify_schema(schema: dict[str, Any]) -> None:
        schema["enum"] = values

    return modify_schema


def get_all_keys_with_aliases() -> list[str]:
    result: list[str] = []

    # add all values  + human readbale enum names
    for key in Keys:
        result.append(key.value)
        result.append(key.name)

    result.extend(KEY_ALIASES.keys())

    return result


@schema(extra=one_of_list(get_all_keys_with_aliases()))
class CustomKey:
    __underlying: Keys

    def __init__(self: Self, key: Keys) -> None:
        self.__underlying = key

    @serializer
    def serialize(self: Self) -> str:
        return str(self)

    @staticmethod
    def __key_from_str(inp: str) -> Optional[Keys]:
        # resolve aliases
        for key, value in KEY_ALIASES.items():
            if key.lower() == inp.lower():
                inp = value.lower()

        for key in Keys:
            # resolve values
            if key.value.lower() == inp.lower():
                return key
            # resolve humand readbale enum format
            if key.name.lower() == inp.lower():
                return key

        return None

    @deserializer
    @staticmethod
    def deserialize_str(inp: str) -> "CustomKey":
        key = CustomKey.__key_from_str(inp)
        if key is None:
            msg = f"Deserialization error: invalid key string: {inp}"
            raise TypeError(msg)
        return CustomKey(key)

    def __str__(self: Self) -> str:
        return str(self.__underlying)

    def __repr__(self: Self) -> str:
        return repr(self.__underlying)

    @property
    def value(self: Self) -> Keys:
        return self.__underlying


@dataclass
class KeyBoardConfig:
    abort: CustomKey

    @staticmethod
    def default() -> "KeyBoardConfig":
        return KeyBoardConfig(abort=CustomKey(Keys.ControlG))


@dataclass
class ParsedConfig:
    general: GeneralConfigParsed
    parser: ParserConfigParsed
    scanner: ScannerConfig
    classifier: ClassifierOptionsConfig
    metadata: MetadataConfig
    picker: LanguagePickerConfig
    keybindings: KeyBoardConfig


logger: Logger = get_logger()


@dataclass
class Config:
    general: Annotated[Optional[GeneralConfig], OneOf]
    parser: Annotated[Optional[ParserConfig], OneOf]
    scanner: Annotated[Optional[ScannerConfig], OneOf]
    classifier: Annotated[Optional[ClassifierOptionsConfig], OneOf]
    metadata: Annotated[Optional[MetadataConfig], OneOf]
    picker: Annotated[Optional[LanguagePickerConfig], OneOf]
    keybindings: Annotated[Optional[KeyBoardConfig], OneOf]

    @staticmethod
    def __defaults() -> "ParsedConfig":
        return ParsedConfig(
            general=GeneralConfigParsed(target_file=Path("data.json")),
            parser=ParserConfigParsed(
                root_folder=Path.cwd(),
                special=[],
                video_formats=["mp4"],
                ignore_files=[],
                exception_on_error=True,
            ),
            scanner=ConfigScannerConfig(scanner_type="config", config=None),
            classifier=ClassifierOptionsConfig.default(),
            metadata=MissingProviderMetadataConfig(type="none"),
            picker=NoLanguagePickerConfig(picker_type="none"),
            keybindings=KeyBoardConfig.default(),
        )

    @staticmethod
    def fill_defaults(config: "Config") -> ParsedConfig:
        defaults = Config.__defaults()

        # TODO this is done manually atm, it can be done more automated, by checking for none on every key and replacing it with the key in defaults, if the key is none!
        parsed_general = defaults.general
        if config.general is not None:
            parsed_general = GeneralConfigParsed(
                target_file=Path(config.general.target_file),
            )

        parsed_parser = defaults.parser
        if config.parser is not None:
            parsed_parser = ParserConfigParsed(
                root_folder=Path(config.parser.root_folder),
                special=(
                    config.parser.special
                    if config.parser.special is not None
                    else defaults.parser.special
                ),
                video_formats=config.parser.video_formats,
                ignore_files=config.parser.ignore_files,
                exception_on_error=(
                    config.parser.exception_on_error
                    if config.parser.exception_on_error is not None
                    else defaults.parser.exception_on_error
                ),
            )

        parsed_scanner = defaults.scanner
        if config.scanner is not None:
            parsed_scanner = config.scanner

        parsed_classifier = defaults.classifier
        if config.classifier is not None:
            parsed_classifier = config.classifier

        parsed_metadata = defaults.metadata
        if config.metadata is not None:
            parsed_metadata = config.metadata

        parsed_picker = defaults.picker
        if config.picker is not None:
            parsed_picker = config.picker

        parsed_keybindings = defaults.keybindings
        if config.keybindings is not None:
            parsed_keybindings = config.keybindings

        return ParsedConfig(
            general=parsed_general,
            parser=parsed_parser,
            scanner=parsed_scanner,
            classifier=parsed_classifier,
            metadata=parsed_metadata,
            picker=parsed_picker,
            keybindings=parsed_keybindings,
        )

    @staticmethod
    def load(config_file: Path) -> Optional[ParsedConfig]:
        if config_file.exists():
            with config_file.open(mode="r") as file:
                suffix: str = config_file.suffix[1:]
                loaded_dict: dict[str, Any]
                match suffix:
                    case "json":
                        loaded_dict = json.load(file)
                    case "yml" | "yaml":
                        loaded_dict = yaml.safe_load(file)
                    case _:
                        msg = f"Config not loadable from '{suffix}' file!"
                        raise RuntimeError(msg)
                try:
                    parsed_dict: Config = deserialize(
                        Config,
                        loaded_dict,
                    )

                    return Config.fill_defaults(parsed_dict)
                except ValidationError as err:
                    msg = f"The config file {config_file} is invalid"
                    logger.error(msg=msg)  # noqa: TRY400
                    for error in err.errors:
                        loc = [str(s) for s in error["loc"]]
                        loc_pretty = ".".join(loc)
                        err_msg = error["err"]

                        msg = f"In location '{loc_pretty}': {err_msg}"
                        logger.error(msg)  # noqa: TRY400

                    return None

        msg = f"The config file {config_file} was not found"
        logger.error(msg)
        return None
