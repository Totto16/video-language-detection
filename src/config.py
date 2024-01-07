import json
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import yaml
from apischema import ValidationError, deserialize, schema
from classifier import ClassifierOptionsParsed
from content.scanner import ConfigScannerConfig, ScannerConfig
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


@dataclass
class ParsedConfig:
    general: GeneralConfigParsed
    parser: ParserConfigParsed
    scanner: ScannerConfig
    classifier: ClassifierOptionsParsed


logger: Logger = get_logger()


@dataclass
class Config:
    general: Optional[GeneralConfig]
    parser: Optional[ParserConfig]
    scanner: Optional[ScannerConfig]
    classifier: Optional[ClassifierOptionsParsed]

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
            classifier=ClassifierOptionsParsed.default(),
        )

    @staticmethod
    def fill_defaults(config: "Config") -> ParsedConfig:
        defaults = Config.__defaults()

        # TODO this is done manually atm, ut can be done more automated, by checking for none on every key and replacing it with the key in defaults, if the key is none!
        parsed_general = defaults.general
        if config.general is not None:
            parsed_general = GeneralConfigParsed(
                target_file=Path(config.general.target_file),
            )

        parsed_parser = defaults.parser
        if config.parser is not None:
            parsed_parser = ParserConfigParsed(
                root_folder=Path(config.parser.root_folder),
                special=config.parser.special
                if config.parser.special is not None
                else defaults.parser.special,
                video_formats=config.parser.video_formats,
                ignore_files=config.parser.ignore_files,
                exception_on_error=config.parser.exception_on_error
                if config.parser.exception_on_error is not None
                else defaults.parser.exception_on_error,
            )

        parsed_scanner = defaults.scanner
        if config.scanner is not None:
            parsed_scanner = config.scanner

        parsed_classifier = defaults.classifier
        if config.classifier is not None:
            parsed_classifier = config.classifier

        return ParsedConfig(
            general=parsed_general,
            parser=parsed_parser,
            scanner=parsed_scanner,
            classifier=parsed_classifier,
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
