import json
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Self

import yaml
from apischema import ValidationError, deserialize, deserializer, schema, serializer
from apischema.metadata import none_as_undefined
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
from helper.result import Result
from helper.types import assert_never


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
    trailer_names: list[str] = field(
        default_factory=list,
        metadata=schema(),
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
    trailer_names: list[str] = field(
        default_factory=list,
        metadata=schema(),
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
class FinalConfig:
    config_name: str
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
    config_name: str
    general: Annotated[Optional[GeneralConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    parser: Annotated[Optional[ParserConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    scanner: Annotated[Optional[ScannerConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    classifier: Annotated[Optional[ClassifierOptionsConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    metadata: Annotated[Optional[MetadataConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    picker: Annotated[Optional[LanguagePickerConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )
    keybindings: Annotated[Optional[KeyBoardConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )

    @staticmethod
    def __defaults() -> "FinalConfig":
        return FinalConfig(
            config_name="<None>",
            general=GeneralConfigParsed(target_file=Path("data.json")),
            parser=ParserConfigParsed(
                root_folder=Path.cwd(),
                special=[],
                video_formats=["mp4"],
                trailer_names=[],
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
    def fill_defaults(configs: "Config | list[Config]") -> list[FinalConfig]:

        def fill_one_default(config: "Config") -> FinalConfig:
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
                    trailer_names=config.parser.trailer_names,
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

            return FinalConfig(
                config_name=config.config_name,
                general=parsed_general,
                parser=parsed_parser,
                scanner=parsed_scanner,
                classifier=parsed_classifier,
                metadata=parsed_metadata,
                picker=parsed_picker,
                keybindings=parsed_keybindings,
            )

        results: list[FinalConfig] = []

        if isinstance(configs, list):
            results.extend(fill_one_default(config) for config in configs)
        else:
            results.append(fill_one_default(configs))

        return results


UseFromCLI = Annotated[
    Literal[True],
    schema(description="Get the name of the config template to use from the cli"),
]


@dataclass
class ConfigTemplateSettings:
    prefer_cli_template: Optional[bool]


@dataclass
class ConfigTemplates:
    defaults: list[Config] | Config
    names: dict[str, Config]
    use: Optional[str | UseFromCLI] = field(
        default=None,
        metadata=none_as_undefined,
    )
    settings: Optional[ConfigTemplateSettings] = field(
        default=None,
        metadata=none_as_undefined,
    )
    aliases: Optional[dict[str, str]] = field(
        default=None,
        metadata=none_as_undefined,
    )


@dataclass
class ConfigTemplate:
    templates: ConfigTemplates


InternalConfig = Config | list[Config] | ConfigTemplate


SchemaConfig = Annotated[InternalConfig, OneOf]

AdvancedConfig__MergeResult = Result[list[FinalConfig], str]

AdvancedConfig__ResolveAdvancedConfig = Result[tuple[list[FinalConfig], str], str]

AdvancedConfig__LoadAndResolveWithInfo = Result[tuple[list[FinalConfig], str], str]

AdvancedConfig__LoadAndResolve = Result[list[FinalConfig], str]


@dataclass
class AdvancedConfig:

    @staticmethod
    def __load(config_file: Path) -> Optional[InternalConfig]:
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
                    return deserialize(
                        SchemaConfig,
                        loaded_dict,
                    )
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
        logger.error(msg=msg)
        return None

    @staticmethod
    def __merge_templates(
        defaults_raw: list[FinalConfig],
        template: Config,
    ) -> AdvancedConfig__MergeResult:

        def merge_template(
            default_raw: FinalConfig,
            template: Config,
        ) -> Optional[FinalConfig]:
            defaults = default_raw

            # TODO this is done manually atm, it can be done more automated
            parsed_general = defaults.general
            if template.general is not None:
                parsed_general = GeneralConfigParsed(
                    target_file=Path(template.general.target_file),
                )

            parsed_parser = defaults.parser
            if template.parser is not None:
                parsed_parser = ParserConfigParsed(
                    root_folder=Path(template.parser.root_folder),
                    special=(
                        template.parser.special
                        if template.parser.special is not None
                        else defaults.parser.special
                    ),
                    video_formats=template.parser.video_formats,
                    trailer_names=template.parser.trailer_names,
                    ignore_files=template.parser.ignore_files,
                    exception_on_error=(
                        template.parser.exception_on_error
                        if template.parser.exception_on_error is not None
                        else defaults.parser.exception_on_error
                    ),
                )

            parsed_scanner = defaults.scanner
            if template.scanner is not None:
                parsed_scanner = template.scanner

            parsed_classifier = defaults.classifier
            if template.classifier is not None:
                parsed_classifier = template.classifier

            parsed_metadata = defaults.metadata
            if template.metadata is not None:
                parsed_metadata = template.metadata

            parsed_picker = defaults.picker
            if template.picker is not None:
                parsed_picker = template.picker

            parsed_keybindings = defaults.keybindings
            if template.keybindings is not None:
                parsed_keybindings = template.keybindings

            return FinalConfig(
                config_name=defaults.config_name,
                general=parsed_general,
                parser=parsed_parser,
                scanner=parsed_scanner,
                classifier=parsed_classifier,
                metadata=parsed_metadata,
                picker=parsed_picker,
                keybindings=parsed_keybindings,
            )

        results: list[FinalConfig] = []

        for default_raw in defaults_raw:
            result: Optional[FinalConfig] = merge_template(
                default_raw=default_raw,
                template=template,
            )
            if result is None:
                return AdvancedConfig__MergeResult.err(default_raw.config_name)

            results.append(result)

        return AdvancedConfig__MergeResult.ok(results)

    @staticmethod
    def __resolve_config_to_use(
        templates: ConfigTemplates,
        cli_name_to_use: Optional[str],
    ) -> tuple[Config, str]:
        all_names: dict[str, Config] = templates.names
        if len(all_names) == 0:
            msg = "No template defined, define at least one template"
            raise TypeError(msg)

        name_to_use = templates.use

        # if no name is provided in the config, we try to use the cli one
        if name_to_use is None or (isinstance(name_to_use, bool) and name_to_use):
            if cli_name_to_use is None:
                msg = "Specified to get the template name from cli, but the cli didn't provide any value"
                raise TypeError(msg)
            name_to_use = cli_name_to_use

        # if the settings say, that the cli one is prefered, we use that one, if it is set
        if (
            templates.settings is not None
            and templates.settings.prefer_cli_template
            and cli_name_to_use is not None
        ):
            name_to_use = cli_name_to_use

        config_to_use: Optional[Config] = None

        aliases: dict[str, str] = {} if templates.aliases is None else templates.aliases

        while True:
            temp_config = all_names.get(name_to_use)

            # if the current name is present, we use that config
            if temp_config is not None:
                config_to_use = temp_config
                break

            # we try to look up an alias, if we find one we repeat the loop, so that redirecting aliases are allowed
            temp_name = aliases.get(name_to_use)
            if temp_name is not None:
                name_to_use = temp_name
                continue

            # otherwise we can't find the name in the conigd ands also no alias, so raise an error
            msg = f"No template or alias with name '{name_to_use}' was found"
            raise TypeError(msg)

        return (config_to_use, name_to_use)

    @staticmethod
    def __resolve_advance_config(
        config: ConfigTemplate,
        cli_name_to_use: Optional[str],
    ) -> AdvancedConfig__ResolveAdvancedConfig:
        templates = config.templates

        defaults: list[FinalConfig] = Config.fill_defaults(templates.defaults)

        config_to_use, name_used = AdvancedConfig.__resolve_config_to_use(
            templates,
            cli_name_to_use,
        )

        final_configs = AdvancedConfig.__merge_templates(defaults, config_to_use)

        if final_configs.is_err():
            return AdvancedConfig__ResolveAdvancedConfig.err(
                f"Error in config '{final_configs.get_err()}'",
            )

        return AdvancedConfig__ResolveAdvancedConfig.ok(
            (
                final_configs.get_ok(),
                f"merging default config and user provided config '{name_used}'",
            ),
        )

    @staticmethod
    def load_and_resolve_with_info(
        config_file: Path,
        cli_name_to_use: Optional[str],
    ) -> AdvancedConfig__LoadAndResolveWithInfo:
        config = AdvancedConfig.__load(config_file)

        if config is None:
            return AdvancedConfig__LoadAndResolveWithInfo.err("No config loaded")
        if isinstance(config, Config):
            final_config = Config.fill_defaults(config)
            return AdvancedConfig__LoadAndResolveWithInfo.ok(
                (final_config, "Normal Config"),
            )
        if isinstance(config, list):
            final_config = Config.fill_defaults(config)
            return AdvancedConfig__LoadAndResolveWithInfo.ok(
                (final_config, "Normal Configs"),
            )
        if isinstance(config, ConfigTemplate):
            resolved_config: Result[tuple[list[FinalConfig], str], str] = (
                AdvancedConfig.__resolve_advance_config(
                    config,
                    cli_name_to_use,
                )
            )

            if resolved_config.is_err():
                return AdvancedConfig__LoadAndResolveWithInfo.err(
                    resolved_config.get_err(),
                )

            merged_config, msg = resolved_config.get_ok()

            return AdvancedConfig__LoadAndResolveWithInfo.ok(
                (merged_config, f"Templated Config created by {msg}"),
            )
        assert_never(config)  # noqa: RET503

    @staticmethod
    def load_and_resolve(
        config_file: Path,
        cli_name_to_use: Optional[str],
    ) -> AdvancedConfig__LoadAndResolve:
        res = AdvancedConfig.load_and_resolve_with_info(config_file, cli_name_to_use)

        if res.is_err():
            return AdvancedConfig__LoadAndResolve.err(res.get_err())

        return AdvancedConfig__LoadAndResolve.ok(res.get_ok()[0])
