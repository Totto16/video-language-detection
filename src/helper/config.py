import json
import os
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Optional, Self, assert_never, override

import pydantic
import pydantic_core
import yaml
from apischema import (
    ValidationError,
    deserialize,
    deserializer,
    schema,
    serializer,
)
from apischema.metadata import none_as_undefined, required
from prompt_toolkit.keys import KEY_ALIASES, Keys

from content.extensions.extensions import ExtensionsConfig
from content.language_picker import (
    LanguagePickerConfig,
    NoLanguagePickerConfig,
)
from content.metadata.config import MetadataConfig
from content.metadata.interfaces import MissingProviderMetadataConfig
from content.scanner import ConfigScannerConfig, ScannerConfig
from helper.apischema import Deprecated, OneOf
from helper.classifier import ClassifierOptionsConfig
from helper.decorator import decorate_class
from helper.filter import ConfigFilter, Filter, SpecialFilter, SpecialFilterType
from helper.log import get_logger
from helper.result import Err, Ok, Result


@dataclass(slots=True, repr=True)
class TargetFileJson:
    type: Literal["json"]
    file: str


@dataclass(slots=True, repr=True)
class TargetFileWip:
    type: Literal["wip"]
    file: str


TargetFile = Annotated[
    TargetFileJson | TargetFileWip | Annotated[str, Deprecated],
    OneOf,
]


@dataclass(slots=True, repr=True)
class GeneralConfig:
    target_file: TargetFile


@dataclass(slots=True, repr=True)
class ParsedTargetFileJson:
    type: Literal["json"]
    file: Path


@dataclass(slots=True, repr=True)
class ParsedTargetFileWip:
    type: Literal["wip"]
    file: Path


ParsedTargetFile = Annotated[ParsedTargetFileJson | ParsedTargetFileWip, OneOf]


@dataclass(slots=True, repr=True)
class GeneralConfigParsed:
    target_file: ParsedTargetFile


@dataclass(slots=True, repr=True)
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


@dataclass(slots=True, repr=True)
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
            # resolve humand readable enum format
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

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        return pydantic_core.core_schema.literal_schema(
            get_all_keys_with_aliases(),
        )


@dataclass(slots=True, repr=True)
class KeyBoardConfig:
    abort: CustomKey

    @staticmethod
    def default() -> "KeyBoardConfig":
        return KeyBoardConfig(abort=CustomKey(Keys.ControlG))


class ConfigType(Enum):
    normal = "normal"
    numerated = "numerated"
    symlinked = "symlinked"


@dataclass(slots=True, repr=True)
class FinalConfig:
    config_name: str
    config_type: ConfigType
    general: GeneralConfigParsed
    parser: ParserConfigParsed
    scanner: ScannerConfig
    classifier: ClassifierOptionsConfig
    metadata: MetadataConfig
    picker: LanguagePickerConfig
    keybindings: KeyBoardConfig
    extensions: ExtensionsConfig


logger: Logger = get_logger()


def parse_target_file(tgt: TargetFile) -> ParsedTargetFile:
    if isinstance(tgt, str):
        return ParsedTargetFileJson("json", file=Path(tgt))

    if isinstance(tgt, TargetFileJson):
        return ParsedTargetFileJson("json", Path(tgt.file))

    if isinstance(tgt, TargetFileWip):
        return ParsedTargetFileWip("wip", Path(tgt.file))

    assert_never(tgt)


@dataclass(slots=True, repr=True)
class ConfigGenericV2:
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
    extensions: Annotated[Optional[ExtensionsConfig], OneOf] = field(
        default=None,
        metadata=none_as_undefined,
    )

    @staticmethod
    def __defaults() -> "FinalConfig":
        return FinalConfig(
            config_name="<None>",
            config_type=ConfigType.normal,
            general=GeneralConfigParsed(
                target_file=ParsedTargetFileJson("json", Path("data.json")),
            ),
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
            extensions=ExtensionsConfig.default(),
        )

    @staticmethod
    def fill_defaults(configs: "ConfigV2 | list[ConfigV2]") -> list[FinalConfig]:

        def fill_one_default(config: "ConfigV2") -> FinalConfig:
            defaults = ConfigV2.__defaults()  # noqa: SLF001

            # TODO this is done manually atm, it can be done more automated, by checking for none on every key and replacing it with the key in defaults, if the key is none!
            parsed_general = defaults.general
            if config.general is not None:
                target_file = parse_target_file(config.general.target_file)

                parsed_general = GeneralConfigParsed(
                    target_file=target_file,
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

            parsed_extensions = defaults.extensions
            if config.extensions is not None:
                parsed_extensions = config.extensions

            return FinalConfig(
                config_name=config.config_name,
                config_type=config.config_type,
                general=parsed_general,
                parser=parsed_parser,
                scanner=parsed_scanner,
                classifier=parsed_classifier,
                metadata=parsed_metadata,
                picker=parsed_picker,
                keybindings=parsed_keybindings,
                extensions=parsed_extensions,
            )

        results: list[FinalConfig] = []

        if isinstance(configs, list):
            results.extend(fill_one_default(config) for config in configs)
        else:
            results.append(fill_one_default(configs))

        return results


@dataclass(slots=True, repr=True)
class TemplateConfig(ConfigGenericV2):
    pass


@dataclass(slots=True, repr=True)
class ConfigV2(ConfigGenericV2):
    version: Literal["2"] = field(metadata=required, default="2")
    config_name: str = field(metadata=required, default="<ERROR>")
    config_type: ConfigType = field(metadata=required, default=ConfigType.normal)


UseFromCLI = Annotated[
    Literal[True],
    schema(description="Get the name of the config template to use from the cli"),
]


@dataclass(slots=True, repr=True)
class ConfigTemplateSettings:
    prefer_cli_template: Optional[bool]


@dataclass(slots=True, repr=True)
class ConfigTemplatesV2:
    defaults: list[ConfigV2] | ConfigV2
    names: dict[str, TemplateConfig]
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


@dataclass(slots=True, repr=True)
class ConfigTemplateV2:
    version: Literal["2"]
    templates: ConfigTemplatesV2


RawConfig = ConfigV2 | list[ConfigV2] | ConfigTemplateV2


SchemaConfig = Annotated[RawConfig, OneOf]


@dataclass(slots=True, repr=True)
class AdvancedConfig:

    @staticmethod
    def __load(config_file: Path) -> Optional[RawConfig]:
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
        template: TemplateConfig,
    ) -> Result[list[FinalConfig], str]:

        def merge_template(
            default_raw: FinalConfig,
            template: TemplateConfig,
        ) -> Optional[FinalConfig]:
            defaults = default_raw

            # TODO this is done manually atm, it can be done more automated
            parsed_general = defaults.general
            if template.general is not None:
                target_file = parse_target_file(template.general.target_file)

                parsed_general = GeneralConfigParsed(
                    target_file=target_file,
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

            parsed_extensions = defaults.extensions
            if template.extensions is not None:
                parsed_extensions = template.extensions

            return FinalConfig(
                config_name=defaults.config_name,
                config_type=defaults.config_type,
                general=parsed_general,
                parser=parsed_parser,
                scanner=parsed_scanner,
                classifier=parsed_classifier,
                metadata=parsed_metadata,
                picker=parsed_picker,
                keybindings=parsed_keybindings,
                extensions=parsed_extensions,
            )

        results: list[FinalConfig] = []

        for default_raw in defaults_raw:
            result: Optional[FinalConfig] = merge_template(
                default_raw=default_raw,
                template=template,
            )
            if result is None:
                return Err(default_raw.config_name)

            results.append(result)

        return Ok(results)

    @staticmethod
    def __resolve_template_to_use(
        templates: ConfigTemplatesV2,
        cli_name_to_use: Optional[str],
    ) -> tuple[TemplateConfig, str]:
        all_names: dict[str, TemplateConfig] = templates.names
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

        # if the settings say, that the cli one is preferred, we use that one, if it is set
        if (
            templates.settings is not None
            and templates.settings.prefer_cli_template
            and cli_name_to_use is not None
        ):
            name_to_use = cli_name_to_use

        template_to_use: Optional[TemplateConfig] = None

        aliases: dict[str, str] = {} if templates.aliases is None else templates.aliases

        while True:
            temp_config = all_names.get(name_to_use)

            # if the current name is present, we use that config
            if temp_config is not None:
                template_to_use = temp_config
                break

            # we try to look up an alias, if we find one we repeat the loop, so that redirecting aliases are allowed
            temp_name = aliases.get(name_to_use)
            if temp_name is not None:
                name_to_use = temp_name
                continue

            # otherwise we can't find the name in the config ands also no alias, so print an error and exit with a failure
            msg = f"No template or alias with name '{name_to_use}' was found"
            logger.error(msg)
            available_names = f"Available names are: {", ".join([*aliases.keys(), *all_names.keys()])}"
            logger.info(available_names)
            sys.exit(1)

        return (template_to_use, name_to_use)

    @staticmethod
    def __resolve_advance_config(
        config: ConfigTemplateV2,
        cli_name_to_use: Optional[str],
    ) -> Result[tuple[list[FinalConfig], str], str]:
        templates = config.templates

        defaults: list[FinalConfig] = ConfigV2.fill_defaults(templates.defaults)

        template_to_use, name_used = AdvancedConfig.__resolve_template_to_use(
            templates,
            cli_name_to_use,
        )

        final_configs = AdvancedConfig.__merge_templates(defaults, template_to_use)

        if final_configs.err():
            return Err(
                f"Error in config '{final_configs.as_err()}'",
            )

        return Ok(
            (
                final_configs.as_ok(),
                f"merging default config and user provided config '{name_used}'",
            ),
        )

    @staticmethod
    def load_and_resolve_with_info(
        config_file: Path,
        cli_name_to_use: Optional[str],
    ) -> Result[tuple[list[FinalConfig], str], str]:
        config = AdvancedConfig.__load(config_file)

        if config is None:
            return Err("No config loaded")

        return AdvancedConfig.resolve_raw_with_info(config, cli_name_to_use)

    @staticmethod
    def resolve_raw_with_info(
        raw_config: RawConfig,
        cli_name_to_use: Optional[str],
    ) -> Result[tuple[list[FinalConfig], str], str]:

        if isinstance(raw_config, ConfigV2):
            final_config = ConfigV2.fill_defaults(raw_config)
            return Ok(
                (final_config, "Normal Config"),
            )
        if isinstance(raw_config, list):
            final_config = ConfigV2.fill_defaults(raw_config)
            return Ok(
                (final_config, "Normal Configs"),
            )
        if isinstance(raw_config, ConfigTemplateV2):
            resolved_config: Result[tuple[list[FinalConfig], str], str] = (
                AdvancedConfig.__resolve_advance_config(
                    raw_config,
                    cli_name_to_use,
                )
            )

            if resolved_config.err():
                return Err(
                    resolved_config.as_err(),
                )

            merged_config, msg = resolved_config.as_ok()

            return Ok(
                (merged_config, f"Templated Config created by {msg}"),
            )
        assert_never(raw_config)

    @staticmethod
    def load_raw(
        config_file: Path,
    ) -> Optional[RawConfig]:
        return AdvancedConfig.__load(config_file)

    @staticmethod
    def load_and_resolve(
        config_file: Path,
        cli_name_to_use: Optional[str],
    ) -> Result[list[FinalConfig], str]:
        res = AdvancedConfig.load_and_resolve_with_info(config_file, cli_name_to_use)

        if res.err():
            return Err(res.as_err())

        return Ok(res.as_ok()[0])

    @staticmethod
    def resolve_raw(
        raw_config: RawConfig,
        cli_name_to_use: Optional[str],
    ) -> Result[list[FinalConfig], str]:
        res = AdvancedConfig.resolve_raw_with_info(raw_config, cli_name_to_use)

        if res.err():
            return Err(res.as_err())

        return Ok(res.as_ok()[0])


def __get_default_configs_impl(configs: list[FinalConfig]) -> list[FinalConfig]:
    # TODO: once numerated configs are fully implemented, use them here too

    return [cfg for cfg in configs if cfg.config_type == ConfigType.normal]


def __filter_configs_impl(
    configs: list[FinalConfig],
    cfg_filter: list[ConfigFilter | SpecialFilter],
) -> list[FinalConfig]:
    if len(cfg_filter) == 0:
        return __get_default_configs_impl(configs)

    def is_valid_name(name: str) -> tuple[bool, int]:
        for idx, cfg in enumerate(configs):
            if name == cfg.config_name:
                return (True, idx)

        return (False, -1)

    def to_dict(lst: list[FinalConfig]) -> dict[str, FinalConfig]:
        return {cfg.config_name: cfg for cfg in lst}

    result: dict[str, FinalConfig] = {}

    for filter_item in cfg_filter:
        if isinstance(filter_item, SpecialFilter):
            if filter_item.name == ConfigFilter.factory_name():
                match filter_item.type:
                    case SpecialFilterType.All:
                        result = to_dict(configs)
                    case SpecialFilterType.Empty:
                        result = {}
                    case SpecialFilterType.Default:
                        result = to_dict(__get_default_configs_impl(configs))
                    case _:
                        assert_never(filter_item.type)
        elif isinstance(filter_item, ConfigFilter):
            val = filter_item.value

            cfg: FinalConfig
            if isinstance(val, int):
                if val < 0 or val >= len(configs):
                    msg = f"Config Filter index is out of bounds, expected >= 0 and < {len(configs)} but got {val}"
                    raise RuntimeError(msg)
                cfg = configs[val]
            elif isinstance(val, str):
                valid_name, idx = is_valid_name(val)
                if not valid_name:
                    msg = f"Config filter name is invalid: '{val}'"
                    raise RuntimeError(msg)
                cfg = configs[idx]
            else:
                assert_never(val)

            cfg_name = cfg.config_name
            if result.get(cfg_name, None) is not None:
                msg = f"Config is already present, duplicate is not allowed: {cfg_name}"
                raise RuntimeError(msg)

            result[cfg_name] = cfg
        else:
            assert_never(filter_item)

    return list(result.values())


def filter_configs(
    configs: list[FinalConfig],
    filters: list[Filter],
) -> list[FinalConfig]:
    cfg_filter: list[ConfigFilter | SpecialFilter] = [
        filter_val
        for filter_val in filters
        if isinstance(filter_val, (ConfigFilter, SpecialFilterType))
    ]

    return __filter_configs_impl(configs, cfg_filter)


class FileLockError(RuntimeError):

    def __init__(self: Self, msg: str) -> None:
        super().__init__(msg)


@decorate_class(slots=True)
class LockFile(AbstractContextManager[None]):
    __lock_file: Path
    __fd: Optional[int]

    def __init__(self: Self, lock_file: Path) -> None:
        super().__init__()
        self.__lock_file = lock_file
        self.__fd = None

    @staticmethod
    def for_file(file: Path) -> "LockFile":
        lock_file = file.parent / (file.stem + ".lock")
        return LockFile(lock_file)

    def __create_lock_file(self: Self) -> None:
        try:
            self.__fd = os.open(
                self.__lock_file,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )

            unique_data = (
                f"PID: {os.getpid()} TID: {threading.current_thread().native_id}"
            )

            os.write(self.__fd, unique_data.encode())

        except FileExistsError:
            msg = "lock for file already held"
            raise FileLockError(msg) from None

    def __remove_lock_file(self: Self) -> None:
        if self.__fd is not None:
            os.close(self.__fd)
            self.__fd = None
            self.__lock_file.unlink()

    @override
    def __enter__(self: Self) -> None:
        self.__create_lock_file()

    @override
    def __exit__(
        self: Self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:  # actually bool
        self.__remove_lock_file()
        return False
