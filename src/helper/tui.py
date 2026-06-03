from logging import Logger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from prompt_toolkit.key_binding import KeyBindings

from content.base_class import LanguageScanner, Scanner
from content.language_picker import (
    LanguagePicker,
    TUIChoiceManager,
    get_picker_from_config,
)
from content.metadata.config import get_metadata_scanner_from_config
from content.summary import Summary
from helper.base import AnyType, parse_contents
from helper.classifier import Classifier, Model
from helper.devices import DeviceManager
from helper.error import ErrorModeFile
from helper.manager import TuiManager
from helper.models import voxlingua107_ecapa_model

if TYPE_CHECKING:
    from content.base_class import Content
    from content.metadata.scanner import MetadataScanner

from content.general import NameParser
from content.scanner import (
    ConfigScanner,
    get_scanner_from_config,
)
from helper.config import FinalConfig, KeyBoardConfig
from helper.translation import get_translator

_ = get_translator()


def get_keybindings(
    logger: Logger,
    kb_config: KeyBoardConfig,
    scanner: Scanner,
) -> KeyBindings:
    kb = KeyBindings()

    @kb.add(kb_config.abort.value)
    def _abort_callback(_event: Any) -> None:
        if isinstance(scanner, ConfigScanner) and scanner.allow_abort:
            scanner.abort()
            logger.info(_("Aborted scan"))

    return kb


def launch_tui(
    logger: Logger,
    config: FinalConfig,
    name_parser: NameParser,
    all_content_type: AnyType,
    config_paramaters: Optional[tuple[int, int]],
) -> None:
    device_manager: DeviceManager = DeviceManager()

    model: Model = voxlingua107_ecapa_model

    classifier = Classifier(
        device_manager=device_manager,
        model=model,
        options=config.classifier,
    )
    language_scanner = LanguageScanner(classifier=classifier)
    metadata_scanner: MetadataScanner = get_metadata_scanner_from_config(
        config.metadata,
    )
    scanner: Scanner = get_scanner_from_config(
        config.scanner,
        language_scanner,
        metadata_scanner=metadata_scanner,
    )

    choice_manager = TUIChoiceManager()

    # TODO: make configurable
    error_mode = ErrorModeFile(Path("error.log"))

    language_picker: LanguagePicker = get_picker_from_config(
        config=config.picker,
        choice_manager=choice_manager,
    )

    # TODO: this doesn't work atm
    # this is also unnecessary complicated for a tui app, do this in the gui instead
    _kb: KeyBindings = get_keybindings(logger, config.keybindings, scanner)

    general_info: list[str] = [
        x
        for x in [
            f"Config: {config.config_name}",
            (
                None
                if config_paramaters is None
                else f"Config progress: {config_paramaters[0]+1} / {config_paramaters[1]}"
            ),
            f"Config type: {config.config_type.value}",
        ]
        if x is not None
    ]

    manager = TuiManager()

    contents: list[Content] = parse_contents(
        root_folder=config.parser.root_folder,
        options={
            "ignore_files": config.parser.ignore_files,
            "video_formats": config.parser.video_formats,
            "trailer_names": config.parser.trailer_names,
            "parse_error_is_exception": config.parser.exception_on_error,
        },
        save_file=config.general.target_file,
        name_parser=name_parser,
        scanner=scanner,
        language_picker=language_picker,
        all_content_type=all_content_type,
        general_info=general_info,
        config_type=config.config_type,
        manager=manager,
        error_mode=error_mode,
    )

    # TODO:validators
    # language validators, check if the language is a correct one
    # metadata checks, check if no duplicates are found, missing episodes, missing seasons
    # check langauge consistency

    # TODO: find duplicates, e.g. simpson s32e10
    # TODO: also display missing episodes / episodes with the "wrong" language etc
    # should i do that in summary generation or when?

    language_summary, metadata_summary = Summary.combine_summaries(
        content.summary() for content in contents
    )

    scan_summary = language_scanner.summary_manager.get_detailed_summary()

    logger.info(language_summary)
    logger.info(metadata_summary)
    logger.info(scan_summary)
