from typing import Annotated, cast

from content.general import OneOf
from content.metadata.interfaces import (
    MissingProvider,
    MissingProviderMetadataConfig,
    Provider,
)
from content.metadata.provider.imdb import IMDBMetadataConfig, IMDBProvider
from content.metadata.provider.tmdb import TMDBMetadataConfig, TMDBProvider
from content.metadata.scanner import MetadataScanner

MetadataConfig = Annotated[
    TMDBMetadataConfig | IMDBMetadataConfig | MissingProviderMetadataConfig,
    OneOf,
]


def __get_provider(config: MetadataConfig) -> Provider:
    match config.type:
        case "imdb":
            cfg1 = cast(IMDBMetadataConfig, config).config
            if cfg1 is not None:
                return IMDBProvider(config=cfg1)

            return MissingProvider()
        case "tmdb":
            cfg2 = cast(TMDBMetadataConfig, config).config
            if cfg2 is not None:
                return TMDBProvider(config=cfg2)

            return MissingProvider()

        case "none":
            return MissingProvider()


def get_metadata_scanner_from_config(config: MetadataConfig) -> MetadataScanner:
    provider = __get_provider(config)
    return MetadataScanner(provider)
