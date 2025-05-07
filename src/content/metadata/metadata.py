from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Self

from apischema import alias, deserializer, schema, serialize, serializer


@dataclass(slots=True, repr=True)
@schema()
class MetadataHandleHelper:
    provider: str
    data: Any


def generate_provider_schema() -> Callable[[dict[str, Any]], None]:
    def make_provider_schema(schema: dict[str, Any]) -> None:
        from content.metadata.provider.imdb import IMDBProvider
        from content.metadata.provider.tmdb import TMDBProvider

        if TYPE_CHECKING:
            from content.general import SchemaType
            from content.metadata.interfaces import Provider

        providers: list[type[Provider]] = [
            IMDBProvider,
            TMDBProvider,
        ]

        provider_schemas: list[SchemaType] = [
            provider.get_metadata_schema() for provider in providers
        ]

        schema["oneOf"] = provider_schemas

    return make_provider_schema


@schema(extra=generate_provider_schema())
class MetadataHandle:
    __provider: str = field(metadata=alias("provider"))
    __data: Any = field(metadata=alias("data"))

    def __init__(self: Self, provider: str, data: Any) -> None:
        self.__provider = provider
        self.__data = data

    @property
    def provider(self: Self) -> str:
        return self.__provider

    @property
    def data(self: Self) -> Any:
        return self.__data

    @serializer
    def serialize(self: Self) -> dict[str, Any]:
        serialized_dict: dict[str, Any] = serialize(
            MetadataHandleHelper,
            MetadataHandleHelper(self.__provider, self.__data),
        )
        return serialized_dict

    @deserializer
    @staticmethod
    def deserialize_handle(data_dict: Any) -> "MetadataHandle":
        if not isinstance(data_dict, dict):
            msg = "Deserialization error: expected input to be dict"
            raise TypeError(msg)

        if data_dict.get("provider", None) is None:
            msg = "Deserialization error: missing property 'provider'"
            raise TypeError(msg)

        if data_dict.get("data", None) is None:
            msg = "Deserialization error: missing property 'data'"
            raise TypeError(msg)

        provider = data_dict["provider"]
        data = data_dict["data"]

        if not isinstance(provider, str):
            msg = "Deserialization error: property 'provider' is not a str"
            raise TypeError(msg)

        if not isinstance(data, dict):
            msg = "Deserialization error: property 'data' is not a dict"
            raise TypeError(msg)

        match provider:
            case "tmdb":
                from content.metadata.provider.tmdb import TMDBProvider

                return MetadataHandle(provider, TMDBProvider.deserialize_metadata(data))
            case "imdb":
                msg = f"Deserialization error: Not implemented for provider {provider}"
                raise RuntimeError(msg)
            case _:
                msg = f"Deserialization error: Unknown provider {provider}"
                raise TypeError(msg)


class SkipHandle:
    # serialize the same as None
    @serializer
    def serialize(self: Self) -> None:
        return None


HandlesType = Optional[list[MetadataHandle] | SkipHandle]

InternalMetadataType = Optional[MetadataHandle | SkipHandle]
